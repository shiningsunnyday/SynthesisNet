import collections
import itertools
import json
import pickle
import random
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import torch
import tqdm
import wandb
from networkx.algorithms.dag import dag_longest_path
from rdkit import Chem

from ga import utils
from ga.config import GeneticSearchConfig, Individual
from synnet.encoding.distances import _tanimoto_similarity
from synnet.encoding.fingerprints import mol_fp
from synnet.utils.data_utils import binary_tree_to_skeleton

Population = List[Individual]


class GeneticSearch:

    def __init__(self, config: GeneticSearchConfig):
        self.config = config

        with open(config.background_set_file, "rb") as f:
            self.background_smiles = set(
                Chem.CanonSmiles(st.root.smiles)
                for st in pickle.load(f).keys()
            )

    def initialize_random(self) -> Population:
        cfg = self.config
        population = []
        for _ in range(cfg.population_size):
            fp = np.random.choice([True, False], size=cfg.fp_bits)
            bt_size = torch.randint(cfg.bt_nodes_min, cfg.bt_nodes_max + 1, size=[1])
            bt = utils.random_binary_tree(bt_size.item())
            population.append(Individual(fp=fp, bt=bt))
        return population

    def initialize_load(self, path: str) -> Population:
        cfg = self.config
        population = []
        df = pd.read_csv(path).sample(cfg.population_size)
        for smiles in df["smiles"].tolist():
            fp = mol_fp(smiles, _nBits=cfg.fp_bits)
            bt_size = torch.randint(cfg.bt_nodes_min, cfg.bt_nodes_max + 1, size=[1])
            bt = utils.random_binary_tree(bt_size.item())
            population.append(Individual(fp=fp, bt=bt))
        return population

    def validate(self, population: Population):
        cfg = self.config
        for ind in population:
            assert ind.fp.shape == (cfg.fp_bits,)
            assert cfg.bt_nodes_min <= ind.bt.number_of_nodes() <= cfg.bt_nodes_max

    def evaluate(self, population: Population) -> Dict[str, float]:

        # Fitness
        scores = [ind.fitness for ind in population]
        scores.sort(reverse=True)
        metrics = {
            "scores/mean": np.mean(scores).item(),
            "scores/stdev": np.std(scores).item(),
        }
        for k in [10, 100]:
            metrics[f"scores/mean_top{k}"] = np.mean(scores[:k]).item()
        for k in range(1, 4):
            metrics[f"scores/top{k}"] = scores[k - 1]

        # Trees
        trees = [ind.bt for ind in population]
        metrics["trees/mean_size"] = np.mean([bt.number_of_nodes() for bt in trees]).item()
        metrics["trees/mean_depth"] = np.mean([len(dag_longest_path(bt)) for bt in trees]).item()

        # Diversity
        distances = []
        fps = [mol_fp(ind.smiles, _nBits=4096) for ind in population if ind.smiles is not None]
        for a, b in itertools.combinations(fps, r=2):
            d = 1 - _tanimoto_similarity(a, b)
            distances.append(d)
        metrics["diversity"] = np.mean(distances).item()

        # Population size
        N = len(population)
        metrics["population_size"] = N

        # Uniqueness
        unique = set(ind.smiles for ind in population if ind.smiles is not None)
        metrics["unique"] = len(unique) / N

        # Novelty
        if unique:
            metrics["novelty"] = len(unique - self.background_smiles) / len(unique)
        else:
            metrics["novelty"] = 0.0

        return metrics

    def cull(self, population: Population) -> Population:
        N = self.config.population_size

        filtered = []
        leftover = []
        seen_smiles = set()
        for ind in population:
            if (ind.smiles is not None) and (ind.smiles not in seen_smiles):
                filtered.append(ind)
                seen_smiles.add(ind.smiles)
            else:
                leftover.append(ind)
        filtered.sort(key=(lambda x: x.fitness), reverse=True)
        filtered = filtered[:N]

        # Add top individuals of leftover
        if len(filtered) < N:
            leftover.sort(key=(lambda x: x.fitness), reverse=True)
            filtered += leftover[:(N - len(filtered))]
            filtered.sort(key=(lambda x: x.fitness), reverse=True)

        return filtered

    def choose_couples(
        self,
        population: Population,
        epoch: int,
    ) -> List[Tuple[Individual, Individual]]:
        population = sorted(population, key=(lambda x: x.fitness))  # ascending
        indices = np.arange(len(population))

        cfg = self.config
        p = indices + 10
        if epoch < 0.8 * cfg.generations:
            p = p / np.sum(p)
        else:
            p = scipy.special.softmax(p)

        couples = []
        for _ in range(cfg.offspring_size):
            i1, i2 = np.random.choice(indices, size=[2], replace=False, p=p)
            couples.append((population[i1], population[i2]))
        return couples

    def crossover(self, parents: Tuple[Individual, Individual]) -> Individual:
        cfg = self.config

        # fp: random bit swap
        if not cfg.freeze_fp:
            n = cfg.fp_bits
            k = np.random.normal(loc=(n / 2), scale=(n / 10), size=1)
            k = np.clip(k, a_min=(0.2 * n), a_max=(0.8 * n))
            mask = utils.random_bitmask(cfg.fp_bits, k=int(k))
            fp = np.where(mask, parents[0].fp, parents[1].fp)
        else:
            fp = parents[0].fp

        # bt:
        if not cfg.freeze_bt:
            if cfg.bt_crossover == "graft":  # random subtree swap
                trees = [parents[0].bt, parents[1].bt]
                random.shuffle(trees)
                bt = utils.random_graft(
                    *trees,
                    min_nodes=cfg.bt_nodes_min,
                    max_nodes=cfg.bt_nodes_max,
                )
            elif cfg.bt_crossover == "inherit":  # inherit from dominant parent
                dominant = 0 if (k >= cfg.fp_bits / 2) else 1
                bt = parents[dominant].bt
            else:
                raise NotImplementedError()
        else:
            bt = parents[0].bt

        return Individual(fp=fp, bt=bt)

    def mutate(self, ind: Individual) -> Individual:
        cfg = self.config

        # fp: random bit flip
        fp = ind.fp
        if (not cfg.freeze_fp) and utils.random_boolean(cfg.fp_mutate_prob):
            mask = utils.random_bitmask(cfg.fp_bits, k=round(cfg.fp_bits * cfg.fp_mutate_frac))
            fp = np.where(mask, ~fp, fp)

        # bt: random add or delete nodes
        bt = ind.bt.copy()
        num_edits = 0 if cfg.freeze_bt else torch.randint(cfg.bt_mutate_edits + 1, size=[1]).item()
        for _ in range(num_edits):
            if bt.number_of_nodes() == cfg.bt_nodes_max:
                add = False
            elif bt.number_of_nodes() == cfg.bt_nodes_min:
                add = True
            else:
                add = utils.random_boolean(0.5)
            if add:
                utils.random_add_leaf(bt)
            else:
                utils.random_remove_leaf(bt)

        return Individual(fp=fp, bt=bt)

    def checkpoint(self, path: str, population: Population) -> None:
        ckpt = []
        for ind in population:
            sk = binary_tree_to_skeleton(ind.bt)
            ckpt.append({
                "smi": ind.smiles,
                "bt": nx.tree_data(sk.tree, sk.tree_root),
                "score": ind.fitness,
            })
        with open(path, "w+") as f:
            json.dump(ckpt, f)

    def optimize(self, fn: Callable[[Population], None]) -> None:
        """Runs a genetic search.

        Args:
            fn: a fitness function that populates the `fitness` and `skeleton`
                fields of an input list of individuals.

        Returns:
            None
        """

        cfg = self.config

        # Seeding
        pl.seed_everything(cfg.seed)

        # Initialize WandB
        if cfg.wandb:
            wandb.init(
                project=cfg.wandb_project,
                dir=cfg.wandb_dir,
                config=dict(cfg),
            )

        # Initialize population
        if cfg.initialize_path is None:
            population = self.initialize_random()
        else:
            population = self.initialize_load(cfg.initialize_path)

        # Track some stats
        early_stop_queue = collections.deque(maxlen=cfg.early_stop_patience)

        # Main loop
        for epoch in tqdm.trange(-1, cfg.generations, desc="Searching"):

            # Crossover & mutation
            if epoch >= 0:
                offsprings = []
                for parents in self.choose_couples(population, epoch):
                    child = self.crossover(parents)
                    child = self.mutate(child)
                    offsprings.append(child)
                fn(offsprings)
                population = self.cull(population + offsprings)
            else:
                fn(population)
            self.validate(population)  # sanity check

            # Scoring
            metrics = self.evaluate(population)

            # Logging
            if cfg.wandb:
                table = [[epoch, ind.smiles, ind.fitness] for ind in population]
                columns = ["generation", "smiles", "fitness"]
                metrics["smiles"] = wandb.Table(columns=columns, data=table)
                wandb.log({"generation": epoch, **metrics}, step=(1 + epoch), commit=True)
            if cfg.checkpoint_path is not None:
                self.checkpoint(cfg.checkpoint_path, population)

            # Early-stopping
            early_stop_queue.append(metrics["scores/mean"])
            if (
                (epoch > cfg.early_stop_warmup)
                and (len(early_stop_queue) == cfg.early_stop_patience)
                and (early_stop_queue[-1] - early_stop_queue[0] < cfg.early_stop_delta)
            ):
                break

        # Cleanup
        if cfg.wandb:
            wandb.finish()
