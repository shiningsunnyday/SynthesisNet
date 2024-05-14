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
from synnet.utils.reconstruct_utils import lookup_skeleton_by_index, predict_skeleton

Population = List[Individual]


class GeneticSearch:

    def __init__(self, config: GeneticSearchConfig):
        self.config = config

        with open(config.background_set_file, "rb") as f:
            self.background_smiles = set(
                Chem.CanonSmiles(st.root.smiles)
                for st in pickle.load(f).keys()
            )

    def initialize(self, path: str) -> Population:
        cfg = self.config
        population = []
        df = pd.read_csv(path).sample(cfg.population_size, random_state=cfg.seed)
        for smiles in df["smiles"].tolist():
            fp = mol_fp(smiles, _nBits=cfg.fp_bits)
            if cfg.bt_ignore:
                bt = None
            else:
                index = predict_skeleton(smiles=None, fp=fp, max_num_rxns=cfg.max_num_rxns)
                sk = lookup_skeleton_by_index(index)
                bt = utils.skeleton_to_binary_tree(sk)
            population.append(Individual(fp=fp, bt=bt, smiles=smiles))
        return population

    def validate(self, population: Population):
        cfg = self.config
        for ind in population:
            assert ind.fp.shape == (cfg.fp_bits,)
            assert 2 <= ind.bt.number_of_nodes()
            assert utils.num_internal(ind.bt) <= cfg.max_num_rxns
            assert all((0 <= d <= 2) for _, d in ind.bt.out_degree())

    def evaluate_scores(self, population: Population, prefix) -> Dict[str, float]:
        scores = [ind.fitness for ind in population]
        scores = sorted(scores, reverse=True)
        metrics = {
            f"{prefix}/mean": np.mean(scores).item(),
            f"{prefix}/stdev": np.std(scores).item(),
        }
        for k in [10, 100]:
            metrics[f"{prefix}/mean_top{k}"] = np.mean(scores[:k]).item()
        for k in range(1, 4):
            metrics[f"{prefix}/top{k}"] = scores[k - 1]
        return metrics

    def evaluate(self, population: Population) -> Dict[str, float]:

        # Fitness
        metrics = self.evaluate_scores(population, prefix="scores")

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

    def choose_couples_and_analogs(
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

        choices = []
        for _ in range(cfg.offspring_size):
            i1, i2 = np.random.choice(indices, size=[2], replace=False, p=p)
            choices.append((population[i1], population[i2]))
        for _ in range(cfg.analog_size):
            i = np.random.choice(indices, size=[1], replace=False, p=p).item()
            choices.append((population[i],))
        return choices

    def crossover_and_mutate(self, parents: Tuple[Individual, Individual]) -> Individual:
        cfg = self.config

        # Crossover: random bit swap
        n = cfg.fp_bits
        k = np.random.normal(loc=(n / 2), scale=(n / 10), size=1)
        k = np.clip(k, a_min=(0.2 * n), a_max=(0.8 * n))
        mask = utils.random_bitmask(cfg.fp_bits, k=int(k))
        fp = np.where(mask, parents[0].fp, parents[1].fp)

        # Mutate: random bit flip
        if utils.random_boolean(cfg.fp_mutate_prob):
            mask = utils.random_bitmask(cfg.fp_bits, k=round(cfg.fp_bits * cfg.fp_mutate_frac))
            fp = np.where(mask, ~fp, fp)

        # Initialize bt
        index = predict_skeleton(smiles=None, fp=fp, max_num_rxns=cfg.max_num_rxns)
        sk = lookup_skeleton_by_index(index)
        bt = utils.skeleton_to_binary_tree(sk)

        return Individual(fp=fp, bt=bt)

    def analog_mutate(self, ind: Individual) -> Individual:
        cfg = self.config
        if cfg.bt_ignore:
            return Individual(fp=ind.fp.copy(), bt=None)
        bt = ind.bt.copy()

        # bt: random add or delete nodes
        num_edits = torch.randint(1, cfg.bt_mutate_edits + 1, size=[1]).item()
        for _ in range(num_edits):
            if utils.random_boolean(0.5):
                utils.random_add_leaf(bt, max_internal=cfg.max_num_rxns)
            else:
                utils.random_remove_leaf(bt)
    
        return Individual(fp=ind.fp.copy(), bt=bt)

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
        population = self.initialize(cfg.initialize_path)

        # Let's also log the seed stats
        fn(population, usesmiles=True)
        metrics = self.evaluate_scores(population, prefix="seeds")
        wandb.log({"generation": -1, **metrics}, step=0, commit=True)
    
        # Safety 
        for ind in population:
            ind.smiles = None
            ind.fitness = None

        # Track some stats
        early_stop_queue = collections.deque(maxlen=cfg.early_stop_patience)

        # Main loop
        for epoch in tqdm.trange(-1, cfg.generations, desc="Searching"):

            # Crossover & mutation
            if epoch >= 0:
                offsprings = []
                for parents in self.choose_couples_and_analogs(population, epoch):
                    if len(parents) == 2:
                        child = self.crossover_and_mutate(parents)
                    else:
                        assert len(parents) == 1
                        child = self.analog_mutate(parents[0])
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
