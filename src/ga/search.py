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

    def predict_bt(self, fp, top_k=[1]):
        cfg = self.config
        if cfg.bt_ignore:
            return None
        sk_index = predict_skeleton(smiles=None, fp=fp, max_num_rxns=cfg.max_num_rxns, top_k=top_k)
        if isinstance(sk_index, list):
            assert len(sk_index) == 1
            sk_index = sk_index[0]
        sk = lookup_skeleton_by_index(sk_index)
        return utils.skeleton_to_binary_tree(sk)

    def initialize(self, path: str) -> Population:
        cfg = self.config
        population = []
        df = pd.read_csv(path).sample(cfg.population_size, random_state=cfg.seed)
        for smiles in df["smiles"].tolist():
            fp = mol_fp(smiles, _nBits=cfg.fp_bits).astype(np.float32)
            bt = self.predict_bt(fp)
            population.append(Individual(fp=fp, bt=bt, smiles=smiles))
        return population

    def initialize_random(self) -> Population:
        cfg = self.config
        population = []
        for _ in range(cfg.population_size):
            fp = np.random.choice([0, 1], size=cfg.fp_bits).astype(np.float32)
            bt = self.predict_bt(fp)
            population.append(Individual(fp=fp, bt=bt))
        return population

    def validate(self, population: Population):
        cfg = self.config
        for ind in population:
            assert ind.fp.shape == (cfg.fp_bits,)
            assert set(np.unique(ind.fp)) == {0, 1}
            if cfg.bt_ignore:
                assert ind.bt is None
            else:
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
        if not self.config.bt_ignore:
            trees = [ind.bt for ind in population]
            metrics["trees/mean_size"] = np.mean([bt.number_of_nodes() for bt in trees]).item()
            metrics["trees/mean_depth"] = np.mean([len(dag_longest_path(bt)) for bt in trees]).item()
            metrics["trees/mean_internal"] = np.mean([utils.num_internal(bt) for bt in trees]).item()

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
        p = p / np.sum(p)

        parents = []
        for _ in range(cfg.offspring_size):
            i1, i2 = np.random.choice(indices, size=[2], replace=False, p=p)
            parents.append((population[i1], population[i2]))
        return parents

    def choose_references(
        self,
        population: Population,
        epoch: int,
    ) -> List[Individual]:
        population = sorted(population, key=(lambda x: x.fitness))  # ascending
        indices = np.arange(len(population))

        cfg = self.config
        p = indices + 10
        p = np.square(p) / np.sum(np.square(p))

        references = []
        for _ in range(cfg.analog_size):
            i = np.random.choice(indices, size=[1], replace=False, p=p).item()
            references.append(population[i])
        return references

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
            fp = np.where(mask, 1 - fp, fp)

        # Initialize bt
        k = torch.randint(1, cfg.bt_mutate_topk + 1, size=[1]).item()
        bt = self.predict_bt(fp, top_k=[k])

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

        if cfg.resume_path is not None:
            print("Initializing from checkpoint", cfg.resume_path)
            with open(cfg.resume_path, "rb") as f:
                population = pickle.load(f)
        
        elif cfg.initialize_path is None:
            print("Initializing random")
            population = self.initialize_random()

        else:
            print("Initializing from SMILES", cfg.initialize_path)
            population = self.initialize(cfg.initialize_path)

            # Let's also log the seed stats
            fn(population, usesmiles=True)
            metrics = self.evaluate_scores(population, prefix="seeds")
            wandb.log({"generation": -1, **metrics}, commit=True)
        
            # Safety 
            for ind in population:
                ind.smiles = None
                ind.fitness = None

        # Track some stats
        num_calls = 0
        history = collections.deque(maxlen=cfg.early_stop_patience)
        history.append(-1000)

        # Main loop
        for epoch in tqdm.trange(-1, cfg.generations, desc="Searching"):

            # Crossover & mutation
            if epoch >= 0:

                mutants = [] 
                references = self.choose_references(population, epoch)
                for ref in references:
                    mutants.append(self.analog_mutate(ref))
                fn(mutants)
                for ref, mut in zip(references, mutants):
                    mut.fp = mol_fp(mut.smiles, _nBits=cfg.fp_bits).astype(np.float32)

                offsprings = []
                for parents in self.choose_couples(population + mutants, epoch):
                    child = self.crossover_and_mutate(parents)
                    offsprings.append(child)
                
                if num_calls + len(offsprings) > cfg.max_oracle_calls:
                    leftover = cfg.max_oracle_calls - num_calls
                    offsprings = random.sample(offsprings, k=leftover)
                fn(offsprings)
                num_calls += len(offsprings)

                population = self.cull(population + offsprings)
            
            elif cfg.resume_path is not None:
                pass
            
            else:
                fn(population)
                num_calls += len(population)
            
            self.validate(population)  # sanity check

            # Scoring
            metrics = self.evaluate(population)

            # Logging
            if cfg.wandb:
                table = [[epoch, ind.smiles, ind.fitness] for ind in population]
                columns = ["generation", "smiles", "fitness"]
                metrics["smiles"] = wandb.Table(columns=columns, data=table)
                wandb.log({"generation": epoch, **metrics}, commit=True)
            if cfg.checkpoint_path is not None:
                with open(cfg.checkpoint_path, "wb") as f:
                    pickle.dump(population, f)

            # Early-stopping
            history.append(metrics["scores/mean"])
            if (
                cfg.early_stop
                and (epoch > cfg.early_stop_warmup)
                and (len(history) == cfg.early_stop_patience)
                and (history[-1] - history[0] < cfg.early_stop_delta)
            ):
                print("Early stopping.")
                break

            # Exhausted oracle calls 
            if num_calls == cfg.max_oracle_calls:
                print("Exhausted oracle calls")
                break

            # Snap fp to SMILES
            for ind in population:
                ind.fp = mol_fp(ind.smiles, _nBits=cfg.fp_bits).astype(np.float32)

        # Cleanup
        if cfg.wandb:
            wandb.finish()
