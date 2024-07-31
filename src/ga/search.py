import collections
import itertools
import json
import multiprocessing as mp
import pickle
import random
from functools import partial
from multiprocessing import Pool
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
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tdc import Oracle

from ga import utils
from ga.config import GeneticSearchConfig, Individual
from synnet.encoding.distances import _tanimoto_similarity
from synnet.encoding.fingerprints import mol_fp
from synnet.utils.data_utils import binary_tree_to_skeleton
from synnet.utils.reconstruct_utils import lookup_skeleton_by_index, predict_skeleton

Population = List[Individual]


def fetch_oracle(objective):
    if objective == "qed":
        # define the oracle function from the TDC
        return Oracle(name="QED")
    elif objective == "logp":
        # define the oracle function from the TDC
        return Oracle(name="LogP")
    elif objective == "jnk":
        # return oracle function from the TDC
        return Oracle(name="JNK3")
    elif objective == "gsk":
        # return oracle function from the TDC
        return Oracle(name="GSK3B")
    elif objective == "drd2":
        # return oracle function from the TDC
        return Oracle(name="DRD2")
    elif objective == "7l11":
        return Oracle(name="7l11_docking")
    elif objective == "drd3":
        return Oracle(name="drd3_docking")
    else:
        raise ValueError("Objective function not implemented")


class GeneticSearch:

    def __init__(self, config: GeneticSearchConfig):
        self.config = config

    def predict_bt(self, fp, top_k=[1]):
        cfg = self.config
        if cfg.bt_ignore:
            return None
        sk_indices = predict_skeleton(smiles=None, fp=fp, top_k=top_k, max_num_rxns=cfg.max_num_rxns)
        if top_k == [1]:
            sk_indices = [sk_indices]
        bts = [
            utils.skeleton_to_binary_tree(lookup_skeleton_by_index(sk_index))
            for sk_index in sk_indices
        ]
        return bts[0] if (top_k == [1]) else bts

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
                assert 1 <= ind.bt.number_of_nodes()
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

    def crossover_and_mutate_fp(self, parents: Tuple[Individual, Individual]) -> Individual:
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

        return fp

    def random_bt_edits(self, ind: Individual) -> Individual:
        bt = ind.bt.copy()

        # bt: random add or delete nodes
        cfg = self.config
        num_edits = torch.randint(1, cfg.bt_mutate_edits + 1, size=[1]).item()
        for _ in range(num_edits):
            if utils.random_boolean(0.5):
                utils.random_add_leaf(bt, max_internal=cfg.max_num_rxns)
            else:
                utils.random_remove_leaf(bt)

        return Individual(fp=ind.fp.copy(), bt=bt)

    def random_fp_flips(self, ind: Individual) -> Individual:
        cfg = self.config
        mask = utils.random_bitmask(cfg.fp_bits, k=round(cfg.fp_bits * cfg.fp_mutate_frac))
        fp = np.where(mask, 1 - ind.fp, ind.fp)

        # Initialize bt
        bt = self.predict_bt(fp)

        return Individual(fp=fp, bt=bt)

    def promote_exploit(self, candidates, gp: GaussianProcessRegressor, best):
        X = np.stack([ind.fp for ind in candidates], axis=0)
        y, std = gp.predict(X, return_std=True)
        z = (y - best) / std
        ei = (y - best) * norm.cdf(z) + std * norm.pdf(z)
        return candidates[ei.argmax()]

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

    def record_history(self, population: Population):
        X = np.stack([ind.fp for ind in population], axis=0)
        y = np.array([ind.fitness for ind in population])
        return X, y

    @staticmethod
    def init_oracle(objective):
        global oracle
        oracle = fetch_oracle(objective)

    def apply_oracle_job(self, smi):
        global oracle

        try:
            score = oracle(smi)
            if not np.isfinite(score):
                print("Oracle NaN on", smi)
                score = 0.0
        except:
            print("Oracle erorr on", smi)
            score = 0.0
        if self.config.objective in ["7l11", "drd3"]:
            score = -score

        return score

    def apply_oracle(self, population: Population, pool) -> None:
        smiles = [ind.smiles for ind in population]
        map_fn = map if (pool is None) else pool.map
        for i, score in enumerate(map_fn(self.apply_oracle_job, smiles)):
            population[i].fitness = score

    def optimize(self, surrogate: Callable[[Population], None]) -> None:
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

        # Oracle Pool
        if cfg.max_oracle_workers > 0:
            pool = Pool(
                processes=cfg.max_oracle_workers,
                initializer=self.init_oracle,
                initargs=[cfg.objective],
            )
        else:
            pool = None
            self.init_oracle(cfg.objective)

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
            self.apply_oracle(population, pool)
            metrics = self.evaluate_scores(population, prefix="seeds")
            wandb.log({"generation": -1, **metrics}, commit=True)

            # Safety
            for ind in population:
                ind.smiles = None
                ind.fitness = None

        # Track some stats
        num_calls = 0
        X_history, y_history = None, None
        score_queue = collections.deque(maxlen=cfg.early_stop_patience)
        score_queue.append(-1000)

        # Main loop
        for epoch in tqdm.trange(-1, cfg.generations, desc="Searching"):

            # Crossover & mutation
            if epoch >= 0:
                offsprings = []

                couples = self.choose_couples(population, epoch)
                for parents in couples:
                    child_fp = self.crossover_and_mutate_fp(parents)

                    child_ids = list(range(cfg.children_per_couple))
                    if cfg.children_strategy == "topk":
                        child_bts = self.predict_bt(fp=child_fp, top_k=child_ids)
                        group = [Individual(fp=child_fp, bt=bt) for bt in child_bts]
                    elif cfg.children_strategy == "edits":
                        child_base = Individual(fp=child_fp, bt=self.predict_bt(fp=child_fp))
                        group = [self.random_bt_edits(child_base) for _ in child_ids]
                    elif cfg.children_strategy == "flips":
                        child_base = Individual(fp=child_fp, bt=None)
                        group = [self.random_fp_flips(child_base) for _ in child_ids]
                    else:
                        raise NotImplementedError()

                    offpsrings.append(group)

                surrogate(sum(offsprings, []), desc="Surrogate")

                # Choose the candidate that maximizes EI
                kernel = RBF(length_scale=1.0)
                gp = GaussianProcessRegressor(kernel=kernel)
                gp.fit(X=X_history, y=y_history)
                promote = partial(self.promote_exploit, gp=gp, best=np.max(y_history))
                offsprings = list(map(promote, offsprings))

                if num_calls + len(offsprings) > cfg.max_oracle_calls:
                    leftover = cfg.max_oracle_calls - num_calls
                    offsprings = random.sample(offsprings, k=leftover)
                self.apply_oracle(offsprings, pool)
                num_calls += len(offsprings)
                X_history, y_history = self.record_history(population + offsprings)

                population = self.cull(population + offsprings)

            elif cfg.resume_path is not None:
                pass

            else:
                surrogate(population, desc="Evaluating initial")
                self.apply_oracle(population, pool)
                num_calls += len(population)
                X_history, y_history = self.record_history(population)

            self.validate(population)  # sanity check

            # Scoring
            metrics = self.evaluate(population)

            # Logging
            if cfg.wandb:
                table = [[epoch, ind.smiles, ind.fitness] for ind in population]
                columns = ["generation", "smiles", "fitness"]
                metrics["smiles"] = wandb.Table(columns=columns, data=table)
                metrics = {"generation": epoch, "oracle_calls": num_calls, **metrics}
                wandb.log(metrics, commit=True)
            if cfg.checkpoint_path is not None:
                with open(cfg.checkpoint_path, "wb") as f:
                    pickle.dump(population, f)

            # Early-stopping
            score_queue.append(metrics["scores/mean"])
            if (
                cfg.early_stop
                and (epoch > cfg.early_stop_warmup)
                and (len(score_queue) == cfg.early_stop_patience)
                and (score_queue[-1] - score_queue[0] < cfg.early_stop_delta)
            ):
                print("Early stopping.")
                break

            # Exhausted oracle calls
            if num_calls == cfg.max_oracle_calls:
                print("Exhausted oracle calls")
                break

        # Cleanup
        if cfg.wandb:
            wandb.finish()

        if pool is not None:
            pool.close()
            pool.join()
