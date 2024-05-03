import collections
import itertools
import json
import random
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import tqdm
import wandb

from ga import utils
from ga.config import GeneticSearchConfig, Individual
from synnet.encoding.distances import _tanimoto_similarity
from synnet.encoding.fingerprints import mol_fp
from synnet.utils.analysis_utils import serialize_string
from synnet.utils.data_utils import binary_tree_to_skeleton

Population = List[Individual]


class GeneticSearch:

    def __init__(self, config: GeneticSearchConfig):
        self.config = config

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
        raise NotImplementedError("(AL) I'm pretty sure there's a bug here")
        with open(path, "r") as f:
            state = json.load(f)

        population = []
        for ind in state:
            bt = nx.tree_graph(ind["bt"])  # only supports node-level attributes
            bt = nx.relabel_nodes(bt, {k: utils.random_name() for k in list(bt.nodes)})
            for n in bt:  # move child node attribute "left" to edge attribute
                preds = list(bt.predecessors(n))
                if len(preds) == 1:
                    pred = preds[0]
                else:
                    continue
                if list(bt[pred]) == 1:   # FIXME: look into this
                    bt.edges[(pred, n)]["left"] = True
                else:
                    assert "left" in bt.nodes[n]
                    bt.edges[(pred, n)]["left"] = bt.nodes[n]["left"]
            if "smi" in ind:
                fp = mol_fp(ind["smi"], _nBits=self.config.fp_bits)
                fp = np.array(fp, dtype=bool)
            else:
                fp = ind["fp"]
                fp = np.array(fp, dtype=bool)
            population.append(Individual(fp=fp, bt=bt))
        return population

    def validate(self, population: Population):
        cfg = self.config
        for ind in population:
            assert ind.fp.shape == (cfg.fp_bits,)
            assert cfg.bt_nodes_min <= ind.bt.number_of_nodes() <= cfg.bt_nodes_max

    def evaluate(self, population: Population) -> Dict[str, float]:
        scores = [ind.fitness for ind in population]
        scores.sort(reverse=True)

        # Fitness
        metrics = {
            "scores/mean": np.mean(scores).item(),
            "scores/stdev": np.std(scores).item(),
        }
        for k in [1, 10, 100]:
            metrics[f"scores/mean_top{k}"] = np.mean(scores[:k]).item()

        # Diversity
        distances = []
        fps = [mol_fp(ind.smiles, _nBits=2048) for ind in population]
        for a, b in itertools.combinations(fps, r=2):
            d = 1 - _tanimoto_similarity(a, b)
            distances.append(d)
        metrics["diversity"] = np.mean(distances).item()

        return metrics

    def cull(self, population: Population) -> Population:
        serials = []
        for indv in population:
            tree = indv.bt
            root = next(v for v, d in tree.in_degree() if d == 0)
            fp_str = "".join(list(map(str, map(int, indv.fp))))
            serial = serialize_string(tree, root) + " " + fp_str
            serials.append(serial)
        _, inds = np.unique(serials, return_index=True)
        population = [population[ind] for ind in inds]
        population = sorted(population, key=(lambda x: x.fitness), reverse=True)
        return population[:self.config.population_size]

    def choose_couples(
        self,
        population: Population,
        epoch: int,
    ) -> List[Tuple[Individual, Individual]]:
        population = sorted(population, key=(lambda x: x.fitness))  # ascending
        indices = np.arange(len(population))

        cfg = self.config
        if cfg.parent_schedule == "anneal":
            t = epoch / cfg.generations
            temp = (1 - t) * cfg.parent_temp_max + t * cfg.parent_temp_min  # LERP
            p = scipy.special.softmax(indices / temp)
        elif cfg.parent_schedule == "synnet":
            p = indices + 10
            if epoch < 0.8 * cfg.generations:
                p = p / np.sum(p)
            else:
                p = scipy.special.softmax(p)
        else:
            raise NotImplementedError()

        couples = []
        for _ in range(cfg.offspring_size):
            i1, i2 = np.random.choice(indices, size=[2], replace=False, p=p)
            couples.append((population[i1], population[i2]))
        return couples

    def crossover(self, parents: Tuple[Individual, Individual]) -> Individual:
        cfg = self.config

        # fp: random bit swap
        n = cfg.fp_bits
        k = scipy.stats.truncnorm.rvs(
            a=(0.2 * n), b=(0.8 * n),
            loc=(n / 2),
            scale=(n / 10),
            size=1,
        )
        k = int(np.round(k))
        mask = utils.random_bitmask(cfg.fp_bits, k=k)
        fp = np.where(mask, parents[0].fp, parents[1].fp)

        # bt: random subtree swap
        trees = [parents[0].bt, parents[1].bt]
        random.shuffle(trees)
        bt = utils.random_graft(
            *trees,
            min_nodes=cfg.bt_nodes_min,
            max_nodes=cfg.bt_nodes_max,
        )

        return Individual(fp=fp, bt=bt)

    def mutate(self, ind: Individual) -> Individual:
        cfg = self.config

        # fp: random bit flip
        fp = ind.fp
        if utils.random_boolean(cfg.fp_mutate_prob):
            mask = utils.random_bitmask(cfg.fp_bits, k=cfg.fp_mutate_bits)
            fp = np.where(mask, ~fp, fp)

        # bt: random add or delete nodes
        bt = ind.bt
        if utils.random_boolean(cfg.bt_mutate_prob):
            bt = bt.copy()
            for _ in range(cfg.bt_mutate_edits):
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
        for epoch in tqdm.trange(cfg.generations + 1, desc="Searching"):

            # Crossover & mutation
            if epoch > 0:
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
                wandb.log({"generation": epoch, **metrics}, step=epoch, commit=True)
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
