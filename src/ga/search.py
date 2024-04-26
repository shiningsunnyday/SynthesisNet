import collections
import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import tqdm
import wandb
import json
from multiprocessing.pool import ThreadPool
import os
import networkx as nx
from ga import utils
from ga.config import GeneticSearchConfig, Individual
from synnet.utils.analysis_utils import serialize_string

Population = List[Individual]


class GeneticSearch:

    def __init__(self, config: GeneticSearchConfig):
        self.config = config

    def initialize(self, path='/home/msun415/SynTreeNet/indvs-qed.json') -> Population:
        if path and os.path.exists(path):
            indvs = json.load(open(path))
            population = []
            for indv in indvs:
                bt_data = indv['bt']
                bt = nx.tree_graph(bt_data) # only supports node-level attributes
                bt = nx.relabel_nodes(bt, dict(zip(list(bt.nodes), [utils.random_name() for _ in bt])))
                for n in bt: # move child node attribute 'left' to edge attribute
                    preds = list(bt.predecessors(n))
                    if len(preds) == 1:
                        pred = preds[0]
                    else:
                        continue
                    if list(bt[pred]) == 1:
                        bt.edges[(pred, n)]['left'] = True
                    else:
                        assert 'left' in bt.nodes[n]
                        bt.edges[(pred, n)]['left'] = bt.nodes[n]['left']                        
                fp = np.array(indv['fp'], dtype=bool)
                population.append(Individual(fp=fp, bt=bt))            
        else:
            cfg = self.config
            population = []
            for _ in range(cfg.population_size):
                fp = np.random.choice([True, False], size=cfg.fp_bits)
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
        if self.config.ncpu > 1:
            with ThreadPool(self.config.ncpu) as p:
                scores = p.map(lambda ind: ind.fitness, population)
        else:
            scores = [ind.fitness for ind in population]
        scores.sort(reverse=True)

        metrics = {
            "scores/mean": np.mean(scores).item(),
            "scores/stdev": np.std(scores).item(),
        }
        for k in [1, 10, 100]:
            metrics[f"scores/mean_top{k}"] = np.mean(scores[:k]).item()

        return metrics

    def cull(self, population: Population) -> Population:
        serials = []
        for indv in population:
            tree = indv.bt
            root = next(v for v, d in tree.in_degree() if d == 0)
            fp_str = ''.join(list(map(str, map(int, indv.fp))))
            serial = serialize_string(tree, root) + ' ' + fp_str
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
        population = sorted(population, key=(lambda x: x.fitness))
        indices = np.arange(len(population))

        cfg = self.config
        t = epoch / cfg.generations
        temp = (1 - t) * cfg.parent_temp_max + t * cfg.parent_temp_min  # LERP
        p = scipy.special.softmax(indices / temp)

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

        # if random.random() < 0.5:
        #     fp = parents[0].fp
        # else:
        #     fp = parents[1].fp

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
        population = self.initialize()

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
            print([x.fitness for x in population])
            print(metrics)

            # Logging
            if cfg.wandb:
                wandb.log({"generation": epoch, **metrics}, step=epoch)

            # TODO: save skeletons (?)

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
