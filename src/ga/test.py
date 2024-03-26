from src.ga.search import GeneticSearch, GeneticSearchConfig


def test_fitness(batch):
    for ind in batch:
        fp = ind.fp
        bt = ind.bt
        leaves = [v for v in bt.nodes() if (bt.out_degree(v) == 0)]
        ind.fitness = 0.05 * (fp[:100].sum() - fp[100:].sum()) + len(leaves)


if __name__ == "__main__":
    config = GeneticSearchConfig(wandb=True)
    GeneticSearch(config).optimize(test_fitness)
