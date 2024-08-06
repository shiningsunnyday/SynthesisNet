import functools
import logging
import pathlib
import pickle

import jsonargparse
import wandb
from torch.utils.benchmark import Timer

from ga.search import GeneticSearch
from sandbox.optimize import OptimizeGAConfig, get_smiles_ours, get_smiles_synnet, logger, test_surrogate
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.distances import cosine_distance
from synnet.models.legacy import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import ReactionSet, SkeletonSet
from synnet.utils.reconstruct_utils import load_data, set_models, test_skeletons

args = None

def main():
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(OptimizeGAConfig, as_positional=False)
    config = OptimizeGAConfig(**parser.parse_args().as_dict())

    global args
    args = config  # Hack so reconstruct_utils.py works

    if config.log_file:
        handler = logging.FileHandler(config.log_file)
        logger.addHandler(handler)

    if config.method == "ours":
        assert config.fp_bits == 2048

        set_models(config, logger)
        load_data(config, logger)
        with open(config.skeleton_set_file, "rb") as f:
            skeletons = pickle.load(f)
        skeleton_set = SkeletonSet().load_skeletons(skeletons)
        test_skeletons(config, skeleton_set, max_rxns=config.max_num_rxns)

        converter = get_smiles_ours

    elif config.method == "synnet":
        assert config.fp_bits == 4096

        # Load the purchasable building block embeddings
        bblocks_molembedder = (
            MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
        )
        bb_emb = bblocks_molembedder.get_embeddings()

        # Load the purchasable building block SMILES to a dictionary
        building_blocks = BuildingBlockFileHandler().load(args.building_blocks_file)

        # A dict is used as lookup table for 2nd reactant during inference:
        bb_dict = {block: i for i, block in enumerate(building_blocks)}

        # Load the reaction templates as a ReactionSet object
        rxns = ReactionSet().load(args.rxns_collection_file).rxns

        # Load the pre-trained modules
        path = pathlib.Path(__file__).parents[1] / "data" / "checkpoints"
        ckpt_files = [find_best_model_ckpt(path / model) for model in "act rt1 rxn rt2".split()]
        act_net, rt1_net, rxn_net, rt2_net = [load_mlp_from_ckpt(file).cpu() for file in ckpt_files]

        converter = functools.partial(
            get_smiles_synnet,
            building_blocks=building_blocks,
            bb_dict=bb_dict,
            rxns=rxns,
            bblocks_molembedder=bblocks_molembedder,
            act_net=act_net, rt1_net=rt1_net, rxn_net=rxn_net, rt2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template="hb",
            nbits=config.fp_bits,
        )

    else:
        raise NotImplementedError()

    search = GeneticSearch(config)
    surrogate = functools.partial(test_surrogate, converter=converter, pool=None, config=config)

    # Initialize WandB
    if config.wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            dir=config.wandb_dir,
            config=dict(config),
        )

    population = search.initialize_random()
    timer = Timer(
        stmt="surrogate(population, None, dict())",
        globals={"population": population, "surrogate": surrogate}
    )
    print(timer.timeit(1))

    # Cleanup
    if config.wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    main()