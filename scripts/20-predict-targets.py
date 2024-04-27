"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""  # TODO: Clean up + dont hardcode file paths
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import numpy as np
np.random.seed(42)
import pandas as pd
import pdb
import pickle
import os
from tqdm import tqdm

from synnet.config import DATA_PREPROCESS_DIR, DATA_RESULT_DIR, MAX_PROCESSES
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.distances import cosine_distance
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import ReactionSet, SyntheticTree, SyntheticTreeSet, SkeletonSet
from synnet.utils.predict_utils import mol_fp, synthetic_tree_decoder_greedy_search

logger = logging.getLogger(__name__)


def _fetch_data_chembl(name: str) -> list[str]:
    raise NotImplementedError
    df = pd.read_csv(f"{DATA_DIR}/chembl_20k.csv")
    smis_query = df.smiles.to_list()
    return smis_query


def _fetch_data_from_file(name: str) -> list[str]:
    if '.json.gz' in name:
        syntree_collection = SyntheticTreeSet().load(name)
        smiles = [syntree.root.smiles for syntree in syntree_collection]        
        return smiles
    else:
        with open(name, "rt") as f:
            smis_query = [line.strip() for line in f]
        return smis_query


def _fetch_data(name: str) -> list[str]:
    if args.data in ["train", "valid", "test"]:
        file = (
            Path(DATA_PREPROCESS_DIR) / "syntrees" / f"synthetic-trees-filtered-{args.data}.json.gz"
        )
        logger.info(f"Reading data from {file}")
        syntree_collection = SyntheticTreeSet().load(file)
        smiles = [syntree.root.smiles for syntree in syntree_collection]
    elif args.data in ["chembl"]:
        smiles = _fetch_data_chembl(name)
    else:  # Hopefully got a filename instead
        smiles = _fetch_data_from_file(name)
    return smiles


def wrapper_decoder(smiles: str, sk_coords=None) -> Tuple[str, float, SyntheticTree]:
    """Generate a synthetic tree for the input molecular embedding."""
    emb = mol_fp(smiles)
    try:
        smi, similarity, tree, action = synthetic_tree_decoder_greedy_search(
            z_target=emb,
            sk_coords=sk_coords,
            building_blocks=bblocks,
            bb_dict=bblocks_dict,
            reaction_templates=rxns,
            mol_embedder=bblocks_molembedder.kdtree,  # TODO: fix this, currently misused
            action_net=act_net,
            reactant1_net=rt1_net,
            rxn_net=rxn_net,
            reactant2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template="hb",  # TODO: Do not hard code
            n_bits=4096,  # TODO: Do not hard code
            beam_width=3,
            max_step=15,
        )
    except Exception as e:
        logger.error(e, exc_info=e)
        action = -1

    if action != 3:  # aka tree has not been properly ended
        smi = None
        similarity = 0
        tree = None

    return smi, similarity, tree


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxns-collection-file",
        type=str,
        help="Input file for the collection of reactions matched with building-blocks.",
    )
    parser.add_argument(
        "--embeddings-knn-file",
        type=str,
        help="Input file for the pre-computed embeddings (*.npy).",
    )
    parser.add_argument(
        "--top-bbs-file",
        type=str,
        help="If given, limit to only bbs from this"
    )
    parser.add_argument(
        "--ckpt-dir", type=str, help="Directory with checkpoints for {act,rt1,rxn,rt2}-model."
    )
    parser.add_argument("--ckpt-versions", type=int, help="If given, use ckpt versions in ckpt-dir", nargs='+')
    parser.add_argument(
        "--skeleton-set-file",
        type=str,
        help="File of skeleton set",
    )           
    parser.add_argument(
        "--output-dir", type=str, default=DATA_RESULT_DIR, help="Directory to save output."
    )
    # Parameters
    parser.add_argument("--num", type=int, default=-1, help="Number of molecules to predict.")
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help="Choose from ['train', 'valid', 'test', 'chembl'] or provide a file with one SMILES per line.",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load skeleton set
    sk_set = None
    if args.skeleton_set_file:
        sk_set = pickle.load(open(args.skeleton_set_file, 'rb'))

    # Load data ...
    logger.info("Start loading data...")
    # ... query molecules (i.e. molecules to decode)
    targets = _fetch_data(args.data)
    if args.num > 0:  # Select only n queries
        targets = targets[: args.num]

    # ... building blocks
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    if args.top_bbs_file:
        bblock_inds = [bblocks.index(l.rstrip('\n')) for l in open(args.top_bbs_file).readlines()]
        bblock_inds = sorted(bblock_inds)
        bblocks = [bblocks[ind] for ind in bblock_inds]
        bblocks_dict = {block: i for i, block in enumerate(bblocks)}
        emb_path = args.top_bbs_file.replace('.txt', '.npy')
        # if not os.path.exists(emb_path):
        data = np.load(args.embeddings_knn_file)
        top_emb = data[bblock_inds]
        np.save(emb_path, top_emb)        
        bblocks_molembedder = (
            MolEmbedder().load_precomputed(emb_path).init_balltree(cosine_distance)
        )     
        bb_emb = bblocks_molembedder.get_embeddings()         
    else:
        bblock_inds = None        
        # A dict is used as lookup table for 2nd reactant during inference:
        bblocks_dict = {block: i for i, block in enumerate(bblocks)}
        logger.info(f"Successfully read {args.building_blocks_file}.")        
        # ... building block embedding
        bblocks_molembedder = (
            MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
        )
        bb_emb = bblocks_molembedder.get_embeddings()
        logger.info(f"Successfully read {args.embeddings_knn_file} and initialized BallTree.")        


    # ... reaction templates
    rxns = ReactionSet().load(args.rxns_collection_file).rxns
    for rxn in rxns:
        for i in range(len(rxn.available_reactants)):
            rxn.available_reactants[i] = [reactant for reactant in rxn.available_reactants[i] if reactant in bblocks]

    logger.info(f"Successfully read {args.rxns_collection_file}.")


    logger.info("...loading data completed.")

    # ... models
    logger.info("Start loading models from checkpoints...")
    path = Path(args.ckpt_dir)
    if args.ckpt_versions:
        versions = args.ckpt_versions
    else:
        versions = [None, None, None, None]
    ckpt_files = []
    for model, version in zip("act rt1 rxn rt2".split(), versions):
        ckpt_file = find_best_model_ckpt(path / model, version)
        ckpt_files.append(ckpt_file)
    print(ckpt_files)
    act_net, rt1_net, rxn_net, rt2_net = [load_mlp_from_ckpt(file) for file in ckpt_files]
    logger.info("...loading models completed.")

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")

    if args.ncpu == 1:
        results = []
        for smi in tqdm(targets):
            if args.skeleton_set_file:
                index = sk_set.lookup[smi].index
                sk_coords = sk_set.coords[index:index+1]
                results.append(wrapper_decoder(smi, sk_coords))
            else:
                results.append(wrapper_decoder(smi))
    else:
        if args.skeleton_set_file:
            for i in range(len(targets)):
                smi = targets[i]
                index = sk_set.lookup[smi].index
                sk_coords = sk_set.coords[index:index+1]            
                targets[i] = (targets[i], sk_coords)
            with mp.Pool(processes=args.ncpu) as pool:
                logger.info(f"Starting MP with ncpu={args.ncpu}")
                results = pool.starmap(wrapper_decoder, tqdm(targets))
        else:
            with mp.Pool(processes=args.ncpu) as pool:
                logger.info(f"Starting MP with ncpu={args.ncpu}")
                results = pool.map(wrapper_decoder, tqdm(targets))
    logger.info("Finished decoding.")

    # Print some results from the prediction
    # Note: If a syntree cannot be decoded within `max_depth` steps (15),
    #       we will count it as unsuccessful. The similarity will be 0.    
    decoded = [smi for smi, _, _ in results]
    similarities = [sim for _, sim, _ in results]
    trees = [tree for _, _, tree in results]

    recovery_rate = (np.asfarray(similarities) == 1.0).sum() / len(similarities)
    avg_similarity = np.mean(similarities)
    n_successful = sum([syntree is not None for syntree in trees])
    logger.info(f"For {args.data}:")
    logger.info(f"  Total number of attempted  reconstructions: {len(targets)}")
    logger.info(f"  Total number of successful reconstructions: {n_successful}")
    logger.info(f"  {recovery_rate=}")
    logger.info(f"  {avg_similarity=}")

    # Save to local dir
    # 1. Dataframe with targets, decoded, smilarities
    # 2. Synthetic trees of the decoded SMILES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir} ...")

    df = pd.DataFrame({"targets": targets, "decoded": decoded, "similarity": similarities})
    df.to_csv(f"{output_dir}/decoded_results.csv.gz", compression="gzip", index=False)
    df.to_csv(f"{output_dir}/decoded_results.csv", index=False)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(f"{output_dir}/decoded_syntrees.json.gz")

    logger.info("Completed.")
