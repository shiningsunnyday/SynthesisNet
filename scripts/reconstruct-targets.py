"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""  # TODO: Clean up + dont hardcode file paths
import json
import logging
import gzip
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import numpy as np
np.random.seed(42)
import pandas as pd
import pdb
import os
import pickle

from synnet.config import DATA_PREPROCESS_DIR, DATA_RESULT_DIR, MAX_PROCESSES
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.distances import cosine_distance
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt, load_gnn_from_ckpt
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import ReactionSet, SyntheticTree, SyntheticTreeSet, Skeleton, SkeletonSet
from synnet.utils.predict_utils import mol_fp, synthetic_tree_decoder_greedy_search
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _fetch_data_chembl(name: str) -> list[str]:
    raise NotImplementedError
    df = pd.read_csv(f"{DATA_DIR}/chembl_20k.csv")
    smis_query = df.smiles.to_list()
    return smis_query


def _fetch_data_from_file(name: str) -> list[str]:
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


@torch.no_grad()
def wrapper_decoder(hash_dir, sk, model_rxn, model_bb):
    """Generate a filled-in skeleton given the input which is only filled with the target."""
    model_rxn.eval()
    model_bb.eval()
    while ((~sk.mask) & (sk.rxns | sk.leaves)).any():
        if sk.mask.sum()>1:
            breakpoint()
            hash_val = sk.hash()
            fpath = os.path.join(cur_dir, f"{hash_val}.json")
            dirpath = os.path.join(cur_dir, f"{hash_val}")
            if os.path.exists(fpath):
                data = json.load(open(fpath, 'r'))
                imposs = np.bool([True for _ in range(91)])
                imposs[data['rxn_ids']] = False
                cur_dir = dirpath
            else:
                breakpoint()
        else:
            cur_dir = hash_dir
            imposs = []
            
        assert sk.rxn_target_down_bb
        # prediction problem
        _, X, _ = sk.get_state()
        data = Data(edge_index=torch.tensor(sk.tree_edges, dtype=torch.int64), 
                    x=torch.Tensor(X))   
        # get frontier rxns     
        frontier_rxns = np.array([False for _ in range(len(sk.tree))])
        for n in np.argwhere((~sk.mask) & sk.rxns).flatten():
            if sk.pred(n)==sk.tree_root or sk.mask[sk.pred(sk.pred(n))]:
                frontier_rxns[n] = True
        frontier_rxns = np.bool_(frontier_rxns)
        # get frontier bbs
        frontier_bbs = np.array([False for _ in range(len(sk.tree))])
        for n in np.argwhere((~sk.mask) & (~sk.rxns) & (sk.leaves)).flatten():
            if sk.mask[sk.pred(n)]: # parent present
                frontier_bbs[n] = True
        if frontier_rxns.any():
            logits = model_rxn(data)[frontier_rxns, -91:]
            logits[:, imposs] = -float("inf") # filter out reactions
            confs, rxn_ids = logits.max(axis=-1)
            node_id = np.arange(len(sk.tree))[frontier_rxns][confs.argmax()]        
            rxn_id = rxn_ids[confs.argmax()].item()
            sk.modify_tree(node_id, rxn_id=rxn_id)        
        else:
            assert frontier_bbs.any()
            logits = model_rxn(data)[frontier_bbs, :256]
            breakpoint()

        
    return sk


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        default="data/assets/building-blocks/enamine_us_matched.csv",  # TODO: change
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        default="data/assets/reaction-templates/hb.txt",  # TODO: change
        help="Input file with reaction templates as SMARTS(No header, one per line).",
    )
    parser.add_argument(
        "--rxns_collection_file",
        type=str,
        default="data/assets/reaction-templates/reactions_hb.json.gz",
    )
    parser.add_argument(
        "--embeddings-knn-file",
        type=str,
        help="Input file for the pre-computed embeddings (*.npy).",
        default="data/assets/building-blocks/enamine_us_emb_fp_256.npy"
    )    
    parser.add_argument(
        "--ckpt-bb", type=str, help="Model checkpoint to use"
    )
    parser.add_argument(
        "--ckpt-rxn", type=str, help="Model checkpoint to use"
    )    
    parser.add_argument(
        "--skeleton-set-file",
        type=str,
        required=True,
        help="Input file for the ground-truth skeletons to lookup target smiles in",
    )           
    parser.add_argument(
        "--hash-dir",
        default="",
        required=True
    )
    # Parameters
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help="Choose from ['train', 'valid', 'test', 'chembl'] or provide a file with one SMILES per line.",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=1, help="Number of cpus")
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
        syntree_set = SyntheticTreeSet().load(args.skeleton_set_file)                
        # Decode
        syntree_set = [syntree for syntree in syntree_set if len(syntree.reactions) == 2]       
        targets = [syntree.root.smiles for syntree in syntree_set]
    else:

        # Load data ...
        logger.info("Start loading data...")
        # ... query molecules (i.e. molecules to decode)    
        targets = _fetch_data(args.data)
    

    # ... building blocks
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    # A dict is used as lookup table for 2nd reactant during inference:
    bblocks_dict = {block: i for i, block in enumerate(bblocks)}
    logger.info(f"Successfully read {args.building_blocks_file}.")

    # ... reaction templates
    rxns = ReactionSet().load(args.rxns_collection_file).rxns
    logger.info(f"Successfully read {args.rxns_collection_file}.")

    # # ... building block embedding
    # bblocks_molembedder = (
    #     MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
    # )
    # bb_emb = bblocks_molembedder.get_embeddings()
    # logger.info(f"Successfully read {args.embeddings_knn_file} and initialized BallTree.")
    # logger.info("...loading data completed.")

    # ... models
    logger.info("Start loading models from checkpoints...")  
    rxn_gnn = load_gnn_from_ckpt(Path(args.ckpt_rxn))
    bb_gnn = load_gnn_from_ckpt(Path(args.ckpt_bb))
    logger.info("...loading models completed.")

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")


    lookup = {}
    # Compute the gold skeleton
    all_smiles = dict(zip([st.root.smiles for st in syntree_set], range(len(syntree_set))))
    for target in targets:
        index = all_smiles[target]
        sk = Skeleton(syntree_set[index], -1)
        lookup[target] = sk

    if args.ncpu == 1:
        results = []
        for smi in targets:
            sk = lookup[smi]
            sk.clear_tree()
            sk.modify_tree(sk.tree_root, smiles=smi)
            sk = wrapper_decoder(args.hash_dir, sk, rxn_gnn, bb_gnn)
            breakpoint()

    # else:
    #     for i in range(len(targets)):
    #         smi = targets[i]
    #         index = sk_set.lookup[smi].index            
    #         targets[i] = (targets[i], sk_coords)
    #     with mp.Pool(processes=args.ncpu) as pool:
    #         logger.info(f"Starting MP with ncpu={args.ncpu}")
    #         results = pool.starmap(wrapper_decoder, targets)
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
