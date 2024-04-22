"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""  # TODO: Clean up + dont hardcode file paths
import json
import logging
import numpy as np
np.random.seed(42)
import pandas as pd
import pdb
from tqdm import tqdm
import os
import pickle
from synnet.utils.reconstruct_utils import *
import multiprocessing as mp

logger = logging.getLogger(__name__)



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
    parser.add_argument("--ckpt-bb", type=str, help="Model checkpoint to use")
    parser.add_argument("--ckpt-rxn", type=str, help="Model checkpoint to use")    
    parser.add_argument("--ckpt-dir", type=str, help="Model checkpoint dir, if given assume one ckpt per class")
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
    parser.add_argument(
        "--out-dir"        
    )
    # Parameters
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help="Choose from ['train', 'valid', 'test', 'chembl'] or provide a file with one SMILES per line.",
    )
    parser.add_argument("--top-bbs-file", help='if given, consider only these bbs')
    parser.add_argument("--top-k", default=1, type=int)
    parser.add_argument("--batch-size", default=10, type=int, help='how often to report metrics')
    parser.add_argument("--filter-only", type=str, nargs='+', choices=['rxn', 'bb'], default=[])
    parser.add_argument("--forcing-eval", action='store_true')
    parser.add_argument("--test-correct-method", default='preorder', choices=['preorder', 'postorder', 'reconstruct'])
    # Visualization
    parser.add_argument("--mermaid", action='store_true')
    parser.add_argument("--one-per-class", action='store_true', help='visualize one skeleton per class')
    # Processing
    parser.add_argument("--ncpu", type=int, default=1, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()    


def main(args):    
    handler = logging.FileHandler(os.path.join(args.out_dir, "log.txt"))
    logger.addHandler(handler)
    logger.info("Start.")
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")
    # ... models

    set_models(args, logger)    
    load_data(args, logger)

    # building blocks
    if args.top_bbs_file:
        TOP_BBS = set([l.rstrip('\n') for l in open(args.top_bbs_file).readlines()])
    else:
        TOP_BBS = BuildingBlockFileHandler().load(args.building_blocks_file)        

    # Load skeleton set
    sk_set = None     
    skeletons = pickle.load(open(args.skeleton_set_file, 'rb'))
    skeleton_set = SkeletonSet().load_skeletons(skeletons)
    syntree_set_all = [st for v in skeletons.values() for st in v]
    syntree_set = []
    SKELETON_INDEX = test_skeletons(args, skeleton_set)
    print(f"SKELETON INDEX: {SKELETON_INDEX}")
    if args.one_per_class:
        rep = set()
    for syntree in syntree_set_all:        
        index = skeleton_set.lookup[syntree.root.smiles][0].index    
        if len(skeleton_set.lookup[syntree.root.smiles]) == 1: # one skeleton per smiles                
            if index in SKELETON_INDEX:
                if args.one_per_class:
                    if index not in rep:
                        rep.add(index)
                        syntree_set.append(syntree)
                else:
                    syntree_set.append(syntree)       
    targets = [syntree.root.smiles for syntree in syntree_set]
    lookup = {}
    # Compute the gold skeleton
    for i, target in enumerate(targets):            
        sk = Skeleton(syntree_set[i], skeleton_set.lookup[target][0].index)             
        smile_set = [c.smiles for c in syntree_set[i].chemicals]
        if len(set(smile_set)) != len(smile_set):
            continue
        good = True     
        for n in sk.tree:
            if sk.leaves[n]:
                if sk.tree.nodes[n]['smiles'] not in TOP_BBS:
                    good = False
        if good:
            lookup[target] = sk
    targets = list(lookup)
    print(f"{len(targets)}/{len(syntree_set_all)} syntrees")


    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")

    # targets = targets[:10]     
    batch_size = args.batch_size        
    # decode('CC(Nc1ccc(I)cc1-c1nc(-c2cc(F)ccc2N=C=O)n[nH]1)c1ccc(N)c(O)c1')
    # decode('CC(C)(N=C=O)C1=Cc2cc(Br)c(C(=O)O)c(N)c2O1')
    all_sks = []
    all_targets = []
    for batch in range((len(targets)+batch_size-1)//batch_size):
        target_batch = targets[batch_size*batch:batch_size*batch+batch_size]
        target_batch = [(deepcopy(lookup[smi]), smi) for smi in target_batch]
        if args.ncpu == 1:
            sks_batch = []
            for arg in tqdm(target_batch):                        
                sks = decode(*arg)
                sks_batch.append(sks)              
        else:
            with mp.Pool(args.ncpu) as p:
                sks_batch = p.starmap(decode, tqdm(target_batch))        
        all_targets += target_batch
        all_sks += sks_batch
        if args.forcing_eval:
            correct_summary = get_metrics(target_batch, sks_batch)
            logger.info(f"correct summary: {correct_summary}")
        else:
            batch_correct, batch_incorrect = get_metrics(target_batch, sks_batch)
            logger.info(f"batch {batch} correct: {format_metrics(batch_correct)}")
            logger.info(f"batch {batch} incorrect: {format_metrics(batch_incorrect)}")
    
    if args.forcing_eval:
        correct_summary = get_metrics(targets, all_sks)
        logger.info(f"correct summary: {correct_summary}")
    else:
        total_correct, total_incorrect = get_metrics(all_targets, all_sks)
        logger.info(f"total correct: {format_metrics(total_correct)}")
        logger.info(f"total incorrect: {format_metrics(total_incorrect)}")        
        
  
    logger.info("Finished decoding.")

    return




if __name__ == "__main__":

    # Parse input args
    args = get_args()
    breakpoint()
    main(args)
