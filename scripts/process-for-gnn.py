from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton, SkeletonSet, \
load_skeletons, compute_md, get_bool_mask, get_wl_kernel, process_syntree_mask, test_is_leaves_up, inds_to_i
import pickle
import os
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/pre-process/syntrees/synthetic-trees.json.gz",
        help="Input file for the generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--skeleton-file",
        type=str,
        default="results/viz/skeletons.pkl",
        help="Input file for the skeletons of syntree-file",
    )       
    parser.add_argument(
        "--skeleton-canonical-file",
        type=str,
        help="If given, use the keys as skeleton classes",
    )          
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="results/viz/",
        help="Input file for the skeletons of syntree-file",
    )
    parser.add_argument(
        "--anchor_type",
        choices=['rset', 'leaves', 'target'],
        default='rset',
        help="What constitutes anchors"
    )
    parser.add_argument(
        "--gnn-datasets", type=int, nargs='+'
    )
    parser.add_argument("--predict_anchor", action='store_true')    
    parser.add_argument(
        "--determine_criteria",
        choices=['leaves_up', 'all_leaves', 'target_down', 'rxn_target_down', 'rxn_target_down_bb', 'rxn_frontier', 'bb_frontier', 'leaf_up_2'],
        default='leaves_up',
        help="""
        Criteria for a determined skeleton:
            leaves_up: all children present
            all_leaves: all leaves present
            target_down: all predecessors present
            rxn_target_down: all non-rxns masked, reaction target down, predict rxns
            rxn_target_down_bb: all interms masked, reaction target down, predict rxns and bb's
            bfs_frontier: bfs frontier, expand from target only
            rxn_frontier: there exists rxn on bfs frontier, predict rxns
            bb_frontier: if there exists rxn on bfs frontier, predict rxns only; else predict all bfs frontier
            leaf_up_2: target and bottom-2 reactions unmasked
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Where to save input and output for GNN"
    )
    parser.add_argument(
        "--num-trees-per-batch",
        type=int,
        default=-1,
        help="Number of trees per batch, if -1 then debug with 1"
    )    
    # Visualization args
    parser.add_argument(
        "--min_count",
        type=int,
        default=10
    )
    parser.add_argument(
        "--num_to_vis",
        type=int,
        default=10
    )   

    # Processing
    parser.add_argument("--ncpu", type=int, help="Number of cpus", default=1)
    return parser.parse_args()


def get_parg(syntree, min_r_set, index, args):
    determine_criteria=args.determine_criteria
    predict_anchor=args.predict_anchor
    sk = Skeleton(syntree, index)
    pargs = []
    if min_r_set[0] != sk.tree_root:
        breakpoint()
    if predict_anchor:
        # fill some anchors, predict the next
        for i in range(2**(len(min_r_set)-1)-1):
            sk.reset([sk.tree_root])
            zero_mask_inds = np.array(min_r_set[1:])
            bool_mask = get_bool_mask(i)
            fill_in = zero_mask_inds[-len(bool_mask):][bool_mask]
            sk.mask = fill_in
            pargs.append([i, sk, args, min_r_set, [sk.tree_root] + fill_in])    
    else:
        # for some, we can compute the valid i easily
        if determine_criteria == 'leaf_up_2':
            sk.mask = [sk.tree_root]
            inds = [inds_to_i(sk.bottom_2_rxns + [sk.tree_root], len(sk.tree), min_r_set)]
        else:
            inds = range(2**(len(sk.tree)-len(min_r_set))-1)

        for i in inds:
            sk.reset(min_r_set)
            zero_mask_inds = np.where(sk.mask == 0)[0]
            bool_mask = get_bool_mask(i)
            sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
            if getattr(sk, determine_criteria):
                pargs.append([i, sk, args, min_r_set])
    return pargs


def main():
    args = get_args()
    skeletons = load_skeletons(args)    
    sk_set = SkeletonSet().load_skeletons(skeletons)
    len_inds = np.argsort([len(skeletons[st]) for st in skeletons])
    kth_largest = np.zeros(len(skeletons))
    kth_largest[len_inds] = np.arange(len(skeletons))[::-1]
    for index, st in tqdm(enumerate(skeletons)):
        if len(list(skeletons[st])) == 0:
            continue
        if args.gnn_datasets is not None and index not in args.gnn_datasets:
            continue    
        # if len(skeletons[st]) < 100:
        #     continue
        # if index < 3:
        #     continue
        if index < 15:
            continue
        # figure out "a" minimal resolving set   
        if kth_largest[index]+1 > 100:
            continue
        print(f"class {index} which is {kth_largest[index]+1}th most represented")
        sk = Skeleton(st, index)
        # sk.visualize(os.path.join(args.visualize_dir, f"{index}.png"))
        # print(os.path.abspath(os.path.join(args.visualize_dir, f"{index}.png")))
        # check that every building block is under a bottom-most-2 reaction
        # sk_copy = deepcopy(sk)
        # sk_copy.mask = [sk_copy.tree_root]
        # if not np.all([list(sk.tree.predecessors(bb))[0] in sk_copy.bottom_2_rxns for bb in np.argwhere(sk.leaves).flatten()]):
        #     print(f"skipping {index}")
        #     continue        

        edge_index = np.array(sk.tree.edges).T           
        pargs = []
        syntree = list(skeletons[st])[0]

        # with Pool(20) as p:        
        #     pargs = p.starmap(get_parg, tqdm([(syntree, min_r_set, index) for syntree in skeletons[st]]))

        sk = Skeleton(syntree, index)
        
        if args.anchor_type == 'rset':
            # Anchors are the min-resolving set
            min_r_set = compute_md(sk.tree, sk.tree_root)        
        elif args.anchor_type == 'leaves':
            # Anchors are all the non-target leaves
            min_r_set = [sk.tree_root] + np.array(sk.tree.nodes)[sk.leaves].tolist()
        else:
            min_r_set = [sk.tree_root]
        
        with Pool(args.ncpu) as p:
            pargs = p.starmap(get_parg, tqdm([[syntree, min_r_set, index, args] for syntree in tqdm(skeletons[st], desc="gathering pargs")]))
        # pargs = [get_parg(*[syntree, min_r_set, index, args]) for syntree in tqdm(skeletons[st])]
        pargs = [parg for parg_sublist in pargs for parg in parg_sublist]
        print(f"mapping {len(pargs)}")
        if args.num_trees_per_batch == -1:
            batch_size = 1
        else:
            batch_size = args.num_trees_per_batch*len(pargs)//len(skeletons[st])        
        num_batches = (len(pargs)+batch_size-1)//batch_size
        print(f"{num_batches} batches")
        for k in tqdm(range(num_batches), desc="batches"): 
            if os.path.exists(os.path.join(args.output_dir, f"{index}_{k}_node_masks.npy")):
                continue
            # print(k)  
            res = []     
            if batch_size > 1:
                with Pool(args.ncpu) as p:
                    res = p.starmap(process_syntree_mask, tqdm(pargs[batch_size*k:batch_size*k+batch_size], desc="featurize a batch"))
            else:
                res.append(process_syntree_mask(*pargs[k]))            
            res = [r for r in res if r is not None]                       
            node_masks = np.concatenate([r[0] for r in res], axis=0)
            Xs = np.concatenate([r[1] for r in res], axis=0)
            ys = np.concatenate([r[2] for r in res], axis=0)              
            smiles = np.array([r[3] for r in res])
            
            Xs = sparse.csr_array(Xs)
            ys = sparse.csr_array(ys)
            node_masks = sparse.csr_array(node_masks)
            sparse.save_npz(os.path.join(args.output_dir, f"{index}_{k}_Xs.npz"), Xs)
            sparse.save_npz(os.path.join(args.output_dir, f"{index}_{k}_ys.npz"), ys)
            np.save(os.path.join(args.output_dir, f"{index}_{k}_smiles.npy"), smiles)
            sparse.save_npz(os.path.join(args.output_dir, f"{index}_{k}_node_masks.npz"), node_masks)
            np.save(os.path.join(args.output_dir, f"{index}_edge_index.npy"), edge_index)

        # with Pool(min(100, len(skeletons[st]))) as p:
        #     res = p.starmap(process_syntree_mask, tqdm(pargs))    
            # res += interm_res
        # with Pool(min(50, len(skeletons[st]))) as p:
        #     res = p.starmap(process_syntree_mask, tqdm(pargs))
        # res = []
        # for parg in tqdm(pargs):
        #     res.append(process_syntree_mask(*parg))
        # node_masks = np.concatenate([r[0] for r in res], axis=0)
        # Xs = np.concatenate([r[1] for r in res], axis=0)
        # ys = np.concatenate([r[2] for r in res], axis=0)              
        # np.save(os.path.join(args.output_dir, f"{index}_Xs.npy"), Xs)
        # np.save(os.path.join(args.output_dir, f"{index}_ys.npy"), ys)
        # np.save(os.path.join(args.output_dir, f"{index}_node_masks.npy"), node_masks)
        # np.save(os.path.join(args.output_dir, f"{index}_edge_index.npy"), edge_index)



if __name__ == "__main__":    
    main()
    # args = get_args()
    # skeletons = pickle.load(open(args.skeleton_file, 'rb'))
    # sk_set = SkeletonSet().load_skeletons(skeletons)
    # bad_key = list(skeletons.keys())[9]
    # for syntree in skeletons[bad_key]:
    #     if syntree.root.smiles == 'CN(CC(N)=O)C(=O)c1cccc(NS(=O)(=O)CCC#Cc2cc(Cl)c(F)c(CNS(=O)(=O)CCCn3cnnn3)c2)c1':
    #         breakpoint()
    #     sk = Skeleton(syntree, 9)
    #     min_r_set = compute_md(sk.tree, sk.tree_root)
    #     if min_r_set[0] != sk.tree_root:
    #         breakpoint()        
            