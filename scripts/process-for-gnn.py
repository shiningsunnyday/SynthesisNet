from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton, SkeletonSet
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool

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
        "--visualize-dir",
        type=str,
        default="results/viz/",
        help="Input file for the skeletons of syntree-file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Where to save input and output for GNN"
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
    parser.add_argument("--ncpu", type=int, help="Number of cpus")
    return parser.parse_args()


def skeleton2graph(skeleton):
    graph = nx.MultiDiGraph()
    count = {}
    lookup = {}
    for n in skeleton.nodes:
        name = n.smiles
        if n.smiles in count:
            name += f":{count[n.smiles]}"
        graph.add_node(name)
        count[n.smiles] = count.get(n.smiles, 0)+1
        lookup[n] = name
    for e in skeleton.edges:
        graph.add_edge(lookup[skeleton.nodes[e[0]]], lookup[skeleton.nodes[e[1]]])
    return graph


def compute_md(tree, root_ind):
    """
    https://en.wikipedia.org/wiki/Metric_dimension_(graph_theory)
    The metric dimension for a path is 1
    The metric dimension for a tree is |leaves|-|joints|
    Leaf is node of degree 1
    Joint is node of degree 3 that has a straight path to at least one leaf
    This function obtains |leaves|-|joints| leaves which is resolving
    Then prune ones which are redundant because the root is also in the set
    """
    leaves = [n for n in tree.nodes() if tree.degree(n) == 1]
    if [n for n in tree.nodes() if tree.degree(n) > 2]:
        joints = dict()
        for leaf in leaves:
            if root_ind == leaf:
                continue
            cur = leaf
            while tree.degree(cur) < 3:
                preds = list(tree.predecessors(cur))
                if len(preds) != 1:
                    breakpoint()
                cur = preds[0]
            joints[cur] = joints.get(cur, []) + [leaf]
        r_set = [root_ind]
        for j, j_leaves in joints.items():
            if len(j_leaves) > 2:
                breakpoint()
            r_set.append(j_leaves[0])
        ntable = {}
        ancs = dict(nx.tree_all_pairs_lowest_common_ancestor(tree))
        dists = dict(nx.all_pairs_shortest_path_length(tree))
        for k in range(len(tree)):
            for i in range(len(tree)):
                for j in range(len(tree)):  
                    if i == j: continue
                    ik = ancs[(i,k)] if (i,k) in ancs else ancs[(k,i)]
                    jk = ancs[(j,k)] if (j,k) in ancs else ancs[(k,j)]
                    d1 = dists[ik][i]+dists[ik][k]
                    d2 = dists[jk][j]+dists[jk][k]    
                    if d1 == d2:
                        ntable[k] = ntable.get(k, []) + [(i, j)]
        for k in ntable:
            ntable[k] = set(ntable[k])
        # greedily remove redundant landmarks
        # later try to prove this will be optimal
        def eval_r_set(r_set):
            cur_set = None
            for r in r_set:
                if cur_set is None:
                    cur_set = deepcopy(ntable[r])
                else:
                    cur_set &= ntable[r]
            return cur_set
        assert eval_r_set(r_set) == set()

        while True:
            for i in range(len(r_set)-1,-1,-1): # save root for last
                new_r_set = r_set[:i]+r_set[i+1:]
                if eval_r_set(new_r_set) == set():
                    r_set = new_r_set
                    break
            if eval_r_set(r_set):
                break
            if i == 0:
                break
            
    else:
        r_set = [root_ind]
    return r_set
            

def get_bool_mask(i):
    return list(map(bool, map(int, format(i,'b'))))



def process_syntree_mask(i, sk, min_r_set):
    sk.reset(min_r_set)
    zero_mask_inds = np.where(sk.mask == 0)[0]    
    bool_mask = get_bool_mask(i)
    sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
    node_mask, X, y = sk.get_state()
    return (node_mask, X, y)



def process_syntree(syntree, min_r_set, index):
    sk = Skeleton(syntree, index)                    
    # check inds 
    zero_mask_inds = np.where(sk.mask == 0)[0]
    # the rest is determined
    node_masks = np.zeros((0, len(sk.tree)))
    Xs, ys = np.zeros((0, 4097)), np.zeros((0, 257))
    for i in range(2**(len(sk.tree)-len(min_r_set))): # num 0's left
        node_mask, X, y = process_syntree_mask(i, sk, zero_mask_inds)
        node_masks = np.concatenate((node_masks, node_mask), axis=0)
        Xs = np.concatenate((Xs, X), axis=0)
        ys = np.concatenate((ys, y), axis=0)
    return node_masks, Xs, ys


if __name__ == "__main__":
    args = get_args()
    syntree_collection = SyntheticTreeSet()
    syntrees = syntree_collection.load(args.input_file)

    if os.path.exists(args.skeleton_file):
        skeletons = pickle.load(open(args.skeleton_file, 'rb'))
    else:
        sts = []
        for st in syntree_collection.sts:
            if st: 
                try:
                    st.build_tree()
                except:
                    breakpoint()
                sts.append(st)
            else:
                breakpoint()
        
        skeletons = {}
        for i, st in tqdm(enumerate(sts)):
            done = False
            for sk in skeletons:
                if st.is_isomorphic(sk): 
                    done = True
                    skeletons[sk].append(st)
                    break
                    
            if not done: 
                skeletons[st] = [st]
                
        for k, v in skeletons.items():
            print(f"count: {len(v)}") 

        pickle.dump(skeletons, open(os.path.join(args.visualize_dir, 'skeletons.pkl'), 'wb+'))
    
    sk_set = SkeletonSet().load_skeletons(skeletons)

    for index, st in enumerate(skeletons):
        # figure out "a" minimal resolving set
        if index < 2:
            continue
        sk = Skeleton(st, index)
        min_r_set = compute_md(sk.tree, sk.tree_root)  
        edge_index = np.array(sk.tree.edges).T           
        pargs = []
        for syntree in skeletons[st]:
            sk = Skeleton(syntree, index)
            for i in range(2**(len(sk.tree)-len(min_r_set))):
                pargs.append([i, sk, min_r_set])
        print(f"mapping {len(pargs)} for index {index}")
        batch_size = 32*2000
        # res = []
        for k in tqdm(range((len(pargs)+batch_size-1)//batch_size)): 
            print(k)       
            with Pool(20) as p:
                res = p.starmap(process_syntree_mask, pargs[batch_size*k:batch_size*k+batch_size])
            # if k == 2:
            #     res = []
            #     for parg in pargs[batch_size*k:batch_size*k+batch_size]:
            #         try:
            #             res.append(process_syntree_mask(parg))
            #         except:
            #             breakpoint()
            #     breakpoint()
            # with Pool(min(20, len(skeletons[st]))) as p:
            #     res = p.starmap(process_syntree_mask, tqdm(pargs))   

            node_masks = np.concatenate([r[0] for r in res], axis=0)
            Xs = np.concatenate([r[1] for r in res], axis=0)
            ys = np.concatenate([r[2] for r in res], axis=0)              
            np.save(os.path.join(args.output_dir, f"{index}_{k}_Xs.npy"), Xs)
            np.save(os.path.join(args.output_dir, f"{index}_{k}_ys.npy"), ys)
            np.save(os.path.join(args.output_dir, f"{index}_{k}_node_masks.npy"), node_masks)
            np.save(os.path.join(args.output_dir, f"{index}_edge_index.npy"), edge_index)

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
