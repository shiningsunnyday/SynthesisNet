from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton, SkeletonSet
import pickle
import os
import networkx as nx
from networkx.algorithms import weisfeiler_lehman_graph_hash
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
        "--anchor_type",
        choices=['rset', 'leaves', 'target'],
        default='rset',
        help="What constitutes anchors"
    )
    parser.add_argument("--predict_anchor", action='store_true')    
    parser.add_argument(
        "--determine_criteria",
        choices=['leaves_up', 'all_leaves', 'rxn_frontier', 'bb_frontier'],
        default='leaves_up',
        help="""
        Criteria for a determined skeleton:
            leaves_up: all children present
            all_leaves: all leaves present
            rxn_frontier: there exists rxn on bfs frontier, predict rxns
            bb_frontier: if there exists rxn on bfs frontier, predict rxns only; else predict all bfs frontier
        """
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
            for i in range(len(r_set)-1,0,-1): # don't remove root
                new_r_set = r_set[:i]+r_set[i+1:]
                if eval_r_set(new_r_set) == set():
                    r_set = new_r_set
                    break
            if eval_r_set(r_set):
                break
            if i == 1:
                break
            
    else:
        r_set = [root_ind]
    return r_set
            

def get_bool_mask(i):
    return list(map(bool, map(int, format(i,'b'))))



def get_wl_kernel(tree: nx.digraph, fill_in=[]):
    for n in tree.nodes():
        tree.nodes[n]['id'] = 0
    for i, n in enumerate(fill_in):
        tree.nodes[n]['id'] = i+1
    return weisfeiler_lehman_graph_hash(tree, iterations=len(tree), node_attr='id')



def process_syntree_mask(i, sk, args, min_r_set, anchors=None):
    if args.determine_criteria in ['leaves_up', 'all_leaves']:
        leaves_up = True
        rxn_frontier = False
        bb_frontier = False
    elif args.determine_criteria == 'rxn_frontier':
        leaves_up = False
        rxn_frontier = True
        bb_frontier = False
    elif args.determine_criteria == 'bb_frontier':
        leaves_up = False
        rxn_frontier = True
        bb_frontier = True        
    if anchors is not None:
        poss_vals = []
        val = get_wl_kernel(sk.tree, min_r_set[:2+len(anchors)])
        sk.reset(min_r_set[:1+len(anchors)])
        for poss in min_r_set[len(anchors)+1:]:
            poss_val = get_wl_kernel(sk.tree, min_r_set[:1+len(anchors)] + [poss])
            if poss_val == val:
                poss_vals.append(poss)
        if len(poss_vals) > 1:
            breakpoint()
        # featurize prediction problem of next anchor, which can be any of poss_vals
        node_mask, X, y = sk.get_partial_state(poss_vals, min_r_set[len(anchors)+1])
        return (node_mask, X, y, sk.tree.nodes[sk.tree_root]['smiles'])
    else:
        sk.reset(min_r_set)
        zero_mask_inds = np.where(sk.mask == 0)[0]    
        bool_mask = get_bool_mask(i)
        sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
        node_mask, X, y = sk.get_state(leaves_up, rxn_frontier, bb_frontier)        
        if args.determine_criteria == 'all_leaves':
            assert sk.all_leaves
        return (node_mask, X, y, sk.tree.nodes[sk.tree_root]['smiles'])        


def test_is_leaves_up(i, sk, min_r_set):
    sk.reset(min_r_set)
    zero_mask_inds = np.where(sk.mask == 0)[0]    
    bool_mask = get_bool_mask(i)
    sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
    return sk.leaves_up


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
        for i in range(2**(len(sk.tree)-len(min_r_set))):
            sk.reset(min_r_set)
            zero_mask_inds = np.where(sk.mask == 0)[0]
            bool_mask = get_bool_mask(i)
            sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
            if getattr(sk, determine_criteria):
                pargs.append([i, sk, args, min_r_set])
    return pargs


def main():
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
    len_inds = np.argsort([len(skeletons[st]) for st in skeletons])
    kth_largest = np.zeros(len(skeletons))
    kth_largest[len_inds] = np.arange(len(skeletons))[::-1]
    for index, st in tqdm(enumerate(skeletons)):
        if len(skeletons[st]) < 100:
            continue
        # if index < 3:
        #     continue
        # if index < 2:
        #     continue
        # figure out "a" minimal resolving set
        if index < 2:
            continue
        sk = Skeleton(st, index)
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
        
        # with Pool(50) as p:
            # pargs = p.starmap(get_parg, tqdm([[syntree, min_r_set, index, args] for syntree in tqdm(skeletons[st])]))
        pargs = [get_parg(*[syntree, min_r_set, index, args]) for syntree in tqdm(skeletons[st])]
        pargs = [parg for parg_sublist in pargs for parg in parg_sublist]
        print(f"mapping {len(pargs)} for class {index} which is {kth_largest[index]+1}th most represented")
        batch_size = 1000*len(pargs)//len(skeletons[st])
        # batch_size = 1    
        num_batches = (len(pargs)+batch_size-1)//batch_size
        print(f"{num_batches} batches")
        for k in tqdm(range(num_batches)): 
            if os.path.exists(os.path.join(args.output_dir, f"{index}_{k}_node_masks.npy")):
                continue
            # print(k)  
            res = []     
            if batch_size > 1:
                try:
                    with Pool(50) as p:
                        res = p.starmap(process_syntree_mask, pargs[batch_size*k:batch_size*k+batch_size])
                except:
                    breakpoint()
                    res = [process_syntree_mask(*pargs[j]) for j in range(batch_size*k, batch_size*k+batch_size)]
            else:
                res.append(process_syntree_mask(*pargs[k]))            
            res = [r for r in res if r is not None]                       
            node_masks = np.concatenate([r[0] for r in res], axis=0)
            Xs = np.concatenate([r[1] for r in res], axis=0)
            ys = np.concatenate([r[2] for r in res], axis=0)              
            smiles = np.array([r[3] for r in res])
            np.save(os.path.join(args.output_dir, f"{index}_{k}_Xs.npy"), Xs)
            np.save(os.path.join(args.output_dir, f"{index}_{k}_ys.npy"), ys)
            np.save(os.path.join(args.output_dir, f"{index}_{k}_smiles.npy"), smiles)
            np.save(os.path.join(args.output_dir, f"{index}_{k}_node_masks.npy"), node_masks)
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
            