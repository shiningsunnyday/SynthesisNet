from synnet.config import MAX_PROCESSES, MP_MIN_COMBINATIONS
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilter,
    ReactionTemplateFileHandler,
)
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton, SkeletonSet, Program, \
get_bool_mask
from synnet.utils.analysis_utils import count_bbs, count_rxns
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import dfs_tree, weisfeiler_lehman_graph_hash
from networkx.readwrite import json_graph
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from collections import deque, defaultdict
from copy import deepcopy

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
        "--skeleton-file",
        type=str,
        default="results/viz/skeletons.pkl",
        help="Input file for the skeletons of syntree-file",
    )   
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="",
        help="Where to visualize any figures",
    )        
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Intermediate results",
    ) 
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Store results",
    )
    # Hash table args
    parser.add_argument("--ncpu", type=int, default=1, help="Number of cpus")
    parser.add_argument("--depth", type=int, default=3, help="depth of enumeration")
    parser.add_argument("--top-bb", type=int)
    parser.add_argument("--top-rxn", type=int)
    parser.add_argument("--stats", type=str, nargs='+')
    parser.add_argument("--keep-prods", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def get_wl_kernel(tree: nx.digraph, fill_in=[]):
    for n in tree.nodes():
        if n in fill_in:
            if 'rxn_id' in tree.nodes[n]:
                rxn_id = tree.nodes[n]['rxn_id']
                tree.nodes[n]['id'] = rxn_id
            elif 'smiles' in tree.nodes[n]:
                smiles = tree.nodes[n]['smiles']
                tree.nodes[n]['id'] = smiles   
            else:
                breakpoint()
        else:
            tree.nodes[n]['id'] = 0
    return weisfeiler_lehman_graph_hash(tree, iterations=len(tree), node_attr='id')



def vis_table(args, table_all):
    for length in table_all:
        table = table_all[length]
        fig_path = os.path.join(args.visualize_dir, f'hash_table_size={length}.png')
        fig = plt.Figure()
        counts = list(table.values())
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(range(len(counts)), sorted(counts, key=lambda x:-x))
        ax.set_xlabel('filled-subtree hash')
        ax.set_ylabel('count')
        ax.set_yscale('log')
        ax.set_title(f'counts of length-{length} subtrees')
        fig.savefig(fig_path)
        print(f"visualized count at {fig_path}")    


def hash_st(st, index):
    sk = Skeleton(st, index)
    k_vals = []
    for node in sk.tree.nodes():
        if 'rxn_id' in sk.tree.nodes[node]:
            g = nx.dfs_tree(sk.tree, node)
            for n in g.nodes():
                for k in sk.tree.nodes[n]:
                    g.nodes[n][k] = sk.tree.nodes[n][k]

            bbs = [n for n in g.nodes() if list(g.successors(n)) == []]
            fill_in = [node] + [bbs]
            k_val = get_wl_kernel(g, fill_in=fill_in)
            k_vals.append((len(g), k_val))
    return k_vals


def get_programs(rxns, keep_prods=0, size=1):
    progs = []
    if size == 1:
        for i, rxn in enumerate(rxns):
            g = nx.DiGraph()
            g.add_node(0, rxn_id=i)
            g.nodes[0]['depth'] = 1
            # g.graph['super'] = False
            prog = Program(g, keep_prods=keep_prods)
            progs.append(prog)
        return {1: progs}
    all_progs = get_programs(rxns, keep_prods=keep_prods, size=size-1)
    for i, r in enumerate(rxns):
        if r.num_reactant == 1:
            A = all_progs[size-1]
            for a in A:
                prog = deepcopy(a)
                prog.add_rxn(i, len(a.rxn_tree)-1)
                progs.append(prog)
        else:
            for j in range(1, size-1):
                A = all_progs[j]
                B = all_progs[size-1-j]
                for a in A:
                    for b in B:                
                        prog = deepcopy(a).combine(deepcopy(b))
                        prog.add_rxn(i, 0, len(a.rxn_tree)-1)
                        progs.append(prog)
    all_progs[size] = progs
    return all_progs


def init_program(prog):    
    """
    Then runs the programs, returning validity and (if prog.keep_prods) storing intermediates
    """    
    prog.init_rxns(bbf.rxns)
    return prog
    


def run_program(prog):    
    """
    Then runs the programs, returning validity and (if prog.keep_prods) storing intermediates
    """    
    start_len, res = prog.run_rxn_tree()
    if start_len:
        print(f"{len(res)}/{start_len} pass")
        return prog
    else:
        return None




def expand_program(i, a, b=None):
    prog = deepcopy(a)
    if b is not None:
        prog = prog.combine(deepcopy(b))      
        prog.add_rxn(i, len(a.rxn_tree)-1, len(prog.rxn_tree)-1)
    else:
        prog.add_rxn(i, len(a.rxn_tree)-1)
    return prog



def expand_programs(all_progs, size):
    progs = []
    pargs = []
    for i, r in tqdm(enumerate(bbf.rxns)):
        if r.num_reactant == 1:
            A = all_progs[size-1]
            for a in A:
                pargs.append((i, a))
        else:
            for j in range(1, size-1):
                A = all_progs[j]
                B = all_progs[size-1-j]
                for a in A:
                    for b in B:   
                        pargs.append((i, a, b))
    with mp.Pool(20) as p:
        progs = p.starmap(expand_program, tqdm(pargs))
    # progs = []
    # for i, parg in enumerate(pargs):
    #     progs.append(expand_program(*parg))      
    all_progs[size] = progs
    return all_progs



def filter_programs(progs):
    new_progs = []
    for p in progs:
        if p is None:
            continue
        good = True
        for e in p.entries:
            if np.prod([len(reactants) for reactants in p.rxn_map[e].available_reactants]):
                continue
            good = False
            break
        if good:
            new_progs.append(p)

    return new_progs



def create_run_programs(args, bbf, size=3):     
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    for d in range(1, size+1):
        cache_fpath = os.path.join(args.cache_dir, f"{d}.pkl")
        exist = os.path.exists(cache_fpath)
        if args.cache_dir and exist:
            all_progs = pickle.load(open(cache_fpath, 'rb'))
            print(f"loaded {len(all_progs[d])} size-{d} programs")
            all_progs[d] = filter_programs(all_progs[d])
            assert d in all_progs
            continue
        if d == 1: 
            progs = get_programs(bbf.rxns, args.keep_prods, size=1)
            all_progs = progs
        else:  
            cache_fpath_pre = cache_fpath.replace(f"{d}.pkl", f"{d}_pre.pkl")
            if args.cache_dir and os.path.exists(cache_fpath_pre):
                all_progs = pickle.load(open(cache_fpath_pre, 'rb'))
            else:
                print(f"expanding size-{d} programs")
                expand_programs(all_progs, d)
                if args.cache_dir:
                    pickle.dump(all_progs, open(cache_fpath_pre, 'wb'))
            print(f"created {len(all_progs[d])} size-{d} programs")

        with mp.Pool(bbf.processes) as p:
            all_progs[d] = p.map(init_program, tqdm(all_progs[d]))          
        # Filter after init prunes the input space
        all_progs[d] = filter_programs(all_progs[d])
        print(f"running {len(all_progs[d])} size-{d} programs")

        """
        Strategy: use mp to run easy programs in parallel
        Run hard programs sequentially, use mp among the input combinations
        """
        easy_prog_inds, hard_prog_inds = [], []
        for i, p in enumerate(all_progs[d]):
            if Program.input_length(p) <= MP_MIN_COMBINATIONS:
                easy_prog_inds.append(i)
            else:
                hard_prog_inds.append((Program.input_length(p), i))
        print(sorted(hard_prog_inds))
        hard_prog_inds = [i for _, i in sorted(hard_prog_inds)]
        progs = [None for _ in all_progs[d]]
       
        easy_done_path = os.path.join(args.cache_dir, f"{d}_easy.pkl")
        if os.path.exists(easy_done_path):
            easy_progs = pickle.load(open(easy_done_path, 'rb'))
        else:
            with mp.Pool(bbf.processes) as p:
                easy_progs = p.map(run_program, tqdm([all_progs[d][i] for i in easy_prog_inds]))             
            # easy_progs = [run_program(all_progs[d][i]) for i in easy_prog_inds]
            if args.cache_dir:
                pickle.dump(easy_progs, open(easy_done_path, 'wb'))
        for i, p in zip(easy_prog_inds, easy_progs):
            progs[i] = p       
        for i in tqdm(hard_prog_inds):
            hard_path_i = os.path.join(args.cache_dir, f"{d}_hard_{i}.pkl")
            print(hard_path_i)
            if os.path.exists(hard_path_i):
                continue
            else:
                p = all_progs[d][i]
                progs[i] = run_program(p)    
                if args.cache_dir:
                    pickle.dump(progs[i], open(hard_path_i, 'wb'))
        for i in tqdm(hard_prog_inds):
            hard_path_i = os.path.join(args.cache_dir, f"{d}_hard_{i}.pkl")
            progs[i] = pickle.load(open(hard_path_i, 'rb'))
        # Filter after reaction is run
        all_progs[d] = filter_programs(progs)
        print(f"done! {len(all_progs[d])} size-{d} programs")

        if args.cache_dir and not exist:
            pickle.dump(all_progs, open(cache_fpath, 'wb'))
    return all_progs


def hash_program(prog, output_dir):
    """
    We perform bfs search on all possible unmasks satisfying topological order.
    Each unmask can expand a reaction on the frontier.
    Each unmask is an int (binary mask over the nodes 0...len(prog)-1)
    """
    tree = prog.rxn_tree
    if sorted(list(tree.nodes())) != list(tree.nodes()):
        breakpoint()
    bfs = deque()    
    start = ''.join(['0' for _ in range(len(tree)-1)] + ['1'])
    vis = dict({start: 1})
    start_hash = prog.hash(list(map(int, start)))
    start_dirname = os.path.join(output_dir, start_hash)
    bfs.append((start, Path(start_dirname)))
    edges = np.array(tree.edges, dtype=np.int32) if len(tree.edges) else np.empty((0, 2), dtype=np.int32)
    
    while bfs:
        cur, cur_dirname = bfs.popleft()
        os.makedirs(cur_dirname, exist_ok=True)
        cur_fpath = cur_dirname.parent / f"{cur_dirname.name}.json"        
        mask = np.array(list(map(int, cur)))                
        frontier = [f for f in edges[mask[edges[:, 0]] == 1][:, 1] if not mask[f]]
        if os.path.exists(cur_fpath):   
            data = json.load(open(cur_fpath, 'r'))          
            assert 'rxn_ids' in data
        else:            
            data = {'rxn_ids': {},
                    'mask': mask.tolist(), # debug
                    'tree': prog.hash(mask, return_json=True) # debug
                    }
            
        # make/append to json file the continuation
        for f in frontier:
            f = f.item()
            data['rxn_ids'][f] = data['rxn_ids'].get(f, []) + [tree.nodes[f]['rxn_id']]
   
        json.dump(data, open(cur_fpath, 'w+'))
        # debug
        vis_fpath = cur_dirname.parent / f"{cur_dirname.name}.png"        
        if not os.path.exists(vis_fpath):
            T = json_graph.tree_graph(data['tree'])
            node_label = {}
            node_color = []
            for n in T.nodes():
                node_label[n] = f"id={n}"
                if 'rxn_id' in T.nodes[n]:
                    rxn_id = T.nodes[n]['rxn_id']
                    rxn_label = f"\nrxn_id={rxn_id}"
                    node_color.append('red')
                else:
                    rxn_label = "\nrxn_id=?"
                    node_color.append('gray')
                node_label[n] += rxn_label
                depth_label = T.nodes[n]['depth']
                node_label[n] += f"\ndepth={depth_label}"
                                      
            T = nx.relabel_nodes(T, node_label)
            fig = plt.Figure(figsize=(10, 10))
            ax = fig.add_subplot(1,1,1)
            pos = Skeleton.hierarchy_pos(T, root=node_label[len(T)-1])
            nx.draw(T, 
                    ax=ax, 
                    pos=pos, 
                    node_size=5000, 
                    node_color=node_color,
                    with_labels=True)
            fig.savefig(vis_fpath)



        for f in frontier:
            if cur[f] != '0':
                breakpoint()
            cur = cur[:f] + '1' + cur[f+1:]
            if cur in vis:
                continue
            cur_hash = prog.hash(list(map(int, cur)))
            vis[cur] = 1
            bfs.append((cur, cur_dirname / cur_hash))
            cur = cur[:f] + '0' + cur[f+1:]


    
            



def hash_programs(all_progs, output_dir):
    """
    We create a recursive directory in output_dir with all programs and their partial
    program representations.
    Each first-level directory in output_dir corresponds to one empty rooted tree structure.
    For each program, we enumerate its partial programs by enumerating all topological sort paths.
    We mask out every reaction except the top, then unmask one-by-one.
    """
    for d in all_progs:
        for p in all_progs[d]:
            hash_program(p, output_dir)

            



if __name__ == "__main__":

    # Parse input args
    args = get_args()
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)    
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)

    if os.path.exists(args.skeleton_file):
        # Use to filter building blocks
        skeletons = pickle.load(open(args.skeleton_file, 'rb'))            
        if args.top_bb: 
            bb_counts = count_bbs(args, skeletons, vis=False)
            for bblock in bblocks:
                bb_counts[bblock]
            bblocks = sorted(bb_counts.keys(), key=lambda x:-bb_counts[x])                       
            bblocks = bblocks[:args.top_bb]
            print(f"top bb have counts: {[bb_counts[x] for x in bblocks]}")                
        if args.top_rxn:            
            rxn_counts = count_rxns(args, skeletons, rxn_templates, vis=False)
            for i, rxn in enumerate(rxn_templates):
                rxn_counts[rxn]            
            rxn_templates = sorted(rxn_templates, key=lambda x:-rxn_counts[x])        
            rxn_templates = rxn_templates[:args.top_rxn]
            print(f"top rxn have counts: {[rxn_counts[x] for x in rxn_templates]}")
        

    # debug
    # test_st = list(skeletons.keys())[0]
    # bblock_inds = [bblocks.index(n.smiles) for n in test_st.chemicals if n.smiles in bblocks]
    # bblocks = [bblocks[ind] for ind in bblock_inds]
    # rxn_templates = [rxn_templates[r.rxn_id] for r in test_st.reactions]


    bbf = BuildingBlockFilter(
        building_blocks=bblocks,
        rxn_templates=rxn_templates,
        verbose=args.verbose,
        processes=args.ncpu,
    )    

    # Count number of unique (uni-reaction, building block) pairs
    bbf._init_rxns_with_reactants()
    bbf.filter()

    # Run programs      
    # progs = get_programs(bbf.rxns, size=2)
    all_progs = create_run_programs(args, bbf, size=args.depth)

    if args.stats:
        os.makedirs(args.visualize_dir, exist_ok=True)
        for stat in args.stats:
            fpath = os.path.join(args.visualize_dir, f"{stat}.png")            
            fig = plt.Figure()
            ax = fig.add_subplot(1,1,1)
            if stat == 'program-count':
                lengths = [len(all_progs[d]) for d in all_progs]
                ax.plot(lengths)
                ax.set_xlabel("depth")
                ax.set_ylabel('number of programs')
            elif stat == 'input-length':
                input_lengths = [Program.avg_input_length(all_progs[d]) for d in all_progs]
                ax.plot(input_lengths)
                ax.set_xlabel("depth")
                ax.set_ylabel('number of building block input sets')
            fig.savefig(fpath)    
    

    if args.output_dir:
        hash_programs(all_progs, args.output_dir)



# sts = []
# for index, sk in enumerate(skeletons):
#     for st in skeletons[sk]:
#         sts.append([st, index])

# if args.ncpu == 1:
#     res = [hash_st(st, index) for st, index in sts]
# else:
#     with Pool(args.ncpu) as p:
#         res = p.starmap(hash_st, tqdm(sts))
# res = [k_val for r in res for k_val in r]
# table = defaultdict(lambda: defaultdict(int))
# for length, k_val in res:
#     table[length][k_val] += 1
# vis_table(args, table)
