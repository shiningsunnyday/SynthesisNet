from synnet.config import MAX_PROCESSES, MP_MIN_COMBINATIONS, PRODUCT_DIR, NUM_THREADS
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilter,
    ReactionTemplateFileHandler,
)
from synnet.utils.data_utils import Skeleton, Program, ProductMap, ProductMapLink
from synnet.utils.logging import create_logger
from synnet.utils.analysis_utils import count_bbs, count_rxns
import pickle
import logging
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import dfs_tree, weisfeiler_lehman_graph_hash
from networkx.readwrite import json_graph
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pprint
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
        help="Where to visualize any figures or log the progress",
    )        
    parser.add_argument("--log_file", required=True)
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
    # Slurm args
    parser.add_argument("--step", choices=['expand', 'init', 'run', 'migrate'])
    parser.add_argument("--d", default=1, type=int)
    parser.add_argument("--batch", default=-1, type=int, help='which batch, i.e. expand_batch_size (expand), init_batch_size (init), run_batch_size (run)')
    # Hash table args
    parser.add_argument("--ncpu", type=int, default=1, help="Number of cpus")
    parser.add_argument("--expand_batch_size", type=int, default=100, help="Number of pargs to batch for expand")
    parser.add_argument("--init_batch_size", type=int, default=100, help="Number of programs to batch for init")
    parser.add_argument("--run_batch_size", type=int, default=100, help="Number of programs to batch for run")
    parser.add_argument("--mp-min-combinations", type=int, default=1000000, help="Min combinations to run mp")    
    parser.add_argument("--num-threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--depth", type=int, default=3, help="depth of enumeration")
    parser.add_argument("--top-bb", type=int, 
                        help='if positive, use only top-k bb/rxns as counted by skeleton-file; if -1, use all with non-zero count')
    parser.add_argument("--top-rxn", type=int, help='use only top-k bb/rxns as counted by skeleton-file; if -1, use all with non-zero count')
    parser.add_argument("--stats", type=str, nargs='+')
    parser.add_argument("--keep-prods", type=int, default=0)
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
    return all_progs


def init_program(prog):    
    """
    Then runs the programs, returning validity and (if prog.keep_prods) storing intermediates
    """   
    logger = logging.getLogger('global_logger')
    logger.info(f"begin init program {prog.logging_info()}")
    prog.init_rxns(bbf.rxns)
    logger.info(f"done init program {prog.logging_info()}")
    return prog
    


def run_program(prog):    
    """
    Then runs the programs, returning validity and (if prog.keep_prods) storing intermediates
    """    
    logger = logging.getLogger('global_logger')
    logger.info(f"begin run program {prog.logging_info()}")
    start_len, res = prog.run_rxn_tree()
    logger.info(f"done run program {prog.logging_info()}")
    if start_len:
        print(f"{len(res)}/{start_len} pass")
        return prog
    else:
        return None



def expand_program(i, a, b=None):   
    logger = logging.getLogger('global_logger')   
    if a != [] and b != []:
        prog_A_info = a.logging_info()
        if b is None:
            task_descr = f"joining program {prog_A_info} with rxn_id {i}"
        else:
            prog_B_info = b.logging_info()
            task_descr = f"joining program {prog_A_info} with {prog_B_info} with rxn_id {i}"      
        logger.info(f"begin {task_descr}")
        if isinstance(a.product_map, ProductMap):
            prog = a.copy()
            if b is None:            
                prog.add_rxn(i, len(a.rxn_tree)-1)
            else:
                prog = prog.combine(b.copy())      
                prog.add_rxn(i, len(a.rxn_tree)-1, len(prog.rxn_tree)-1)
                prog.product_map.unload()             
        else:
            assert not a.product_map._loaded
            prog = a.copy()
            if b is None:
                prog.add_rxn(i, len(a.rxn_tree)-1)
            else:
                prog = prog.combine(b.copy())        
                prog.add_rxn(i, len(a.rxn_tree)-1, len(prog.rxn_tree)-1)    
        logger.info(f"done {task_descr}")
    else:
        # i is a bi-mol reaction, so i is also in _entries
        # first: update _entries at the reactant_idx level properly
        # then: add_rxn, updating the correct 'child'        
        if b == []: # (i, a, []) means a is left child of i
            prog_A_info = a.logging_info()
            task_descr = f"joining program {prog_A_info}, [] with rxn_id {i}"
            logger.info(f"begin {task_descr}")
            if isinstance(a.product_map, ProductMap):
                prog = a.copy()
            else:
                assert not a.product_map._loaded
                prog = a.copy()
            prog.combine_bi_mol(i, child='left')
        else:
            prog_B_info = b.logging_info()
            task_descr = f"joining program [], {prog_B_info} with rxn_id {i}"
            logger.info(f"begin {task_descr}")
            if isinstance(b.product_map, ProductMap):
                prog = b.copy()                
            else:
                assert not b.product_map._loaded
                prog = b.copy()
            prog.combine_bi_mol(i, child='right')
        logger.info(f"done {task_descr}")
    return prog
        


def expand_programs(args, all_progs, size):
    logger = logging.getLogger('global_logger')
    prefix = f"{size}_expand"
    progs = []
    parg_path = os.path.join(args.cache_dir, f"{prefix}_pargs.pkl")
    if os.path.exists(parg_path):
        pargs = pickle.load(open(parg_path, 'rb'))
    else:
        pargs = []
        for i, r in tqdm(enumerate(bbf.rxns)):
            if r.num_reactant == 1:
                A = all_progs[size-1]
                for a in range(len(A)):
                    pargs.append((i, A[a]))
            else:
                for j in range(1, size-1):
                    A = all_progs[j]
                    B = all_progs[size-1-j]
                    for a in range(len(A)):
                        for b in range(len(B)):   
                            pargs.append((i, A[a], B[b]))
                A = all_progs[size-1]
                for a in range(len(A)):
                    pargs.append((i, A[a], [])) # a is left child
                    pargs.append((i, [], A[a])) # a is right child
                # TODO: add all the cases of only one reaction, but i is also entry
        pickle.dump(pargs, open(parg_path, 'wb+'))    
        if args.step == 'expand' and args.batch == -1: # job to create pargs
            logger.info(f"prepared {len(pargs)} pargs")
            raise
    """
    Create a hash of args, all_progs, size
    Only if EVERYTHING is the same, use checkpointing
    """                
    num_batches = (len(pargs)+args.expand_batch_size-1)//args.expand_batch_size
    logger.info(f"{num_batches} batches to expand")    
    if args.step is None or (args.step == 'expand' and args.batch == -1):
        batch_iter = range(num_batches)
    else:        
        assert args.step == 'expand'
        batch_iter = [args.batch]
    all_progs[size] = []
    for j in batch_iter:
        expand = False
        if os.path.exists(os.path.join(args.cache_dir, f"{prefix}_{j}.pkl")):
            logger.info(f"loading {prefix}_{j}.pkl")
            progs = pickle.load(open(os.path.join(args.cache_dir, f"{prefix}_{j}.pkl"), 'rb'))
            # check all the paths exist
            for p in progs:
                if isinstance(p.product_map, ProductMap):
                    if not os.path.exists(p.product_map.fpath):
                        expand = True
                else:
                    for fpath in p.product_map.fpaths.values():
                        if not os.path.exists(fpath):
                            expand = True
        else:
            expand = True
        if expand:
            pargs_batch = pargs[j*args.expand_batch_size:(j+1)*args.expand_batch_size]        
            logger.info(f"=====expanding {len(pargs_batch)} programs (batch {j}/{num_batches})=====")
            if args.ncpu > 1:
                with mp.Pool(args.ncpu) as p:
                    progs = p.starmap(expand_program, tqdm(pargs_batch, desc="expanding progs"))
            else:            
                progs = []
                for i, parg in enumerate(tqdm(pargs_batch, desc="expanding progs")):       
                    progs.append(expand_program(*parg))      
            logger.info(f"dumping {prefix}_{j}.pkl")
            pickle.dump(progs, open(os.path.join(args.cache_dir, f"{prefix}_{j}.pkl"), 'wb+'))
        all_progs[size] += progs
    
    return all_progs



def filter_programs(progs):
    """
    We filter out all progs for which there's no possible reactant combination
    Specifically, we filter all progs where the product of possible reactants is 0
    """
    logger = logging.getLogger('global_logger')
    logger.info(f"begin filtering {len(progs)} programs")
    new_progs = []
    for p in progs:
        if p is None:
            continue
        good = True
        for e in p.entries:
            if isinstance(e, tuple):
                r, idx = e
                poss_reactants = len(p.rxn_map[r].available_reactants[idx])
            else:
                poss_reactants = np.prod([len(reactants) for reactants in p.rxn_map[e].available_reactants])
            if poss_reactants:
                continue
            good = False
            break
        if good:
            new_progs.append(p)
    logger.info(f"done filtering {len(progs)}->{len(new_progs)} programs")    
    return new_progs



def strategy(progs, rxns):
    easy_prog_inds, hard_prog_inds = [], []
    for i, p in enumerate(progs):
        if Program.input_length(p, rxns) <= MP_MIN_COMBINATIONS:
            easy_prog_inds.append(i)
        else:
            hard_prog_inds.append(i)
    return easy_prog_inds, hard_prog_inds


def get_descr(all_progs):
    descr = "\nLet's summarize the current status of all programs\n"
    for d in all_progs:
        descr += f"{len(all_progs[d])} depth-{d} programs\n"
        avg_input_length = Program.avg_input_length(all_progs[d])
        max_input_length = max([Program.input_length(p) for p in all_progs[d]])
        descr += f"with average input length {avg_input_length}\n"
        descr += f"and maximum input length {max_input_length}\n"
    descr += "\n"
    return descr


def get_cache_fpaths(all_progs):
    fpaths = []
    for d in all_progs:
        for p in all_progs[d]:
            if isinstance(p.product_map, ProductMap):
                fpath = p.product_map.fpath
                if not os.path.exists(fpath):
                    print(fpath)
                    raise
                fpaths.append(fpath)
            else:
                for fpath in p.product_map.fpaths.values():
                    if not os.path.exists(fpath):
                        print(fpath)
                        raise
                    fpaths.append(fpath)
    return fpaths


def clean_cache(args, all_progs):
    logger = logging.getLogger('global_logger')
    fpath_set = set(get_cache_fpaths(all_progs))    
    logger.info(f"begin cleaning cache, keep {len(fpath_set)} fpaths")
    removed = 0
    for f in os.listdir(args.cache_dir):        
        if '.pkl' in f:
            continue
        if os.path.join(args.cache_dir, f) not in fpath_set:
            os.remove(os.path.join(args.cache_dir, f))
            removed += 1
    logger.info(f"finish cleaning cache, removed {removed}")       



def migrate(args, all_progs):
    """
    Now, it's time to convert all ProductMap to ProductMapLinks
    using product maps in all_progs[d-1] as the "base" files
    """   
    logger = logging.getLogger('global_logger')   
    logger.info("begin migrating to ProductMapLink")                 
    for depth in all_progs:
        logger.info(f"begin migrating depth {depth}")                 
        if args.ncpu > 1:
            with mp.Pool(args.ncpu) as p:
                all_progs[depth] = p.map(Program.migrate, tqdm(all_progs[depth]))
        else:
            for p in tqdm(all_progs[depth]):
                Program.migrate(p)                
        logger.info(f"done migrating depth {depth}")          
    logger.info(f"done migrating depth")     


def expand(args, all_progs, d):
    logger = logging.getLogger('global_logger')   
    cache_fpath_pre = os.path.join(args.cache_dir, f"{d}_pre.pkl")
    if d == 1: 
        progs = get_programs(bbf.rxns, args.keep_prods, size=1)
        all_progs = progs
        if args.cache_dir:
            if args.step is None or (args.step == 'expand' and args.batch == -1):
                logger.info(f"begin cache-dumping all pre-programs at {cache_fpath_pre}")                     
                pickle.dump(all_progs, open(cache_fpath_pre, 'wb'))
                logger.info(f"done cache-dumping all pre-programs at {cache_fpath_pre}")         
    else:          
        if args.cache_dir and os.path.exists(cache_fpath_pre):
            all_progs = pickle.load(open(cache_fpath_pre, 'rb'))
        else:
            logger.info(f"begin expanding size-{d} programs")                
            expand_programs(args, all_progs, d)
            logger.info(f"done expanding size-{d} programs")
            if args.cache_dir:
                if args.step is None or (args.step == 'expand' and args.batch == -1):
                    logger.info(f"begin cache-dumping all pre-programs at {cache_fpath_pre}")                     
                    pickle.dump(all_progs, open(cache_fpath_pre, 'wb'))
                    logger.info(f"done cache-dumping all pre-programs at {cache_fpath_pre}")                   
        logger.info(f"created {len(all_progs[d])} size-{d} programs")    


def load_and_filter(args, all_progs, d):
    logger = logging.getLogger('global_logger')   
    cache_fpath = os.path.join(args.cache_dir, f"{d}.pkl")
    all_progs = pickle.load(open(cache_fpath, 'rb'))
    logger.info(f"loaded {len(all_progs[d])} size-{d} programs")
    all_progs[d] = filter_programs(all_progs[d])
    assert d in all_progs
    return all_progs
    # """
    # Sanity checks
    # """
    # for f in get_cache_fpaths(all_progs):
    #     assert os.path.exists(f)
    return all_progs



def init_and_filter(args, all_progs, d):
    """
    Strategy: use mp to init easy programs in parallel
    Run hard programs sequentially, use mp to filter the intermediates
    """    
    logger = logging.getLogger('global_logger')   
    logger.info(f"strategize how to init {len(all_progs[d])} depth-{d} programs")  
    progs = [None for _ in all_progs[d]] 
    easy_prog_inds, hard_prog_inds = strategy(all_progs[d], bbf.rxns)
    logger.info(f"parallel run easy programs {easy_prog_inds}")          
    cache_fpath_init = os.path.join(args.cache_dir, f"{d}_init.pkl")
    # Batch using init_batch_size
    num_batches = (len(easy_prog_inds)+args.init_batch_size-1)//args.init_batch_size
    logger.info(f"prepared {num_batches} batches to init")
    prefix = f"{d}_init"    
    if args.step is None or (args.step == 'init' and args.batch == -1):
        batch_iter = range(num_batches)
    else:        
        assert args.step == 'init'
        assert len(hard_prog_inds) == 0
        batch_iter = [args.batch]    
    all_inds = range(len(all_progs[d]))
    for j in batch_iter:
        inds = all_inds[j*args.init_batch_size:(j+1)*args.init_batch_size]
        progs_batch = run_or_init_batch(f"{prefix}_{j}", all_progs[d], inds, easy_prog_inds, hard_prog_inds)
        for i, p in zip(inds, progs_batch):
            progs[i] = p 

    if args.step is None or (args.step == 'init' and args.batch == -1):
        # Filter after init prunes the input space
        all_progs[d] = filter_programs(progs)       
        pickle.dump(all_progs, open(cache_fpath_init, 'wb'))
        logger.info(f"done init and filter, {len(all_progs[d])} programs")


def run_or_init_batch(prefix, d_progs, inds, easy_prog_inds, hard_prog_inds):
    logger = logging.getLogger('global_logger')   
    easy_prog_inds_batch = [ind for ind in inds if ind in easy_prog_inds]
    hard_prog_inds_batch = [ind for ind in inds if ind in hard_prog_inds]
    # if 'run' in prefix and len(hard_prog_inds_batch) == 0:
    #     raise
    batch_path = os.path.join(args.cache_dir, f"{prefix}.pkl")
    easy_batch_path = os.path.join(args.cache_dir, f"{prefix}_easy.pkl")
    hard_batch_path = os.path.join(args.cache_dir, f"{prefix}_hard.pkl")
    if 'run' in prefix:
        func = run_program
    else:
        func = init_program
    if os.path.exists(batch_path):
        logger.info(f"loading {prefix}.pkl")
        progs_batch = pickle.load(open(batch_path, 'rb'))
    else:
        if os.path.exists(easy_batch_path):
            easy_progs_batch = pickle.load(open(easy_batch_path, 'rb'))
        else:
            if args.ncpu > 1:
                with mp.Pool(args.ncpu) as p:
                    easy_progs_batch = p.map(func, tqdm([d_progs[i] for i in easy_prog_inds_batch], 
                                                                desc=f"{prefix} easy progs"))             
            else:
                easy_progs_batch = [func(d_progs[i]) for i in easy_prog_inds_batch]                
            logger.info(f"begin dumping easy programs batch at {easy_batch_path}")
            pickle.dump(easy_progs_batch, open(easy_batch_path, 'wb+'))
            logger.info(f"done dumping easy programs batch at {easy_batch_path}")
        if os.path.exists(hard_batch_path):
            hard_progs_batch = pickle.load(open(hard_batch_path, 'rb'))
        else:
            hard_progs_batch = []
            for i in tqdm(hard_prog_inds_batch):
                hard_path_i = os.path.join(args.cache_dir, f"{prefix}_hard_{i}.pkl")            
                if os.path.exists(hard_path_i):
                    logger.info(f"hard program {hard_path_i} exists")
                    p = pickle.load(open(hard_path_i, 'rb'))
                else:
                    p = d_progs[i]
                    logger.info(f"{prefix} hard program {hard_path_i}")
                    p = func(p)    
                    if args.cache_dir:
                        logger.info(f"begin cache-dumping hard program at {hard_path_i}")  
                        pickle.dump(p, open(hard_path_i, 'wb'))
                        logger.info(f"done cache-dumping hard program at {hard_path_i}")             
                hard_progs_batch.append(p)                
            logger.info(f"begin dumping hard batch programs at {hard_batch_path}")
            pickle.dump(hard_progs_batch, open(hard_batch_path, 'wb+'))
            logger.info(f"done dumping hard batch programs at {hard_batch_path}")
        progs_batch = []
        for ind in inds:
            if ind in easy_prog_inds_batch:
                p = easy_progs_batch[easy_prog_inds_batch.index(ind)]
            else:
                p = hard_progs_batch[hard_prog_inds_batch.index(ind)]
            progs_batch.append(p)
        logger.info(f"begin dumping batch programs at {batch_path}")
        pickle.dump(progs_batch, open(batch_path, 'wb+'))
        logger.info(f"done dumping batch programs at {batch_path}")    
    return progs_batch


def run(args, all_progs, d):
    """
    Strategy: use mp to run easy programs in parallel
    Run hard programs sequentially, use mp among the input combinations
    """
    cache_fpath = os.path.join(args.cache_dir, f"{d}.pkl")
    logger = logging.getLogger('global_logger')   
    progs = [None for _ in all_progs[d]]    
    easy_prog_inds, hard_prog_inds = strategy(all_progs[d], bbf.rxns)
    logger.info(f"parallel run easy programs {easy_prog_inds}")          
    # Batch using run_batch_size
    num_batches = (len(all_progs[d])+args.run_batch_size-1)//args.run_batch_size
    logger.info(f"prepared {num_batches} batches to run")
    prefix = f"{d}_run"
    if args.step is None or (args.step == 'run' and args.batch == -1):
        batch_iter = range(num_batches)
    else:        
        assert args.step == 'run'
        batch_iter = [args.batch] 
    all_inds = range(len(all_progs[d]))
    for j in batch_iter:
        inds = all_inds[j*args.run_batch_size:(j+1)*args.run_batch_size]
        progs_batch = run_or_init_batch(f"{prefix}_{j}", all_progs[d], inds, easy_prog_inds, hard_prog_inds)
        for i, p in zip(inds, progs_batch):
            progs[i] = p 
    if args.step is None or (args.step == 'run' and args.batch == -1):
        # Filter after reaction is run          
        all_progs[d] = filter_programs(progs) 
        if args.cache_dir:   
            logger.info(get_descr(all_progs))  
            logger.info(f"begin cache-dumping all programs at {cache_fpath}")  
            pickle.dump(all_progs, open(cache_fpath, 'wb'))
            logger.info(f"done cache-dumping all programs at {cache_fpath}")           
            logger.info(f"done! {len(all_progs[d])} size-{d} programs")
            # Eliminate unnecessary cache and save
            clean_cache(args, all_progs)


def create_run_programs(args, bbf, size=3):          
    logger = create_logger('global_logger', args.log_file)
    logger.info('args:{}'.format(pprint.pformat(args)))
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    all_progs = {}
    for d in range(1, size+1):      
        if args.step is not None and d > args.d:
            break
        cache_fpath = os.path.join(args.cache_dir, f"{d}.pkl")
        exist = os.path.exists(cache_fpath)
        if args.step is not None:
            if d < args.d:
                assert exist
            elif d == args.d:
                assert not exist
        if args.cache_dir and exist:
            all_progs = load_and_filter(args, all_progs, d)
            continue
        if args.step is None or args.step == 'migrate':
            if args.step == 'migrate':
                assert d == args.keep_prods+1
            if d == args.keep_prods+1:
                migrate(args, all_progs)
        if args.step is None or args.step == 'expand':
            expand(args, all_progs, d)
        if args.step is None or args.step == 'init':
            if args.step == 'init':
                all_progs = pickle.load(open(os.path.join(args.cache_dir, f"{d}_pre.pkl"), "rb"))
            init_and_filter(args, all_progs, d)        
        if args.step is None or args.step == 'run':            
            if args.step == 'run':
                all_progs = pickle.load(open(os.path.join(args.cache_dir, f"{d}_init.pkl"), "rb"))
            logger.info(f"running {len(all_progs[d])} size-{d} programs")
            run(args, all_progs, d)
    return all_progs


def hash_program(prog, output_dir, make_dir=True):
    """
    We perform bfs search on all possible unmasks satisfying topological order.
    Each unmask can expand a reaction on the frontier.
    Each unmask is an int (binary mask over the nodes 0...len(prog.rxn_tree)-1)
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
        if make_dir:
            os.makedirs(cur_dirname, exist_ok=True)
        cur_fpath = cur_dirname.parent / f"{cur_dirname.name}.json"        
        # if cur_dirname.name == "7848891a8df1e8a0c1e0d95248466e01":
        #     breakpoint()
        mask = np.array(list(map(int, cur)))                
        frontier = [f for f in edges[mask[edges[:, 0]] == 1][:, 1] if not mask[f]]        
        if os.path.exists(cur_fpath):   
            data = ProductMap.json_load(open(cur_fpath, 'r'))          
            assert 'rxn_ids' in data
        else:            
            data = {'rxn_ids': {},
                    'mask': mask.tolist(), # debug
                    'tree': prog.hash(mask, return_json=True) # debug
                    }
            
            bb_poss = {}
            for e in prog.entries:               
                if mask[(e[0] if isinstance(e, tuple) else e)]:
                    bb_poss[e] = []
                    if isinstance(e, tuple):
                        bb_poss[e].append(prog.rxn_map[e[0]].available_reactants[e[1]])
                    else:
                        for avail_reactants in prog.rxn_map[e].available_reactants:
                            bb_poss[e].append(avail_reactants)
            if bb_poss:
                data['bbs'] = bb_poss                    
            
        # make/append to json file the continuation
        for f in frontier:
            f = f.item()
            data['rxn_ids'][f] = data['rxn_ids'].get(f, []) + [tree.nodes[f]['rxn_id']]
   
        ProductMap.json_dump(data, open(cur_fpath, 'w+'))
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
        for p in tqdm(all_progs[d], desc=f"enumerating size-{d} programs"):
            hash_program(p, output_dir, make_dir=True)

            



if __name__ == "__main__":
    # Parse input args
    args = get_args()    
    if args.cache_dir and args.cache_dir != PRODUCT_DIR:
        breakpoint()
    if args.mp_min_combinations != MP_MIN_COMBINATIONS:
        breakpoint()        
    if args.num_threads != NUM_THREADS:
        breakpoint()                
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)    
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)
    if os.path.exists(args.skeleton_file):
        # Use to filter building blocks
        skeletons = pickle.load(open(args.skeleton_file, 'rb'))            
        if args.top_bb: 
            bb_counts = count_bbs(args, skeletons, vis=False)
            if args.top_bb != -1:
                for bblock in bblocks:
                    bb_counts[bblock]
            bblocks = sorted(bb_counts.keys(), key=lambda x:-bb_counts[x])                       
            if args.top_bb != -1:                
                bblocks = bblocks[:args.top_bb]
                bb_path = os.path.join(args.visualize_dir, f'bblocks-top-{args.top_bb}.txt')
                with open(bb_path, 'w+') as f:
                    for bb in bblocks:
                        f.write(f"{bb}\n")
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


    if os.path.exists(os.path.join(args.cache_dir, "bbf.pkl")):
        bbf = pickle.load(open(os.path.join(args.cache_dir, "bbf.pkl"), 'rb'))
    else:
        bbf = BuildingBlockFilter(
            building_blocks=bblocks,
            rxn_templates=rxn_templates,
            verbose=args.verbose,
            processes=100,
        )    

        # Count number of unique (uni-reaction, building block) pairs
        bbf._init_rxns_with_reactants()
        bbf.filter()        
        pickle.dump(bbf, open(os.path.join(args.cache_dir, "bbf.pkl"), 'wb+'))

    # Run programs      
    # progs = get_programs(bbf.rxns, size=2)
    all_progs = create_run_programs(args, bbf, size=args.depth)
    if args.step is None:
        if args.visualize_dir:
            os.makedirs(args.visualize_dir, exist_ok=True)           
        if args.stats:        
            for stat in args.stats:
                fpath = os.path.join(args.visualize_dir, f"{stat}.png")            
                fig = plt.Figure()            
                if stat == 'program-count':
                    ax = fig.add_subplot(1,1,1)
                    lengths = [len(all_progs[d]) for d in all_progs]
                    ax.plot(list(all_progs), lengths)
                    ax.set_xlabel("depth")
                    ax.set_ylabel('number of programs')
                elif stat == 'input-length':
                    ax = fig.add_subplot(1,1,1)
                    input_lengths = [Program.avg_input_length(all_progs[d]) for d in all_progs]
                    ax.plot(input_lengths)
                    ax.set_xlabel("depth")
                    ax.set_ylabel('number of building block input sets')
                elif stat == 'input-lengths':
                    input_lengths = [Program.input_length(p) for p in all_progs[args.depth]]
                    if args.depth == 2:
                        ax = fig.add_subplot(1,2,1)
                        ax2 = fig.add_subplot(1,2,2)                
                        ax2.hist(input_lengths, bins=100)
                        ax2.set_title("depth 2")
                        ax2.set_xlabel('#inputs')                            

                        input_lengths = [Program.input_length(p) for p in all_progs[1]]                    
                        ax.hist(input_lengths)
                        ax.set_title("depth 1")
                        ax.set_xscale('log')
                        ax.set_xlabel('#inputs')  
                    else:
                        ax = fig.add_subplot(1,1,1)
                        ax.hist(input_lengths, bins=100)
                        ax.set_xscale('log')
                        ax.set_xlabel('number of building block input sets')                            
                                        
                fig.savefig(fpath) 
                print(fpath)   
        
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
