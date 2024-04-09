"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""  # TODO: Clean up + dont hardcode file paths
import json
import yaml
import logging
import gzip
import multiprocessing as mp
from pathlib import Path
import networkx as nx
from typing import Tuple

import numpy as np
np.random.seed(42)
import pandas as pd
import pdb
from tqdm import tqdm
import os
import pickle
from copy import deepcopy

from synnet.config import DATA_PREPROCESS_DIR, DATA_RESULT_DIR, MAX_PROCESSES, MAX_DEPTH, NUM_POSS, DELIM
from synnet.data_generation.preprocessing import BuildingBlockFileHandler, ReactionTemplateFileHandler
from synnet.visualize.drawers import MolDrawer, RxnDrawer
from synnet.visualize.writers import SynTreeWriter, SkeletonPrefixWriter
from synnet.visualize.visualizer import SkeletonVisualizer
from synnet.encoding.distances import cosine_distance
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt, load_gnn_from_ckpt
from synnet.models.gnn import PtrDataset
from synnet.models.mlp import nn_search_list
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import Reaction, ReactionSet, SyntheticTreeSet, Skeleton, SkeletonSet, Program
from synnet.utils.predict_utils import mol_fp, synthetic_tree_decoder_greedy_search
import torch
import rdkit.Chem as Chem
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


def get_anc(cur, rxn_graph):
    lca = cur
    while rxn_graph.nodes[lca]['depth'] != MAX_DEPTH:
        ancs = list(rxn_graph.predecessors(lca))
        if ancs: 
            lca = ancs[0]
        else: 
            break         
    return lca     


def filter_imposs(args, rxn_graph, sk, cur, n):
    max_depth = max([rxn_graph.nodes[n]['depth'] for n in rxn_graph])
    paths = []
    mask_imposs = [False for _ in range(NUM_POSS)]            
    # use sk to filter out reaction type
    bi_mol = len(list(sk.tree.successors(n))) == 2
    for i in range(NUM_POSS):
        if bi_mol != (rxns[i].num_reactant == 2):
            mask_imposs[i] = True         
    if 'rxn' not in args.filter_only:
        return mask_imposs, None
    rxn_imposs = deepcopy(mask_imposs)    
    attr_name = 'rxn_id_forcing' if args.forcing_eval else 'rxn_id'
    if max_depth < MAX_DEPTH or rxn_graph.nodes[cur]['depth'] == MAX_DEPTH:
        # try every reaction, and use existence of hash to filter possibilites            
        term = rxn_graph.nodes[cur][attr_name]
        for rxn_id in range(NUM_POSS):
            rxn_graph.nodes[cur][attr_name] = rxn_id
            p = Program(rxn_graph)
            path = os.path.join(args.hash_dir, p.get_path())
            mask_imposs[rxn_id] = mask_imposs[rxn_id] or not os.path.exists(path)
            if os.path.exists(path):
                paths.append(path)
            else:
                paths.append('')
        rxn_graph.nodes[cur][attr_name] = term
    elif rxn_graph.nodes[cur]['depth'] < MAX_DEPTH:
        lca = get_anc(cur, rxn_graph)
        # use prog_graph to hash, navigate the file system
        term = rxn_graph.nodes[cur][attr_name]
        for rxn_id in range(NUM_POSS):
            rxn_graph.nodes[cur][attr_name] = rxn_id
            p = Program(rxn_graph)
            if 'path' not in rxn_graph.nodes[lca]:
                breakpoint()
            if rxn_graph.nodes[lca]['path'][-5:] == '.json': # prev lca exist              
                path_stem = rxn_graph.nodes[lca]['path'][:-5]                
                path_stem = Path(path_stem).stem
                path = os.path.join(args.hash_dir, path_stem, p.get_path())
                mask_imposs[rxn_id] = mask_imposs[rxn_id] or not os.path.exists(path)
                if os.path.exists(path):
                    paths.append(path)
                else:
                    paths.append('')                        
            else:
                mask_imposs[rxn_id] = True
                paths.append('')

        rxn_graph.nodes[cur][attr_name] = term     
    if sum(mask_imposs) == NUM_POSS:
        mask_imposs = rxn_imposs
    return mask_imposs, paths


def fill_in(args, sk, n, logits_n, bb_emb, rxn_templates, bbs, top_bb=1):
    """
    if rxn
        detect if n is within MAX_DEPTH of root
        if not fill in as usual
        if yes, find LCA (MAX_DEPTH from root), then use the subtree to constrain possibilities
    else
        find LCA (MAX_DEPTH from n), then use it to constrain possibilities
    """            
    rxn_graph, node_map, _ = sk.rxn_graph()       
    if sk.rxns[n]:  
        cur = node_map[n]          
        mask_imposs, paths = filter_imposs(args, rxn_graph, sk, cur, n)
        assert sum(mask_imposs) < NUM_POSS # TODO: handle failure
        logits_n[-NUM_POSS:][mask_imposs] = float("-inf")                  
        rxn_id = logits_n[-NUM_POSS:].argmax(axis=-1).item()     
        # Sanity check for forcing eval
        sk.modify_tree(n, rxn_id=rxn_id, suffix='_forcing' if args.forcing_eval else '')
        sk.tree.nodes[n]['smirks'] = rxn_templates[rxn_id]
        rxn_graph.nodes[cur]['rxn_id'] = rxn_id                   
        # mask the intermediates so they're not considered on frontier
        for succ in sk.tree.successors(n):
            if not sk.leaves[succ]:
                sk.mask = [succ]
        if 'rxn' in args.filter_only:
            if len(paths):
                sk.tree.nodes[n]['path'] = paths[rxn_id]
        # print("path", os.path.join(args.hash_dir, p.get_path()))            
    else:   
        assert sk.leaves[n]
        emb_bb = logits_n[:-NUM_POSS]
        pred = list(sk.tree.predecessors(n))[0]           
        if 'bb' in args.filter_only:            
            if rxn_graph.nodes[node_map[pred]]['depth'] > MAX_DEPTH:
                exist = False
            else:
                path = sk.tree.nodes[pred]['path']     
                exist = os.path.exists(path)
        else:
            exist = False
        failed = False
        if exist:                    
            e = str(node_map[pred])
            data = json.load(open(path))
            succs = list(sk.tree.successors(pred))
            second = len(succs) == 2 and n == succs[1]
            if e in data['bbs']:
                bbs_child = data['bbs'][e][int(second)]
            else:
                bbs_child = data['bbs'][f"{e}{DELIM}{int(second)}"]
                assert len(bbs_child) == 1
                bbs_child = bbs_child[0]
            # if args.forcing_eval:
            #     assert sk.tree.nodes[n]['smiles'] in bbs_child
            indices = [bbs.index(smi) for smi in bbs_child]
            if len(indices) >= top_bb:
                bb_ind = nn_search_list(emb_bb, bb_emb[indices], top_k=top_bb).item()
                smiles = bbs[indices[bb_ind]]
            else:
                failed = True
        if not exist or failed:
            bb_ind = nn_search_list(emb_bb, bb_emb, top_k=top_bb).item()
            smiles = bbs[bb_ind]
        sk.modify_tree(n, smiles=smiles, suffix='_forcing' if args.forcing_eval else '')    



def dist(emb, bb_emb):
    dists = (emb-bb_emb).abs().sum(axis=-1)
    return dists.min()


def pick_node(sk, logits, bb_emb):
    """
    implement strategies here,
    each node returned will add a beam to decoding
    """
    # pick frontier-rxn with highest logit if there is frontier-rxn
    # else pick random bb
    best_conf = float("-inf")
    best_rxn_n = None
    for n in logits:
        if sk.rxns[n]:
            conf = logits[n][-NUM_POSS:].max()
            if conf > best_conf:
                best_conf = conf
                best_rxn_n = n
    if best_rxn_n is not None:
        return [n]
    else:
        dists = [dist(logits[n][:-NUM_POSS], bb_emb) for n in logits]
        n = list(logits)[np.argmin(dists)]
        return [n]
        # return [n for n in logits if not sk.rxns[n]]


@torch.no_grad()
def wrapper_decoder(args, sk, model_rxn, model_bb, bb_emb, rxn_templates, bblocks, skviz=None):
    top_k = args.top_k
    """Generate a filled-in skeleton given the input which is only filled with the target."""
    model_rxn.eval()
    model_bb.eval()
    # Following how SynNet reports reconstruction accuracy, we decode top-3 reactants, 
    # corresponding to the first bb chosen
    # To make the code more general, we implement this with a stack
    sks = [sk]    
    final_sks = []
    while len(sks):
        sk = sks.pop(-1)
        if ((~sk.mask) & (sk.rxns | sk.leaves)).any():
            """
            while there's reaction nodes or leaf building blocks to fill in
                compute, for each vacant reaction node or vacant building block, the possible
                predict on all of these
                pick the highest confidence one
                fill it in
            """        
            # print(f"decode step {sk.mask}")
            # prediction problem        
            _, X, _ = sk.get_state(rxn_target_down_bb=True, rxn_target_down=True)
            for i in range(len(X)):
                if i != sk.tree_root and not sk.rxns[i] and not sk.leaves[i]:
                    X[i] = 0
            
            edges = sk.tree_edges
            tree_edges = np.concatenate((edges, edges[::-1]), axis=-1)
            edge_input = torch.tensor(tree_edges, dtype=torch.int64)        
            if model_rxn.layers[0].in_channels != X.shape[1]:
                pe = PtrDataset.positionalencoding1d(32, len(X))
                x_input_rxn = np.concatenate((X, pe), axis=-1)
            else:
                x_input_rxn = X
            if model_bb.layers[0].in_channels != X.shape[1]:
                pe = PtrDataset.positionalencoding1d(32, len(X))
                x_input_bb = np.concatenate((X, pe), axis=-1)            
            else:
                x_input_bb = X
            data_rxn = Data(edge_index=edge_input, x=torch.Tensor(x_input_rxn))
            data_bb = Data(edge_index=edge_input, x=torch.Tensor(x_input_bb))
            logits_rxn = model_rxn(data_rxn)
            logits_bb = model_bb(data_bb)
            logits = {}
            frontier_nodes = [n for n in set(sk.frontier_nodes) if not sk.mask[n]]
            for n in frontier_nodes:
                if sk.rxns[n]:
                    logits[n] = logits_rxn[n]
                else:                
                    assert sk.leaves[n]               
                    logits[n] = logits_bb[n]        
            poss_n = pick_node(sk, logits, bb_emb)
            for n in poss_n:
                logits_n = logits[n].clone()
                sk_n = deepcopy(sk)
                first_bb = sk_n.leaves[n] and (sk_n.leaves)[sk_n.mask == 1].sum() == 0
                if top_k > 1 and first_bb: # first bb                
                    for k in range(1, 1+top_k):
                        sk_copy = deepcopy(sk_n)
                        fill_in(args, sk_copy, n, logits_n, bb_emb, rxn_templates, bblocks, top_bb=k)
                        sks.append(sk_copy)
                else:
                    fill_in(args, sk_n, n, logits_n, bb_emb, rxn_templates, bblocks, top_bb=1)
                    sks.append(sk_n)
                    if skviz is not None:
                        mermaid_txt = skviz.write(node_mask=sk_n.mask)             
                        mask_str = ''.join(map(str,sk_n.mask))
                        outfile = skviz.path / f"skeleton_{sk_n.index}_{mask_str}.md"  
                        SynTreeWriter(prefixer=SkeletonPrefixWriter()).write(mermaid_txt).to_file(outfile)      
                        print(f"Generated markdown file.", os.path.join(os.getcwd(), outfile))            
        else:
            final_sks.append(sk)
    # print(len(final_sks), "beams")
    return final_sks


def test_correct(sk, sk_true, rxns, method='preorder', forcing=False):
    if method == 'preorder':
        if forcing:
            for n in sk.tree:
                attrs = [attr for attr in list(sk.tree.nodes[n]) if '_forcing' in attr]
                for attr in attrs:
                    # we re-store the attributes containing predictions into 
                    # original attributes
                    sk.tree.nodes[n][attr[:-len('_forcing')]] = sk.tree.nodes[n][attr]
        total_incorrect = {}
        preorder = list(nx.dfs_preorder_nodes(sk.tree, source=sk.tree_root))
        correct = True
        seq_correct = []
        for i in preorder:                
            if sk.rxns[i]:
                if sk.tree.nodes[i]['rxn_id'] != sk_true.tree.nodes[i]['rxn_id']:
                    correct = False
                    total_incorrect[i] = 1                    
                seq_correct.append(i not in total_incorrect)
            elif sk.leaves[i]:
                if sk.tree.nodes[i]['smiles'] != sk_true.tree.nodes[i]['smiles']:
                    correct = False
                    total_incorrect[i] = 1   
                seq_correct.append(i not in total_incorrect)
        if forcing:                     
            return seq_correct
        else:
            return correct, total_incorrect
    elif method == 'postorder':
        # compute intermediates and target
        postorder = list(nx.dfs_postorder_nodes(sk.tree, source=sk.tree_root))
        for i in postorder:
            if sk.rxns[i]:
                succ = list(sk.tree.successors(i))
                if sk.tree.nodes[succ[0]]['child'] == 'right':
                    succ = succ[::-1]
                reactants = tuple(sk.tree.nodes[j]['smiles'] for j in succ)
                if len(reactants) != rxns[sk.tree.nodes[i]['rxn_id']].num_reactant:
                    return False
                interm = Reaction(sk.tree.nodes[i]['smirks']).run_reaction(reactants)              
                pred = list(sk.tree.predecessors(i))[0]
                if interm is None:
                    return False
                sk.tree.nodes[pred]['smiles'] = interm
        smi1 = Chem.CanonSmiles(sk.tree.nodes[sk.tree_root]['smiles'])
        smi2 = Chem.CanonSmiles(sk_true.tree.nodes[sk_true.tree_root]['smiles'])
        correct = smi1 == smi2
    else:
        assert method == 'reconstruct'
        breakpoint()
    return correct


def update(dic_total, dic):
    for k in dic:
        if k not in dic_total:
            dic_total[k] = 0
        dic_total[k] += dic[k]


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

def get_metrics(targets, all_sks):
    assert len(targets) == len(all_sks)
    total_incorrect = {} 
    if args.forcing_eval:
        correct_summary = {}
    else:
        total_correct = {}                     
        
    for smi, sks in zip(targets, all_sks):
        tree_id = str(np.array(sks[0].tree.edges))
        if tree_id not in total_incorrect: 
            total_incorrect[tree_id] = {}
        if not args.forcing_eval and tree_id not in total_correct:
            total_correct[tree_id] = {'correct': 0, 'total': 0, 'all': []}
        if args.forcing_eval and tree_id not in correct_summary:
            correct_summary[tree_id] = {'sum_pool_correct': 0, 'total_pool': 0}            
        if args.forcing_eval:            
            best_correct_steps = []            
            for sk in sks:
                correct_steps = test_correct(sk, lookup[smi], rxns, method='preorder', forcing=True)
                if sum(correct_steps) >= sum(best_correct_steps):
                    best_correct_steps = correct_steps                    
            correct_summary[tree_id]['sum_pool_correct'] += sum(best_correct_steps)
            correct_summary[tree_id]['total_pool'] += len(best_correct_steps)                
            correct_summary[tree_id]['step_by_step'] = correct_summary[tree_id]['sum_pool_correct']/correct_summary[tree_id]['total_pool']                              
            summary = {k: v['step_by_step'] for (k, v) in correct_summary.items()}
            print(f"step-by-step: {summary}")                
        else:
            correct = False
            for sk in sks:
                match = test_correct(sk, lookup[smi], rxns, method=args.test_correct_method)
                if args.test_correct_method == 'postorder':
                    correct = match
                elif args.test_correct_method == 'preorder':
                    correct, incorrect = match                        
                    if not correct:
                        update(total_incorrect[tree_id], incorrect)                    
                else:
                    assert args.test_correct_method == 'reconstruct'
                    sim = match
                if correct:
                    break
            total_correct[tree_id]['correct'] += correct
            total_correct[tree_id]['total'] += 1
            total_correct[tree_id]['all'] += [correct]
            # print(f"tree: {sk.tree.edges} total_incorrect: {total_incorrect}")
            breakpoint()
            summary = {k: v['correct']/v['total'] for (k, v) in total_correct.items()}
            if args.test_correct_method == 'preorder':
                print(f"total summary: {summary}, total incorrect: {total_incorrect}")     
            else:
                print(f"total summary: {summary}")
    if args.forcing_eval:
        return correct_summary
    else:
        return total_correct, total_incorrect



def decode(smi):
    sk = deepcopy(lookup[smi])    
    sk.clear_tree(forcing=args.forcing_eval)
    sk.modify_tree(sk.tree_root, smiles=smi)              
    if args.mermaid:
        skviz = SkeletonVisualizer(skeleton=sk, outfolder=args.out_dir).with_drawings(mol_drawer=MolDrawer, rxn_drawer=RxnDrawer)                       
    else:
        skviz = None
    # print(f"begin decoding {smi}")
    if 'rxn_models' in globals():
        rxn_gnn = rxn_models[sk.index]
    if 'bb_models' in globals():
        bb_gnn = bb_models[sk.index]
    try:
        sks = wrapper_decoder(args, sk, rxn_gnn, bb_gnn, bb_emb, rxn_templates, bblocks, skviz)
        print(f"done decoding {smi}")
        return sks
    except Exception as e:
        pdb.post_mortem()
        logger.error(f"{smi} {e}")


def format_metrics(metrics):
    res = ""
    for k, v in metrics.items():
        res += k + '\n'
        res += json.dumps(v) + '\n'
        res += '\n'
    return res


def load_from_dir(dir, constraint):
    models = {}
    for version in os.listdir(dir):        
        hparams_filepath = os.path.join(dir, version, 'hparams.yaml')        
        hparams_file = yaml.safe_load(open(hparams_filepath))      
        match = True
        for k in constraint:
            if str(constraint[k]) != str(hparams_file[k]):
                match = False
        if match:       
            fpaths = list(Path(os.path.join(dir, version)).glob("*.ckpt"))
            if len(fpaths) != 1:
                print(f"{version} has {len(fpaths)} ckpts")
                continue
            models[int(hparams_file['datasets'])] = load_gnn_from_ckpt(fpaths[0])
    return models
    



def main(args):
    handler = logging.FileHandler(os.path.join(args.out_dir, "log.txt"))
    logger.addHandler(handler)
    logger.info("Start.")
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")
    # ... models
    logger.info("Start loading models from checkpoints...")  
    if os.path.isdir(args.ckpt_dir):
        constraint = {'valid_loss': 'accuracy_loss'}
        rxn_models = load_from_dir(args.ckpt_dir, constraint)
        globals()['rxn_models'] = rxn_models
        constraint = {'valid_loss': 'nn_accuracy_loss'}
        bb_models = load_from_dir(args.ckpt_dir, constraint)
        globals()['bb_models'] = bb_models    
    else:
        rxn_gnn = load_gnn_from_ckpt(Path(args.ckpt_rxn))
        globals()['rxn_gnn'] = rxn_gnn
        bb_gnn = load_gnn_from_ckpt(Path(args.ckpt_bb))
        globals()['bb_gnn'] = bb_gnn
    logger.info("...loading models completed.")    

    # ... reaction templates
    rxns = ReactionSet().load(args.rxns_collection_file).rxns
    logger.info(f"Successfully read {args.rxns_collection_file}.")
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)    

    # building blocks
    if args.top_bbs_file:
        TOP_BBS = set([l.rstrip('\n') for l in open(args.top_bbs_file).readlines()])
    else:
        TOP_BBS = BuildingBlockFileHandler().load(args.building_blocks_file)    

    # Load skeleton set
    sk_set = None
    if args.skeleton_set_file:        
        skeletons = pickle.load(open(args.skeleton_set_file, 'rb'))
        skeleton_set = SkeletonSet().load_skeletons(skeletons)                
        syntree_set_all = [st for v in skeletons.values() for st in v]
        syntree_set = []
        SKELETON_INDEX = []
        for ind in range(len(skeleton_set.sks)):
            sk = skeleton_set.sks[ind]      
            if 'rxn_models' in globals() and 'bb_models' in globals():
                if ind in globals()['rxn_models'] and ind in globals()['bb_models']:
                    SKELETON_INDEX.append(ind)                
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

    # # ... building block embedding
    bblocks_molembedder = (
        MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
    )
    bb_emb = bblocks_molembedder.get_embeddings()
    
    bb_emb = torch.as_tensor(bb_emb, dtype=torch.float32)
    logger.info(f"Successfully read {args.embeddings_knn_file} and initialized BallTree.")
    logger.info("...loading data completed.")

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")

    # Set some global vars for mp
    for var_name in ['args', 'lookup', 'bb_emb', 'rxn_templates', 'bblocks', 'rxns']:
        globals()[var_name] = locals()[var_name]    
    # targets = targets[:10]     
    batch_size = args.batch_size        
    # decode('CC(Nc1ccc(I)cc1-c1nc(-c2cc(F)ccc2N=C=O)n[nH]1)c1ccc(N)c(O)c1')
    # decode('CC(C)(N=C=O)C1=Cc2cc(Br)c(C(=O)O)c(N)c2O1')
    all_sks = []
    all_targets = []
    for batch in range((len(targets)+batch_size-1)//batch_size):
        target_batch = targets[batch_size*batch:batch_size*batch+batch_size]
        if args.ncpu == 1:
            sks_batch = []
            for _, smi in tqdm(enumerate(target_batch)):            
                sks = decode(smi)
                sks_batch.append(sks)              
        else:
            with mp.Pool(args.ncpu) as p:
                sks_batch = p.map(decode, tqdm(target_batch))        
        all_targets += target_batch
        all_sks += sks_batch     
        batch_correct, batch_incorrect = get_metrics(all_targets, all_sks)        
        logger.info(f"batch {batch} correct: {format_metrics(batch_correct)}")
        logger.info(f"batch {batch} incorrect: {format_metrics(batch_incorrect)}")
    
    if args.forcing_eval:
        correct_summary = get_metrics(targets, all_sks)
        logger.info(f"correct summary: {correct_summary}")
    else:
        total_correct, total_incorrect = get_metrics(targets, all_sks)
        logger.info(f"total correct: {format_metrics(total_correct)}")
        logger.info(f"total incorrect: {format_metrics(total_incorrect)}")        
        

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



if __name__ == "__main__":

    # Parse input args
    args = get_args()
    breakpoint()
    main(args)
