from synnet.data_generation.preprocessing import BuildingBlockFileHandler, ReactionTemplateFileHandler
from synnet.visualize.drawers import MolDrawer, RxnDrawer
from synnet.visualize.writers import SynTreeWriter, SkeletonPrefixWriter
from synnet.visualize.visualizer import SkeletonVisualizer
from synnet.encoding.distances import cosine_distance
from synnet.models.common import load_gnn_from_ckpt, find_best_model_ckpt, load_mlp_from_ckpt
from synnet.models.gnn import PtrDataset
from synnet.models.mlp import nn_search_list
from synnet.data_generation.syntrees import (
    IdentityIntEncoder,
    MorganFingerprintEncoder,
    SynTreeFeaturizer,
)
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.predict_utils import mol_fp, tanimoto_similarity
from synnet.utils.analysis_utils import serialize_string
from synnet.policy import RxnPolicy
import rdkit.Chem as Chem
from synnet.config import DATA_PREPROCESS_DIR, DATA_RESULT_DIR, MAX_PROCESSES, MAX_DEPTH, NUM_POSS, DELIM
from synnet.utils.data_utils import ReactionSet, SyntheticTreeSet, Skeleton, SkeletonSet, Program
from zss import simple_distance
from pathlib import Path
import numpy as np
import networkx as nx
from typing import Tuple
from torch_geometric.data import Data
import torch
import os
import yaml
import json
import gzip
from copy import deepcopy
import fcntl
import random
from tqdm import tqdm

def lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        return False
    return True


def get_metrics(targets, all_sks):
    assert len(targets) == len(all_sks)
    total_incorrect = {} 
    if args.forcing_eval:
        correct_summary = {}
    else:
        total_correct = {}                     
        
    for (sk_true, smi), sks in zip(targets, all_sks):
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
                correct_steps = test_correct(sk, sk_true, rxns, method='preorder', forcing=True)
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
                match = test_correct(sk, sk_true, rxns, method=args.test_correct_method)
                if args.test_correct_method == 'postorder':
                    correct = match
                elif args.test_correct_method == 'preorder':
                    correct, incorrect = match                        
                    if not correct:
                        update(total_incorrect[tree_id], incorrect)                    
                else:
                    assert args.test_correct_method == 'reconstruct'
                    correct = max(correct, match)
                if correct == 1:
                    break
            total_correct[tree_id]['correct'] += correct
            total_correct[tree_id]['total'] += 1
            total_correct[tree_id]['all'] += [correct]
            # print(f"tree: {sk.tree.edges} total_incorrect: {total_incorrect}")
            summary = {k: v['correct']/v['total'] for (k, v) in total_correct.items()}
            if args.test_correct_method == 'preorder':
                print(f"total summary: {summary}, total incorrect: {total_incorrect}")     
            else:
                print(f"total summary: {summary}")
    if args.forcing_eval:
        return correct_summary
    else:
        return total_correct, total_incorrect



def decode(sk, smi): 
    sk.clear_tree(forcing=args.forcing_eval)
    sk.modify_tree(sk.tree_root, smiles=smi)              
    if args.mermaid:
        skviz = lambda sk: SkeletonVisualizer(skeleton=sk, outfolder=args.out_dir).with_drawings(mol_drawer=MolDrawer, rxn_drawer=RxnDrawer)                       
    else:
        skviz = None
    if 'bblock_inds' in globals():
        bblock_inds = globals()['bblock_inds']
    else:
        bblock_inds = None
    # print(f"begin decoding {smi}")
    if 'rxn_models' in globals():
        rxn_gnn = rxn_models[sk.index]
    else:
        rxn_gnn = globals()['rxn_gnn']

    if 'bb_models' in globals():
        bb_gnn = bb_models[sk.index]
    else:
        bb_gnn = globals()['bb_gnn']
    sks = wrapper_decoder(args, sk, rxn_gnn, bb_gnn, bb_emb, rxn_templates, bblocks, skviz=skviz, bblock_inds=bblock_inds)
    ans = serialize_string(sk.tree, sk.tree_root)        
    return sks


def build_mc(): # build a markov chain    
    tree_lookup = globals()['skeleton_index_lookup']
    trees = list(tree_lookup)
    dists = np.zeros((len(trees), len(trees)))
    for i in range(dists.shape[0]):
        for j in range(i+1, dists.shape[0]):
            _, tree_1, _ = tree_lookup[trees[i]]
            _, tree_2, _ = tree_lookup[trees[j]]   
            dists[i][j] = simple_distance(tree_1, tree_2)
            dists[j][i] = simple_distance(tree_2, tree_1)
        dists[i][i] = float("inf")
    adj = (dists == dists.min(axis=-1, keepdims=True)) # whether smallest possible change
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(8, 6))
    # cax = ax.imshow(adj, cmap='hot', interpolation='nearest')
    # fig.colorbar(cax)
    # ax.set_title('Heatmap Example')
    # fig.savefig('/home/msun415/heatmap.png')  # Save to file
    adj = adj.astype(int) / adj.sum(axis=-1, keepdims=True)
    return adj



def mcmc(sk, smi, beta=1., T=10):
    adj = build_mc()
    rec = [reconstruct(sk, smi) for sk in decode(sk, smi)]
    ind = np.argmax([r[0] for r in rec])    
    res = [(rec[ind][0], rec[ind][1], sk.index)]
    for _ in range(1, T+1):
        tree_key = serialize_string(sk.tree, sk.tree_root)
        index = list(globals()['skeleton_index_lookup']).index(tree_key)        
        nei = np.random.choice(np.arange(adj.shape[1]), p=adj[index])
        tree_key_nei = list(globals()['skeleton_index_lookup'])[nei]
        j_xy = adj[index, nei]
        j_yx = adj[nei, index]        
        sk_y = globals()['skeleton_index_lookup'][tree_key_nei][2]        
        sks_x = decode(sk, smi)
        sks_y = decode(sk_y, smi)
        x_rec = [reconstruct(sk, smi) for sk in sks_x]
        y_rec = [reconstruct(sk, smi) for sk in sks_y]
        ind_x = np.argmax([x[0] for x in x_rec])
        ind_y = np.argmax([y[0] for y in y_rec])
        x_smi = x_rec[ind_x][1]
        y_smi = y_rec[ind_y][1]
        sim_x = x_rec[ind_x][0]
        sim_y = y_rec[ind_y][0]
        pi_x = np.exp(-beta*(1-sim_x))
        pi_y = np.exp(-beta*(1-sim_y))
        a_xy = min(1, pi_y*j_yx/(pi_x*j_xy))
        if random.random() < a_xy:
            sk = sk_y
            res.append((sim_y, y_smi, sk_y.index))
        else:
            res.append((sim_x, x_smi, sk.index))
    return res





def format_metrics(metrics, cum=False):
    res = ""
    for k, v in metrics.items():
        res += k + '\n'
        res += json.dumps(v) + '\n'
        res += '\n'

    if cum:
        cum = []
        for k in metrics:
            cum += metrics[k]['all']
        score = np.mean(cum)
        num = len(cum)
        res += f"Total: {score}/{num}\n"
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



def test_skeletons(args, skeleton_set, max_rxns=0):
    if max_rxns == 0:
        if args.ckpt_dir:
            SKELETON_INDEX = []
            for ind in range(len(skeleton_set.sks)):
                sk = skeleton_set.sks[ind]      
                if 'rxn_models' in globals() and 'bb_models' in globals():
                    if ind in globals()['rxn_models'] and ind in globals()['bb_models']:
                        SKELETON_INDEX.append(ind)
                else:
                    SKELETON_INDEX.append(ind)
        else:        
            dirname = os.path.dirname(args.ckpt_rxn)
            config_file = os.path.join(dirname, 'hparams.yaml')
            config = yaml.safe_load(open(config_file))
            SKELETON_INDEX = list(map(int, config['datasets'].split(',')))
    elif max_rxns == -1:
        SKELETON_INDEX = list(range(len(skeleton_set.skeletons)))
    else:
        SKELETON_INDEX = []
        for index, sk in enumerate(skeleton_set.skeletons):
            sk = Skeleton(sk, index)
            if sk.rxns.sum() <= max_rxns:
                SKELETON_INDEX.append(index)

    globals()['skeleton_index_lookup'] = {}
    globals()['skeleton_index_lookup_by_num_rxns'] = {}
    sks = list(skeleton_set.skeletons)
    for index in SKELETON_INDEX:
        sk = Skeleton(sks[index], index)
        num_rxns = sk.rxns.sum()
        tree_key = serialize_string(sk.tree, sk.tree_root)
        globals()['skeleton_index_lookup'][tree_key] = (index, sk.zss_tree, sk)
        if num_rxns not in globals()['skeleton_index_lookup_by_num_rxns']:
            globals()['skeleton_index_lookup_by_num_rxns'][num_rxns] = {}
        globals()['skeleton_index_lookup_by_num_rxns'][num_rxns][tree_key] = (index, sk.zss_tree, sk)

    
    if hasattr(args, 'strategy') and args.strategy == 'topological':
        globals()['all_topological_sorts'] = {}
        for index in SKELETON_INDEX:
            sk = Skeleton(sks[index], index)
            if sk.rxns.sum() > args.max_num_rxns: # ignore, won't use to decode
                continue
            top_sorts = nx.all_topological_sorts(sk.tree)
            top_sort_set = set()
            for top_sort in top_sorts:
                top_sort = [n for n in top_sort if sk.rxns[n] or sk.leaves[n]]
                top_sort_set.add(tuple(top_sort))    
            tree_key = serialize_string(sk.tree, sk.tree_root)
            globals()['all_topological_sorts'][tree_key] = list(top_sort_set)

    return SKELETON_INDEX


def lookup_skeleton_key(zss_tree, tree_key):    
    if tree_key in globals()['skeleton_index_lookup']:
        return globals()['skeleton_index_lookup'][tree_key][0]
    else: # return skeleton with nearest tree edit distance
        min_dist = float("inf")
        index = -1
        for index, cand_zss_tree in globals()['skeleton_index_lookup'].values():
            dist = simple_distance(zss_tree, cand_zss_tree)
            if dist < min_dist:
                ans = index
        return ans
            



def set_models(args, logger=None):
    if logger is not None:
        logger.info("Start loading models from checkpoints...")  
    if args.ckpt_dir and os.path.isdir(args.ckpt_dir):
        constraint = {'valid_loss': 'accuracy_loss'}
        rxn_models = load_from_dir(args.ckpt_dir, constraint)
        globals()['rxn_models'] = rxn_models
        constraint = {'valid_loss': 'nn_accuracy_loss'}
        bb_models = load_from_dir(args.ckpt_dir, constraint)
        globals()['bb_models'] = bb_models    
    else:
        if not os.path.isfile(args.ckpt_rxn):
            best_ckpt = find_best_model_ckpt(args.ckpt_rxn, key="val_accuracy_loss")
            setattr(args, "ckpt_rxn", best_ckpt)
        rxn_gnn = load_gnn_from_ckpt(Path(args.ckpt_rxn))
        globals()['rxn_gnn'] = rxn_gnn
        if not os.path.isfile(args.ckpt_bb):
            best_ckpt = find_best_model_ckpt(args.ckpt_bb, key="val_nn_accuracy_loss")
            setattr(args, "ckpt_bb", best_ckpt)        
        bb_gnn = load_gnn_from_ckpt(Path(args.ckpt_bb))
        globals()['bb_gnn'] = bb_gnn    
    if hasattr(args, 'ckpt_recognizer') and args.ckpt_recognizer:
        if os.path.isfile(args.ckpt_recognizer):
            recognizer = load_mlp_from_ckpt(args.ckpt_recognizer)
            config_path = os.path.join(Path(args.ckpt_recognizer).parent, 'config.json')
        else:
            recognizer_ckpt = find_best_model_ckpt(args.ckpt_recognizer)
            recognizer = load_mlp_from_ckpt(recognizer_ckpt)
            config_path = os.path.join(Path(args.ckpt_recognizer), 'config.json')
        globals()['recognizer'] = recognizer
        globals()['encoder'] = MorganFingerprintEncoder(2, 2048)
        
        config = json.load(open(config_path)        )
        globals()['skeleton_classes'] = config['datasets']
    if logger is not None:
        logger.info("...loading models completed.")    




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
    else:
        assert args.hash_dir
    rxn_imposs = deepcopy(mask_imposs)    
    attr_name = 'rxn_id_forcing' if args.forcing_eval else 'rxn_id'
    # if max_depth < MAX_DEPTH or rxn_graph.nodes[cur]['depth'] == MAX_DEPTH:
    #     # try every reaction, and use existence of hash to filter possibilites            
    #     term = rxn_graph.nodes[cur][attr_name]
    #     for rxn_id in range(NUM_POSS):
    #         rxn_graph.nodes[cur][attr_name] = rxn_id
    #         p = Program(rxn_graph)
    #         path = os.path.join(args.hash_dir, p.get_path())
    #         mask_imposs[rxn_id] = mask_imposs[rxn_id] or not os.path.exists(path)
    #         if os.path.exists(path):
    #             paths.append(path)
    #         else:
    #             paths.append('')
    #     rxn_graph.nodes[cur][attr_name] = term
    # elif rxn_graph.nodes[cur]['depth'] < MAX_DEPTH:
    #     lca = get_anc(cur, rxn_graph)
    #     # use prog_graph to hash, navigate the file system
    #     term = rxn_graph.nodes[cur][attr_name]
    #     for rxn_id in range(NUM_POSS):
    #         rxn_graph.nodes[cur][attr_name] = rxn_id
    #         p = Program(rxn_graph)
    #         if 'path' not in rxn_graph.nodes[lca]:
    #             breakpoint()
    #         if rxn_graph.nodes[lca]['path'][-5:] == '.json': # prev lca exist              
    #             path_stem = rxn_graph.nodes[lca]['path'][:-5]                
    #             path_stem = Path(path_stem).stem
    #             path = os.path.join(args.hash_dir, path_stem, p.get_path())
    #             mask_imposs[rxn_id] = mask_imposs[rxn_id] or not os.path.exists(path)
    #             if os.path.exists(path):
    #                 paths.append(path)
    #             else:
    #                 paths.append('')                        
    #         else:
    #             mask_imposs[rxn_id] = True
    #             paths.append('')

    #     rxn_graph.nodes[cur][attr_name] = term     
    # if sum(mask_imposs) == NUM_POSS:
    #     mask_imposs = rxn_imposs    
    if rxn_graph.nodes[cur]['depth'] == 1:    
        base_case = False                    
        r_preds = list(rxn_graph.predecessors(cur))
        if len(r_preds) == 0:
            base_case = True
        else:
            r_pred = r_preds[0]
            pred = sk.pred(sk.pred(sk.pred(n)))
            depth = rxn_graph.nodes[r_pred]['depth']
            if depth > 2:
                base_case = True
        if base_case:
            paths = []
            for i in range(91):
                g = nx.DiGraph()
                g.add_node(0, rxn_id=i, depth=1)   
                path = Program(g).get_path()
                path = os.path.join(args.hash_dir, path)
                if os.path.exists(path):
                    paths.append(path)
                    rxn_imposs[i] = False
                else:
                    paths.append('')
                    rxn_imposs[i] = True
            return rxn_imposs, paths
        policy = RxnPolicy(4096+2*91, 2, 2*91, sk.subtree(pred), args.hash_dir, rxns)
        obs = np.zeros((4096+2*91,))        
        rxn_id = rxn_graph.nodes[r_pred]['rxn_id']
        obs[-91+rxn_id] = 1        
        mask, paths = policy.action_mask(obs, return_paths=True)
        mask_imposs = ~mask[:91]
    elif rxn_graph.nodes[cur]['depth'] == 2:
        pred = sk.pred(n)
        policy = RxnPolicy(4096+2*91, 2, 2*91, sk.subtree(pred), args.hash_dir, rxns)
        obs = np.zeros((4096+2*91,))        
        mask, paths = policy.action_mask(obs, return_paths=True)
        mask_imposs = ~mask[91:]
    else:
        mask_imposs = rxn_imposs
        paths = []
    return mask_imposs, paths


def fill_in(args, sk, n, logits_n, bb_emb, rxn_templates, bbs, top_bb=1, top_rxn=1, bblock_inds=None):
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
        # assert sum(mask_imposs) < NUM_POSS # TODO: handle failure
        logits_n[-NUM_POSS:][mask_imposs] = float("-inf")                  
        rxn_id = logits_n[-NUM_POSS:].argsort(axis=-1)[-top_rxn].item()     
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
                if 'path' in sk.tree.nodes[pred]:
                    path = sk.tree.nodes[pred]['path']     
                    exist = os.path.exists(path)
                else:
                    exist = False
        else:
            exist = False
        failed = False
        if exist:                
            e = str(node_map[pred])
            if rxn_graph.nodes[int(e)]['depth'] == 2:
                e = '1'
            else:
                assert rxn_graph.nodes[int(e)]['depth'] == 1
                e = '0'
            data = json.load(open(path))
            succs = list(sk.tree.successors(pred))
            second = sk.tree.nodes[n]['child'] == 'right'
            if e in data['bbs']:
                bbs_child = data['bbs'][e][int(second)]
            else:
                bbs_child = data['bbs'][f"{e}{DELIM}{int(second)}"]
                assert len(bbs_child) == 1
                bbs_child = bbs_child[0]
            if args.forcing_eval:
                if sk.tree.nodes[n]['smiles'] not in bbs_child:
                    bad = False
                    for m in sk.tree:
                        if 'rxn_id' in sk.tree.nodes[m]:
                            if sk.tree.nodes[m]['rxn_id_forcing'] != sk.tree.nodes[m]['rxn_id']:
                                bad = True
                    # if not bad:
                    #     breakpoint()
            indices = [bbs.index(smi) for smi in bbs_child]
            if len(indices) >= top_bb:
                bb_ind = nn_search_list(emb_bb, bb_emb[indices], top_k=top_bb).item()
                smiles = bbs[indices[bb_ind]]
            else:
                failed = True
        if not exist or failed:
            if bblock_inds is not None:
                indices = bblock_inds
            else:
                indices = list(range(bb_emb.shape[0]))
            pred_rxn_id = sk.tree.nodes[pred]['rxn_id']            
            if hasattr(rxns[pred_rxn_id], 'bblock_mask'):
                second = sk.tree.nodes[n]['child'] == 'right'
                indices = rxns[pred_rxn_id].bblock_mask[second]
            bb_ind = nn_search_list(emb_bb, bb_emb[indices], top_k=top_bb).item()
            smiles = bbs[indices[bb_ind]]
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
        return [best_rxn_n]
    else:
        dists = [dist(logits[n][:-NUM_POSS], bb_emb) for n in logits]
        n = list(logits)[np.argmin(dists)]
        return [n]
        # return [n for n in logits if not sk.rxns[n]]


@torch.no_grad()
def wrapper_decoder(args, sk, model_rxn, model_bb, bb_emb, rxn_templates, bblocks, bblock_inds=None, skviz=None):
    top_k_bb = args.top_k
    top_k_rxn = args.top_k_rxn
    """Generate a filled-in skeleton given the input which is only filled with the target."""
    model_rxn.eval()
    model_bb.eval()
    # Following how SynNet reports reconstruction accuracy, we decode top-3 reactants, 
    # corresponding to the first bb chosen
    # To make the code more general, we implement this with a stack
    if args.strategy == 'topological':
        sks = []        
        tree_key = serialize_string(sk.tree, sk.tree_root)
        for top_sort in globals()['all_topological_sorts'][tree_key]:
            sks.append((deepcopy(sk), list(top_sort)))
    elif args.strategy == 'conf':
        sks = [sk]    
    else:
        raise NotImplementedError    
    final_sks = [] 
    while len(sks):
        sk = sks.pop(-1)
        if isinstance(sk, tuple):
            sk, next_node = sk
        else:
            next_node = None
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
            if skviz is not None and args.attn_weights:
                logits_rxn, rxn_attns = model_rxn(data_rxn, return_attention=True)
                logits_bb, bb_attns= model_bb(data_bb, return_attention=True)
            else:
                logits_rxn = model_rxn(data_rxn)
                logits_bb = model_bb(data_bb)
            logits = {}
            if args.strategy == 'topological':
                frontier_nodes = [next_node[0]]
            else:
                frontier_nodes = [n for n in set(sk.frontier_nodes) if not sk.mask[n]]
            for n in frontier_nodes:
                if sk.rxns[n]:
                    logits[n] = logits_rxn[n]
                else:                
                    assert sk.leaves[n]               
                    logits[n] = logits_bb[n]        
            if args.strategy =='topological':        
                poss_n = [next_node.pop(0)]
            else:
                poss_n = pick_node(sk, logits, bb_emb)
            for n in poss_n:
                logits_n = logits[n].clone()
                sk_n = deepcopy(sk)
                first_bb = sk_n.leaves[n] and (sk_n.leaves)[sk_n.mask == 1].sum() == 0
                first_rxn = sk_n.rxns[n] and (sk_n.rxns)[sk_n.mask == 1].sum() == 0
                if top_k_bb > 1 and first_bb: # first bb                
                    for k in range(1, 1+top_k_bb):
                        sk_copy = deepcopy(sk_n)
                        fill_in(args, sk_copy, n, logits_n, bb_emb, rxn_templates, bblocks, top_bb=k, bblock_inds=bblock_inds)
                        if next_node is None:
                            sks.append(sk_copy)
                        else:
                            sks.append((sk_copy, deepcopy(next_node)))
                elif top_k_rxn > 1 and first_rxn:
                    for k in range(1, 1+top_k_rxn):
                        sk_copy = deepcopy(sk_n)
                        fill_in(args, sk_copy, n, logits_n, bb_emb, rxn_templates, bblocks, top_rxn=k, bblock_inds=bblock_inds)
                        if next_node is None:
                            sks.append(sk_copy)
                        else:
                            sks.append((sk_copy, deepcopy(next_node)))                    
                else:
                    fill_in(args, sk_n, n, logits_n, bb_emb, rxn_templates, bblocks, top_bb=1, bblock_inds=bblock_inds)
                    if next_node is None:
                        sks.append(sk_n)
                    else:
                        sks.append((sk_n, next_node))
                    if skviz is not None:
                        sk_viz_n = skviz(sk_n)
                        mermaid_txt = sk_viz_n.write(node_mask=sk_n.mask)             
                        mask_str = ''.join(map(str,sk_n.mask))
                        outfile = sk_viz_n.path / f"skeleton_{sk_n.index}_{mask_str}.md"  
                        SynTreeWriter(prefixer=SkeletonPrefixWriter()).write(mermaid_txt).to_file(outfile)
                        if args.attn_weights:
                            mask = edge_input[1] == n
                            if sk.rxns[n]:                                
                                attns = torch.stack([rxn_attns[layer][mask] for layer in range(len(rxn_attns))], dim=0).mean(axis=0)
                            else:
                                attns = torch.stack([bb_attns[layer][mask] for layer in range(len(bb_attns))], dim=0).mean(axis=0)
                            attns = attns.mean(axis=-1)
                            fpath = os.path.join(outfile.parent, f"{outfile.stem}.png")
                            sk.visualize(fpath, attn=(edge_input[:, mask], attns))
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

        sk.reconstruct(rxns)
        smis = sk_true.tree.nodes[sk_true.tree_root]['smiles'].split(DELIM)
        smis = [Chem.CanonSmiles(smi) for smi in smis]
        correct = smi2 in smis
    else:
        assert method == 'reconstruct'
        sk.reconstruct(rxns)
        smiles = []
        for n in sk.tree:
            if 'smiles' in sk.tree.nodes[n]:
                if sk.tree.nodes[n]['smiles']:
                    smiles += sk.tree.nodes[n]['smiles'].split(DELIM)
        smi2 = Chem.CanonSmiles(sk_true.tree.nodes[sk_true.tree_root]['smiles'])
        sims = tanimoto_similarity(mol_fp(smi2, 2, 4096), smiles)
        correct = max(sims)
    return correct


def update(dic_total, dic):
    for k in dic:
        if k not in dic_total:
            dic_total[k] = 0
        dic_total[k] += dic[k]


def load_data(args, logger=None):
    # ... reaction templates
    rxns = ReactionSet().load(args.rxns_collection_file).rxns
    if logger is not None:
        logger.info(f"Successfully read {args.rxns_collection_file}.")
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)    

    # # ... building blocks
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    if args.top_bbs_file:
        bblock_inds = [bblocks.index(l.rstrip('\n')) for l in open(args.top_bbs_file).readlines()]
        globals()['bblock_inds'] = bblock_inds
    # # A dict is used as lookup table for 2nd reactant during inference:
    # bblocks_dict = {block: i for i, block in enumerate(bblocks)}
    # logger.info(f"Successfully read {args.building_blocks_file}.")

    # # ... building block embedding
    # bblocks_molembedder = (
    #     MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
    # )
    # bb_emb = bblocks_molembedder.get_embeddings()    
    # bb_emb = torch.as_tensor(bb_emb, dtype=torch.float32)
    bb_emb = torch.FloatTensor(np.load(args.embeddings_knn_file))
    if logger is not None:
        logger.info(f"Successfully read {args.embeddings_knn_file}.")    
        logger.info("...loading data completed.")            
    # remember indices of bblocks
    bb_index_lookup = dict(zip(bblocks, range(len(bblocks))))
    for i, r in tqdm(enumerate(rxns)):
        bblock_mask = []
        for j in range(len(r.available_reactants)):
            mask = [False for _ in bblocks]            
            for k in range(len(r.available_reactants[j])):
                bb_index = bb_index_lookup[r.available_reactants[j][k]]
                mask[bb_index] = True
            bblock_mask.append(np.argwhere(mask).flatten())
        setattr(r, 'bblock_mask', bblock_mask)
    globals()['rxns'] = rxns
    globals()['rxn_templates'] = rxn_templates    
    globals()['bblocks'] = bblocks
    globals()['bb_emb'] = bb_emb
    globals()['args'] = args



def predict_skeleton(smiles, max_num_rxns=-1):
    assert 'recognizer' in globals()
    model = globals()['recognizer']
    model.eval()    
    encoder = globals()['encoder']    
    probs = model(torch.FloatTensor(encoder.encode(smiles)))    
    if max_num_rxns == -1:
        ind = probs.argmax(axis=-1).item()        
    else:
        inds = []
        for d in range(1, max_num_rxns+1):
            for ind, _, _ in globals()['skeleton_index_lookup_by_num_rxns'][d].values():
                inds.append(ind)
        sorted_inds = sorted(inds)
        inds = [globals()['skeleton_classes'].index(ind) for ind in sorted_inds]
        assert probs.shape[0] == 1
        ind = probs[0, inds].argmax(axis=-1).item()
        ind = inds[ind]
    return globals()['skeleton_classes'][ind]


# For reconstruct without true skeleton
def reconstruct(sk, smi): 
    rxns = globals()['rxns']
    sk.reconstruct(rxns)
    smiles = []
    for n in sk.tree:
        if 'smiles' in sk.tree.nodes[n]:
            if sk.tree.nodes[n]['smiles']:
                smiles += sk.tree.nodes[n]['smiles'].split(DELIM)
    if isinstance(smi, np.ndarray):
        sims = tanimoto_similarity(smi, smiles)
    else:
        smi2 = Chem.CanonSmiles(smi)
        sims = tanimoto_similarity(mol_fp(smi2, 2, 4096), smiles)
    correct = max(sims)                    
    best_smi = smiles[np.argmax(sims)]
    return correct, best_smi



# For surrogate within GA
def surrogate(sk, fp, oracle):
    sks = decode(sk, fp)
    ans = 0.
    ans_smi = ''
    if sks is None:
        return 0., ''
    for sk in sks:
        sk.reconstruct(rxns)
        sk.visualize('/home/msun415/test.png')
        smi = sk.tree.nodes[sk.tree_root]['smiles']
        for smi in smi.split(DELIM):
            score = oracle(smi)
            if score > ans:
                ans = score
                ans_smi = smi
    return ans, ans_smi