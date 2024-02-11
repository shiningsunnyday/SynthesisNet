"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""  # TODO: Clean up + dont hardcode file paths
import json
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
import os
import pickle
from copy import deepcopy

from synnet.config import DATA_PREPROCESS_DIR, DATA_RESULT_DIR, MAX_PROCESSES, MAX_DEPTH, NUM_POSS
from synnet.data_generation.preprocessing import BuildingBlockFileHandler, ReactionTemplateFileHandler
from synnet.visualize.drawers import MolDrawer, RxnDrawer
from synnet.visualize.writers import SynTreeWriter, SkeletonPrefixWriter
from synnet.visualize.visualizer import SkeletonVisualizer
from synnet.encoding.distances import cosine_distance
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt, load_gnn_from_ckpt
from synnet.models.gnn import PtrDataset
from synnet.models.mlp import nn_search_list
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import ReactionSet, SyntheticTreeSet, Skeleton, SkeletonSet, Program
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



def fill_in(args, sk, n, logits, bb_emb, bbs):
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
        mask_imposs = [False for _ in range(NUM_POSS)]            
        paths = []
        if rxn_graph.nodes[cur]['depth'] < MAX_DEPTH:
            lca = cur
            while rxn_graph.nodes[lca]['depth'] != MAX_DEPTH:
                ancs = list(rxn_graph.predecessors(lca))
                if ancs: 
                    lca = ancs[0]
                else: 
                    break              
            prog_graph = Skeleton.ego_graph(rxn_graph, lca, rxn_graph.nodes[lca]['depth'])                                
            # use prog_graph to hash, navigate the file system
            term = rxn_graph.nodes[cur]['rxn_id']
            for rxn_id in range(NUM_POSS):
                rxn_graph.nodes[cur]['rxn_id'] = rxn_id
                p = Program(rxn_graph)
                assert rxn_graph.nodes[lca]['path'][-5:] == '.json'
                path_stem = rxn_graph.nodes[lca]['path'][:-5]
                path_stem = Path(path_stem).stem
                path = os.path.join(args.hash_dir, path_stem, p.get_path())
                mask_imposs[rxn_id] = not os.path.exists(path)
                if os.path.exists(path):
                    paths.append(path)
                else:
                    paths.append('')
            rxn_graph.nodes[cur]['rxn_id'] = term    
        elif rxn_graph.nodes[cur]['depth'] == MAX_DEPTH:
            # try every reaction, and use existence of hash to filter possibilites            
            term = rxn_graph.nodes[cur]['rxn_id']
            for rxn_id in range(NUM_POSS):
                rxn_graph.nodes[cur]['rxn_id'] = rxn_id
                p = Program(rxn_graph)
                path = os.path.join(args.hash_dir, p.get_path())
                mask_imposs[rxn_id] = not os.path.exists(path)
                if os.path.exists(path):
                    paths.append(path)
                else:
                    paths.append('')
            rxn_graph.nodes[cur]['rxn_id'] = term
        if sum(mask_imposs) < NUM_POSS:
            logits[n][-NUM_POSS:][mask_imposs] = float("-inf")
        rxn_id = logits[n][-NUM_POSS:].argmax(axis=-1).item()              
        sk.modify_tree(n, rxn_id=rxn_id)
        rxn_graph.nodes[cur]['rxn_id'] = rxn_id
        sk.tree.nodes[n]['smirks'] = rxn_templates[rxn_id]                
        # mask the intermediates so they're not considered on frontier
        for succ in sk.tree.successors(n):
            if not sk.leaves[succ]:
                sk.mask = [succ]
        sk.tree.nodes[n]['path'] = paths[rxn_id]
        # print("path", os.path.join(args.hash_dir, p.get_path()))            
    else:   
        assert sk.leaves[n]
        emb_bb = logits[n][:-NUM_POSS]
        pred = list(sk.tree.predecessors(n))[0]
        path = sk.tree.nodes[pred]['path']
        exist = os.path.exists(path)
        exist = False
        if exist:
            e = str(node_map[pred])
            data = json.load(open(path))
            succs = list(sk.tree.successors(pred))
            second = len(succs) == 2 and n == succs[1]
            indices = [bbs.index(smi) for smi in data['bbs'][e][second]]
            bb_ind = nn_search_list(emb_bb, bb_emb[indices]).item()                         
            smiles = bbs[indices[bb_ind]]
        else:
            bb_ind = nn_search_list(emb_bb, bb_emb).item()                   
            smiles = bbs[bb_ind]
        sk.modify_tree(n, smiles=smiles)    



def pick_node(sk, logits):
    """
    implement strategies here
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
        return n
    else:
        return list(logits)[0]


@torch.no_grad()
def wrapper_decoder(hash_dir, sk, model_rxn, model_bb, bb_emb, rxn_templates, bblocks, skviz=None):
    """Generate a filled-in skeleton given the input which is only filled with the target."""
    model_rxn.eval()
    model_bb.eval()      
    while ((~sk.mask) & (sk.rxns | sk.leaves)).any():
        """
        while there's reaction nodes or leaf building blocks to fill in
            compute, for each vacant reaction node or vacant building block, the possible
            predict on all of these
            pick the highest confidence one
            fill it in
        """        
        
        # prediction problem        
        _, X, _ = sk.get_state(rxn_target_down_bb=True, rxn_target_down=True)
        for i in range(len(X)):
            if i != sk.tree_root and not sk.rxns[i] and not sk.leaves[i]:
                X[i] = 0
        
        edges = sk.tree_edges
        tree_edges = np.concatenate((edges, edges[::-1]), axis=-1)
        edge_input = torch.tensor(tree_edges, dtype=torch.int64)        
        pe = PtrDataset.positionalencoding1d(32, len(X))
        x_input = np.concatenate((X, pe), axis=-1)        
        data_rxn = Data(edge_index=edge_input, x=torch.Tensor(X))
        data_bb = Data(edge_index=edge_input, x=torch.Tensor(x_input))
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
        n = pick_node(sk, logits)
        fill_in(args, sk, n, logits, bb_emb, bblocks)                
        if skviz is not None:
            mermaid_txt = skviz.write(node_mask=sk.mask)             
            mask_str = ''.join(map(str,sk.mask))
            outfile = skviz.path / f"skeleton_{sk.index}_{mask_str}.md"  
            SynTreeWriter(prefixer=SkeletonPrefixWriter()).write(mermaid_txt).to_file(outfile)      
            print(f"Generated markdown file.", os.path.join(os.getcwd(), outfile))            
        # # get frontier rxns     
        # frontier_rxns = np.array([False for _ in range(len(sk.tree))])
        # for n in np.argwhere((~sk.mask) & sk.rxns).flatten():
        #     if sk.pred(n)==sk.tree_root or sk.mask[sk.pred(sk.pred(n))]:
        #         frontier_rxns[n] = True
        # frontier_rxns = np.bool_(frontier_rxns)
        # # get frontier bbs
        # frontier_bbs = np.array([False for _ in range(len(sk.tree))])
        # for n in np.argwhere((~sk.mask) & (~sk.rxns) & (sk.leaves)).flatten():
        #     if sk.mask[sk.pred(n)]: # parent present
        #         frontier_bbs[n] = True

        # if sk.mask.sum()>1:
        #     hash_val = sk.hash()
        #     fpath = os.path.join(cur_dir, f"{hash_val}.json")
        #     dirpath = os.path.join(cur_dir, f"{hash_val}")
        #     if os.path.exists(fpath):
        #         data = json.load(open(fpath, 'r'))
        #         imposs = np.bool([True for _ in range(NUM_POSS)])
        #         imposs[data['rxn_ids']] = False
        #         cur_dir = dirpath
        #     else:
        #         breakpoint()
        # else:
        #     cur_dir = hash_dir
        #     imposs = []        
        
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
        "--syntree-set-file",
        type=str,
        required=True,
        help="Input file for the ground-truth syntrees to lookup target smiles in",
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
    # Visualization
    parser.add_argument(
        "--mermaid"        , action='store_true'
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

    # ... reaction templates
    rxns = ReactionSet().load(args.rxns_collection_file).rxns
    logger.info(f"Successfully read {args.rxns_collection_file}.")
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)    

    # Load skeleton set
    sk_set = None
    if args.syntree_set_file:
        syntree_set_all = SyntheticTreeSet().load(args.syntree_set_file)                
        skeletons = pickle.load(open(args.skeleton_set_file, 'rb'))
        skeleton_set = SkeletonSet().load_skeletons(skeletons)        
        syntree_set = []
        # SKELETON_INDEX = [ind for ind in range(len(skeleton_set.sks)) if len(list(skeleton_set.skeletons)[ind].reactions)==2]
        SKELETON_INDEX = [7]
        # rep = set()
        for syntree in syntree_set_all:        
            index = skeleton_set.lookup[syntree.root.smiles][0].index    
            if len(skeleton_set.lookup[syntree.root.smiles]) == 1:
                if index in SKELETON_INDEX:
                # if len(syntree.reactions) == 2:
                #     if rxns[syntree.reactions[0].rxn_id].num_reactant == 2:
                #         if rxns[syntree.reactions[1].rxn_id].num_reactant == 1:
                #             breakpoint()
                    syntree_set.append(syntree)
                # if index in [0,1,2,3,4]:
                #     syntree_set.append(syntree)
        print(f"{len(syntree_set)}/{len(syntree_set_all)} syntrees")
        targets = [syntree.root.smiles for syntree in syntree_set]
        lookup = {}
        # Compute the gold skeleton
        all_smiles = dict(zip([st.root.smiles for st in syntree_set], range(len(syntree_set))))
        for i, target in enumerate(targets):
            sk = Skeleton(syntree_set[i], skeleton_set.lookup[target][0].index)
            lookup[target] = sk
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
    # ... models
    logger.info("Start loading models from checkpoints...")  
    rxn_gnn = load_gnn_from_ckpt(Path(args.ckpt_rxn))
    bb_gnn = load_gnn_from_ckpt(Path(args.ckpt_bb))
    logger.info("...loading models completed.")

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")

    if args.ncpu == 1:
        results = []
        total_correct = 0
        total_incorrect = {}
        for no, smi in enumerate(targets):
            sk = deepcopy(lookup[smi])
            tree_id = str(np.array(sk.tree.edges))
            if tree_id not in total_incorrect: 
                total_incorrect[tree_id] = {}
            sk.clear_tree()
            sk.modify_tree(sk.tree_root, smiles=smi)
            if args.mermaid:
                skviz = SkeletonVisualizer(skeleton=sk, outfolder=args.out_dir).with_drawings(mol_drawer=MolDrawer, rxn_drawer=RxnDrawer)                       
            else:
                skviz = None
            sk = wrapper_decoder(args.hash_dir, sk, rxn_gnn, bb_gnn, bb_emb, rxn_templates, bblocks, skviz)
            correct = True
            preorder = list(nx.dfs_preorder_nodes(sk.tree, source=sk.tree_root))
            for i in preorder:                
                if sk.rxns[i]:
                    if sk.tree.nodes[i]['rxn_id'] != lookup[smi].tree.nodes[i]['rxn_id']:
                        correct = False
                        if i not in total_incorrect[tree_id]:
                            total_incorrect[tree_id][i] = 0
                        total_incorrect[tree_id][i] += 1
                        break
                elif sk.leaves[i]:
                    if sk.tree.nodes[i]['smiles'] != lookup[smi].tree.nodes[i]['smiles']:
                        correct = False
                        if i not in total_incorrect[tree_id]:
                            total_incorrect[tree_id][i] = 0
                        total_incorrect[tree_id][i] += 1
                        break
            total_correct += correct
            print(f"tree: {sk.tree.edges} total_incorrect: {total_incorrect}")
            print(f"{total_correct}/{no+1}")            
        

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
