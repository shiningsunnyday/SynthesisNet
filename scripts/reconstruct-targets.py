"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""  # TODO: Clean up + dont hardcode file paths
import json
import logging
import time
import fcntl
import numpy as np
np.random.seed(42)
import pandas as pd
import pdb
from tqdm import tqdm
import os
import pickle
import hashlib
from synnet.utils.reconstruct_utils import *
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import random
random.seed(42)

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
    parser.add_argument("--ckpt-recognizer", type=str, help="Recognizer checkpoint to use")    
    parser.add_argument("--max_rxns", type=int, help="Restrict syntree test set to max number of reactions (-1 to do syntrees, 0 to syntrees whose skleton class was trained on by ckpt_dir)", default=-1)
    parser.add_argument("--max_num_rxns", type=int, help="Restrict skeleton prediction to max number of reactions", default=-1)    
    parser.add_argument("--num-analogs", type=int, help="Number of analogs (by varying skeleton) to generate", default=1)
    parser.add_argument("--num", type=int, help="Number of targets", default=-1)
    parser.add_argument("--ckpt-dir", type=str, help="Model checkpoint dir, if given assume one ckpt per class")
    parser.add_argument(
        "--skeleton-set-file",
        type=str,
        help="Input file for the ground-truth skeletons to lookup target smiles in",
    )                
    parser.add_argument(
        "--hash-dir",
        default=""
    )
    parser.add_argument(
        "--out-dir"        
    )
    # Parameters
    parser.add_argument(
        "--data",
        type=str,        
        help="Path to list of SMILES"
    )
    parser.add_argument("--top-bbs-file", help='if given, consider only these bbs')
    parser.add_argument("--top-k", default=1, type=int, help="Beam width for first bb")
    parser.add_argument("--top-k-rxn", default=1, type=int, help="Beam width for first rxn")
    parser.add_argument("--batch-size", default=10, type=int, help='how often to report metrics')
    parser.add_argument("--filter-only", type=str, nargs='+', choices=['rxn', 'bb'], default=[])
    parser.add_argument("--strategy", default='conf', choices=['conf', 'topological'], help="""
        Strategy to decode:
            Conf: Decode all reactions before bbs. Choose highest-confidence reaction. Choose closest neighbor bb.
            Topological: Decode every topological order of the rxn+bb nodes.
    """)
    parser.add_argument("--forcing-eval", action='store_true')
    parser.add_argument("--test-correct-method", default='preorder', choices=['preorder', 'postorder', 'reconstruct'])
    # Visualization
    parser.add_argument("--mermaid", action='store_true')
    parser.add_argument("--one-per-class", action='store_true', help='visualize one skeleton per class')
    parser.add_argument("--attn_weights", action='store_true', help='visualize attn weights for each step')
    # Processing
    parser.add_argument("--ncpu", type=int, default=1, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    # I/O
    parser.add_argument("--sender-filename")
    parser.add_argument("--receiver-filename")
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

    skeletons = pickle.load(open(args.skeleton_set_file, 'rb'))
    skeleton_set = SkeletonSet().load_skeletons(skeletons)
    syntree_set_all = [st for v in skeletons.values() for st in v]    
    SKELETON_INDEX = test_skeletons(args, skeleton_set, args.max_rxns)        
    # Load data
    if args.data: 
        assert os.path.exists(args.data)
        if 'chembl' in args.data:
            df = pd.read_csv(args.data, sep='\t')
            col_name = 'canonical_smiles'
            if col_name in df:
                targets = list(df[col_name])
            else:
                breakpoint()
        else:
            raise NotImplementedError           
        random.shuffle(targets)     
    else:
        syntree_set = []        
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
        random.shuffle(syntree_set)
        targets = [syntree.root.smiles for syntree in syntree_set]                    
    if args.num != -1:
        targets = targets[:args.num]
    
    # populating lookup takes time, so cache it
    args_dict = {k: str(v) for (k, v) in args.__dict__.items()}
    serialized = json.dumps(args_dict, sort_keys=True).encode('utf-8')
    name = hashlib.md5(serialized).hexdigest()
    path = os.path.join(args.out_dir, f"{name}.pkl")
    if os.path.exists(path):
        lookup = pickle.load(open(path, 'rb'))
    else:
        lookup = {}    
        for i, target in tqdm(enumerate(targets), "initializing skeletons"):
            if args.data:        
                assert args.ckpt_recognizer
                good = True
            else:
                # Use the gold skeleton or predict the skeleton
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
                if target not in lookup:
                    lookup[target] = []
                if args.ckpt_recognizer:                
                    if args.num_analogs == 1:
                        pred_index = predict_skeleton(target, max_num_rxns=args.max_num_rxns)
                        sk = Skeleton(list(skeletons)[pred_index], pred_index)
                        if args.max_num_rxns != -1:
                            assert sk.rxns.sum() <= args.max_num_rxns
                        lookup[target].append(sk)
                    else:                    
                        pred_indices = predict_skeleton(target, max_num_rxns=args.max_num_rxns, top_k=list(range(1, args.num_analogs+1)))
                        for pred_index in pred_indices:
                            sk = Skeleton(list(skeletons)[pred_index], pred_index)
                            if args.max_num_rxns != -1:
                                assert sk.rxns.sum() <= args.max_num_rxns
                            lookup[target].append(sk)
                        
                else:
                    lookup[target] = sk
        pickle.dump(lookup, open(path, 'wb+'))
        
    targets = list(lookup)
    print(f"{len(targets)} targets")
    

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")
 
    batch_size = args.batch_size        
    # decode('CC(Nc1ccc(I)cc1-c1nc(-c2cc(F)ccc2N=C=O)n[nH]1)c1ccc(N)c(O)c1')
    # decode('CC(C)(N=C=O)C1=Cc2cc(Br)c(C(=O)O)c(N)c2O1')
    all_sks = []
    all_targets = []
    for batch in range((len(targets)+batch_size-1)//batch_size):
        target_batch = targets[batch_size*batch:batch_size*batch+batch_size]
        target_batch = [(lookup[smi][j], smi) for smi in target_batch for j in range(len(lookup[smi]))]

        if args.test_correct_method == 'reconstruct' and args.sender_filename and args.receiver_filename: # assume background processes
            open(args.sender_filename, 'w+').close() # clear
            open(args.receiver_filename, 'w+').close() # clear            
            while(True):
                with open(args.sender_filename, 'r') as fr:
                    editable = lock(fr)
                    if editable:
                        with open(args.sender_filename, 'w') as fw:
                            for sk, smi in target_batch:
                                sample = DELIM.join([smi, str(sk.index)])
                                fw.write('{}\n'.format(sample))
                        break
                    fcntl.flock(fr, fcntl.LOCK_UN)     
            num_samples = len(target_batch)
            print("Waiting for reconstruct evaluation...")       
            while(True):
                with open(args.receiver_filename, 'r') as fr:
                    editable = lock(fr)
                    if editable:                        
                        lines = fr.readlines()
                        if len(lines) >= num_samples:
                            status = [None for _ in range(num_samples)]
                            for idx, line in enumerate(lines):
                                splitted_line = line.strip().split()
                                key = int(splitted_line[0])                                
                                status[key] = (splitted_line[0], splitted_line[1], splitted_line[2])
                            if np.all([x is not None for x in status]):
                                break
                        else:
                            # write to another file for viewing progress
                            progress_file = args.receiver_filename.replace(".txt", "_progress.txt")
                            with open(progress_file, 'w+') as f:
                                f.writelines(lines)
                    fcntl.flock(fr, fcntl.LOCK_UN)
                time.sleep(1)
            assert len(target_batch) == len(status)            
            mean_score = sum([float(score) for _, _, score in status])/len(status)
            logger.info(f"Batch {batch} mean score {str(mean_score)}")
            all_sks += status
            all_targets += target_batch
            data = []
            for res, (sk, target) in zip(all_sks, all_targets):
                smiles, best_smi, index = res[1].split(DELIM)
                assert smiles == target
                assert int(index) == sk.index
                score = res[2]
                data_dict = {'target': target, 'smiles': best_smi, 'index': int(index), 'sim': score}
                data.append(data_dict)
            df = pd.DataFrame(data)
            path = os.path.join(args.out_dir, 'reconstruct.csv')
            df.to_csv(path)
            print(os.path.abspath(path))
        else:            
            target_batch = [(deepcopy(lookup[smi][j]), smi) for smi in target_batch for j in range(len(lookup[smi]))]
            if args.ncpu == 1:
                sks_batch = []
                for arg in tqdm(target_batch):                        
                    sks = decode(*arg)
                    sks_batch.append(sks)                                      
            else:
                with ThreadPool(args.ncpu) as p:
                    sks_batch = p.starmap(decode, tqdm(target_batch))        
            mask = [sks is not None for sks in sks_batch]
            target_batch = [t for (t, b) in zip(target_batch, mask) if b]
            sks_batch = [t for (t, b) in zip(sks_batch, mask) if b]
            all_targets += target_batch
            all_sks += sks_batch
            if args.forcing_eval:
                correct_summary = get_metrics(target_batch, sks_batch)
                logger.info(f"correct summary: {correct_summary}")
            else:
                batch_correct, batch_incorrect = get_metrics(target_batch, sks_batch)
                logger.info(f"batch {batch} correct: {format_metrics(batch_correct, cum=True)}")
                logger.info(f"batch {batch} incorrect: {format_metrics(batch_incorrect)}")
        
            if args.forcing_eval:
                correct_summary = get_metrics(targets, all_sks)
                logger.info(f"correct summary: {correct_summary}")
            else:
                total_correct, total_incorrect = get_metrics(all_targets, all_sks)
                logger.info(f"total correct: {format_metrics(total_correct, cum=True)}")
                logger.info(f"total incorrect: {format_metrics(total_incorrect)}")        
            
            if args.mermaid:
                breakpoint()
        
  
    logger.info("Finished decoding.")

    return




if __name__ == "__main__":

    # Parse input args
    args = get_args()
    breakpoint()
    main(args)
