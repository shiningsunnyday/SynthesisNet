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
from synnet.utils.reconstruct_utils import *
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import random
random.seed(42)
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
from concurrent.futures import ProcessPoolExecutor
import torch



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
    parser.add_argument("--max_rxns", type=int, help="Restrict syntree test set to max number of reactions (-1 to do syntrees, 0 to syntrees whose skleeton class was trained on by ckpt_dir)", default=-1)
    parser.add_argument("--max_num_rxns", type=int, help="Restrict skeleton prediction to max number of reactions", default=-1)    
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
    # MCMC params
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--mcmc_timesteps", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=1)
    # Visualization
    parser.add_argument("--mermaid", action='store_true')
    parser.add_argument("--one-per-class", action='store_true', help='visualize one skeleton per class')
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
        targets = targets[:1000]
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
            if args.ckpt_recognizer:
                pred_index = predict_skeleton(target, max_num_rxns=args.max_num_rxns)
                sk = Skeleton(list(skeletons)[pred_index], pred_index)
                if args.max_num_rxns != -1:
                    assert sk.rxns.sum() <= args.max_num_rxns
                lookup[target] = sk
            else:
                lookup[target] = sk
    targets = list(lookup)
    print(f"{len(targets)} targets")
    

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")

    # targets = targets[:10]     
    batch_size = args.batch_size        
    # decode('CC(Nc1ccc(I)cc1-c1nc(-c2cc(F)ccc2N=C=O)n[nH]1)c1ccc(N)c(O)c1')
    # decode('CC(C)(N=C=O)C1=Cc2cc(Br)c(C(=O)O)c(N)c2O1')
    all_sks = []    
    all_smiles = []
    for batch in range((len(targets)+batch_size-1)//batch_size):
        target_batch = targets[batch_size*batch:batch_size*batch+batch_size]        
        if args.sender_filename and args.receiver_filename:
            open(args.sender_filename, 'w+').close() # clear
            open(args.receiver_filename, 'w+').close() # clear  
            while(True):
                with open(args.sender_filename, 'r') as fr:
                    editable = lock(fr)
                    if editable:
                        with open(args.sender_filename, 'w') as fw:
                            for smi in target_batch:
                                sample = DELIM.join([smi, str(lookup[smi].index)])
                                fw.write('{}\n'.format(sample))
                        break
                    fcntl.flock(fr, fcntl.LOCK_UN)     
            num_samples = len(target_batch)
            print("Waiting for batch mcmc...")       
            while(True):
                with open(args.receiver_filename, 'r') as fr:
                    editable = lock(fr)
                    if editable:
                        status = []
                        lines = fr.readlines()
                        if len(lines) == num_samples:
                            for idx, line in enumerate(lines):
                                splitted_line = line.strip().split()
                                status.append((splitted_line[0], splitted_line[2], splitted_line[3], splitted_line[4]))
                            break
                    fcntl.flock(fr, fcntl.LOCK_UN)
                time.sleep(1)
            assert len(target_batch) == len(status)
            status = sorted(status, key=lambda tup: int(tup[0]))
            sks_batch = []
            for tup in status:
                sks = []
                for score, smi, index in zip(tup[1].split(','), tup[2].split(','), tup[3].split(',')):
                    sks.append((float(score), smi, index))
                sks_batch.append(sks)
        else:
            if args.ncpu == 1:
                sks_batch = []
                for smi in tqdm(target_batch):                        
                    sks = mcmc(deepcopy(lookup[smi]), smi, args.beta, args.mcmc_timesteps)
                    sks_batch.append(sks)                                      
            else:
                torch.set_num_threads(1)
                with ProcessPoolExecutor(max_workers=args.ncpu) as exe:      
                    batch_future = exe.map(mcmc, tqdm([deepcopy(lookup[smi]) for smi in target_batch]),
                                            target_batch,
                                            [args.beta for _ in target_batch],
                                            [args.mcmc_timesteps for _ in target_batch],
                                            chunksize=args.chunk_size)
                    sks_batch = []
                    for sks in tqdm(batch_future, total=len(target_batch), desc="MCMC"):
                        sks_batch.append(sks)
        all_sks += sks_batch
        all_smiles += target_batch

        # plot average scores per iter
        fig = plt.Figure((10, 10))
        ax = fig.add_subplot(1,1,1)
        scores = [[r[0] for r in res] for res in all_sks]
        ax.plot(np.mean(scores, axis=0))
        path = os.path.join(args.out_dir, 'mcmc.png')
        fig.savefig(path)
        print(os.path.abspath(path))

        # save results
        data = []
        for res, smiles in zip(all_sks, all_smiles):
            data_dict = {'smiles': smiles}
            for iter, (score, smi, index) in enumerate(res):
                data_dict[f'score_{iter+1}'] = score
                data_dict[f'smiles_{iter+1}'] = smi
                data_dict[f'sk_{iter+1}'] = index
            data_dict = {k: data_dict[k] for k in sorted(data_dict)}
            data.append(data_dict)
        path = os.path.join(args.out_dir, 'mcmc.csv')
        df = pd.DataFrame(data)
        df.to_csv(path)
  
    logger.info("Finished mcmc.")
    return




if __name__ == "__main__":

    # Parse input args
    args = get_args()
    breakpoint()
    main(args)
