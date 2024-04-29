from ga.config import *
from ga.search import *
from ga.utils import *
from synnet.utils.analysis_utils import serialize_string
from synnet.utils.reconstruct_utils import *
from synnet.utils.data_utils import binary_tree_to_skeleton
import pickle
from functools import partial
from tqdm import tqdm
import logging
from tdc import Oracle
import time

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
    parser.add_argument("--top-bbs-file", help='if given, consider only these bbs')
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
        "--skeleton-file",
        type=str,
        default="results/viz/top_1000/skeletons-top-1000.pkl",
        help="Input file for the skeletons of syntree-file",
    )   
    parser.add_argument("--forcing-eval", action='store_true')
    parser.add_argument("--mermaid", action='store_true')
    parser.add_argument("--top-k", default=1, type=int)
    parser.add_argument("--filter-only", type=str, nargs='+', choices=['rxn', 'bb'], default=[])    
    parser.add_argument("--log_file")
    parser.add_argument("--config_file", required=True)
    # Processing
    parser.add_argument("--ncpu", type=int, help="Number of cpus")
    # Evaluation
    parser.add_argument(
        "--objective", type=str, default="qed", help="Objective function to optimize",
        choices=['qed', 'logp', 'jnk', 'gsk', 'drd2', '7l11', 'drd3']
    )    
    # I/O
    parser.add_argument("--sender-filename")
    parser.add_argument("--receiver-filename")    
    return parser.parse_args()  


def test_fitness(batch):
    for ind in batch:
        fp = ind.fp
        bt = ind.bt
        leaves = [v for v in bt.nodes() if (bt.out_degree(v) == 0)]
        ind.fitness = 0.05 * (fp[:100].sum() - fp[100:].sum()) + len(leaves)



def test_surrogate(ncpu, sender_filename, receiver_filename, batch):
    if sender_filename and receiver_filename:        
        open(sender_filename, 'w+').close() # clear
        open(receiver_filename, 'w+').close() # clear            
        while(True):
            with open(sender_filename, 'r') as fr:
                editable = lock(fr)
                if editable:
                    with open(sender_filename, 'w') as fw:
                        for ind in batch:
                            sk = binary_tree_to_skeleton(ind.bt)
                            tree_key = serialize_string(sk.tree, sk.tree_root)
                            bits = list(map(str, map(int, ind.fp)))
                            fp = ''.join(bits)
                            index = lookup_skeleton_key(sk.zss_tree, tree_key)
                            sample = DELIM.join([fp, str(index)])
                            fw.write('{}\n'.format(sample))
                    break
                fcntl.flock(fr, fcntl.LOCK_UN)     
        num_samples = len(batch)
        print("Waiting for reconstruct evaluation...")       
        while(True):
            with open(args.receiver_filename, 'r') as fr:
                editable = lock(fr)
                if editable:
                    res = []
                    lines = fr.readlines()
                    if len(lines) >= num_samples:  
                        seen_idxes = set()                      
                        for idx, line in enumerate(lines):
                            splitted_line = line.strip().split()
                            key = splitted_line[0]
                            if key in seen_idxes:
                                continue
                            seen_idxes.add(key)
                            best_smi = splitted_line[1].split(DELIM)[1]
                            score = oracle(best_smi)
                            res.append((score, best_smi))
                        break
                fcntl.flock(fr, fcntl.LOCK_UN)
            time.sleep(1)
    else:
        pargs = []
        for ind in batch:
            fp = ind.fp
            bt = ind.bt        
            sk = binary_tree_to_skeleton(bt)  
            pargs.append((sk, fp, oracle))        
        # res = [(random.random(), '') for _ in pargs]
        if ncpu > 1:
            with ThreadPool(ncpu) as p:      
                res = p.starmap(surrogate, tqdm(pargs, desc="test_surrogate"))
        else:
            res = [surrogate(*parg) for parg in pargs]


    for ind, (score, smi) in zip(batch, res):
        if score > globals()['best_score']:
            globals()['best_score'] = score
            print("best score", score, smi)
        ind.fitness = score
        ind.smi = smi


def set_oracle(args):
    obj = args.objective
    if obj == "qed":
        # define the oracle function from the TDC
        return Oracle(name="QED")        
    elif obj == "logp":
        # define the oracle function from the TDC
        return Oracle(name="LogP")        
    elif obj == "jnk":
        # return oracle function from the TDC
        return Oracle(name="JNK3")        
    elif obj == "gsk":
        # return oracle function from the TDC
        return Oracle(name="GSK3B")        
    elif obj == "drd2":
        # return oracle function from the TDC
        return Oracle(name="DRD2")        
    elif obj == "7l11":
        return dock_7l11
    elif obj == "drd3":
        return dock_drd3
    else:
        raise ValueError("Objective function not implemneted")    



def init_global_vars(args):
    set_models(args, logger)
    load_data(args, logger)
    skeletons = pickle.load(open(args.skeleton_set_file, 'rb'))
    skeleton_set = SkeletonSet().load_skeletons(skeletons)
    SKELETON_INDEX = test_skeletons(args, skeleton_set)    
    print(f"SKELETON INDEX: {SKELETON_INDEX}")
    logger.info(f"SKELETON INDEX: {SKELETON_INDEX}")
    oracle = set_oracle(args)
    globals()['oracle'] = oracle
    globals()['best_score'] = float("-inf")



def main(args):
    if args.log_file:
        handler = logging.FileHandler(args.log_file)
    logger.addHandler(handler)    
    init_global_vars(args)    
    config_file = json.load(open(args.config_file))
    config = GeneticSearchConfig(**config_file)
    GeneticSearch(config).optimize(partial(test_surrogate, config.ncpu, args.sender_filename, args.receiver_filename))



if __name__ == "__main__":
    args = get_args()
    breakpoint()
    main(args)
    