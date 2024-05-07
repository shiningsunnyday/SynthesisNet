import torch
import torch.multiprocessing as mp
import numpy as np
import fcntl
import argparse
import setproctitle
from rdkit import Chem
from synnet.utils.reconstruct_utils import set_models, load_data, decode, reconstruct, test_skeletons, lock
from synnet.utils.data_utils import Skeleton, SkeletonSet
from synnet.config import DELIM
import pickle

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
        default="test",
        help="Choose from ['train', 'valid', 'test', 'chembl'] or provide a file with one SMILES per line.",
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
    # Processing
    parser.add_argument("--ncpu", type=int, default=1, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    # For running process
    parser.add_argument('--proc_id', type=int, default=1, help="process id")
    parser.add_argument('--filename', type=str, default="generated_samples.txt", help="file name to lister")
    parser.add_argument('--output_filename', type=str, default="output_syn.txt", help="file name to output")
    return parser.parse_args()    





def main(proc_id, filename, output_filename):
    args = get_args()
    set_models(args)
    load_data(args)
    skeletons = pickle.load(open(args.skeleton_set_file, 'rb'))
    skeleton_set = SkeletonSet().load_skeletons(skeletons)
    SKELETON_INDEX = test_skeletons(args, skeleton_set)
    print(f"SKELETON INDEX: {SKELETON_INDEX}")    
    while(True):        
        selected_mol = None
        with open(filename, 'r') as f:
            editable = lock(f)
            if editable:
                lines = f.readlines()
                num_samples = len(lines)
                new_lines = []
                for idx, line in enumerate(lines):
                    splitted_line = line.strip().split()
                    if len(splitted_line) == 1 and (selected_mol is None):
                        selected_mol = (idx, splitted_line[0])
                        new_line = "{} {}\n".format(splitted_line[0], "working")
                    else:
                        new_line = "{}\n".format(" ".join(splitted_line))
                    new_lines.append(new_line)
                with open(filename, 'w') as fw:
                    for _new_line in new_lines:
                        fw.write(_new_line)
                fcntl.flock(f, fcntl.LOCK_UN)
        if selected_mol is None:            
            continue
        
        print("====Working for sample {}/{}====".format(selected_mol[0], num_samples))
        try:
            smiles, index = selected_mol[1].split(DELIM)
        except:
            print(selected_mol)
        if set([c for c in smiles]) == set(['0', '1']): # fp
            smiles = np.array(list(map(int, smiles)), dtype=bool)
        index = int(index)
        st = list(skeletons)[index]               
        sk = Skeleton(st, index)
        sks = decode(sk, smiles)
        ans = 0.
        best_smi = ''
        for sk in sks:
            score, smi = reconstruct(sk, smiles)
            if score > ans:
                ans = score
                best_smi = smi                
    
        res = DELIM.join([selected_mol[1].split(DELIM)[0], best_smi, str(index)])
        while(True):
            with open(output_filename, 'a') as f:
                editable = lock(f)
                if editable:
                    f.write("{} {} {}\n".format(selected_mol[0], res, ans))
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break


if __name__ == "__main__":
    args = get_args()
    setproctitle.setproctitle("reconstruct_listener")
    main(args.proc_id, args.filename, args.output_filename)
