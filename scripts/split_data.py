import os
import numpy as np
import shutil
from tqdm import tqdm


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument("--in-dir", type=str)
    parser.add_argument("--partition_size", type=int, default=1)
    parser.add_argument("--gnn-datasets", type=int, nargs='+')
    parser.add_argument("--out-dir", type=str)    
    return parser.parse_args()



def main(args):
    input_dir = args.in_dir
    p_size = args.partition_size
    used_is = []
    i = 0
    if not args.gnn_datasets:
        files = os.listdir(input_dir)        
        indices = [f.split('_')[0] for f in files]
        indices = list(map(int, set(indices)))
        indices = sorted(indices)
        print(f"gnn-datasets has been set to {indices}")
        setattr(args, 'gnn_datasets', indices)    
    while i < 100:
        if i not in args.gnn_datasets:
            i += 1
            continue
        does_exist = os.path.exists(os.path.join(input_dir, f"{i}_edge_index.npy"))        
        if not does_exist:
            i += 1
            continue
        # copy edge_index
        shutil.copyfile(os.path.join(input_dir, f"{i}_edge_index.npy"), os.path.join(args.out_dir, f"{i}_edge_index.npy"))
        edge_index = np.load(os.path.join(input_dir, f"{i}_edge_index.npy"))
        num_nodes = edge_index.max()+1
        index = 0                
        smiles, node_mask, X, y = None, None, None, None
        while os.path.exists(os.path.join(input_dir, f"{i}_{index}_node_masks.npy")):
            # y = np.load(os.path.join(input_dir, f"{i}_{index}_ys.npy"))            
            print(f"{i}_{index}")
            if node_mask is not None:
                node_mask = np.concatenate((node_mask, np.load(os.path.join(input_dir, f"{i}_{index}_node_masks.npy"))), axis=0)
            else:
                node_mask = np.load(os.path.join(input_dir, f"{i}_{index}_node_masks.npy"))
            if smiles is not None:
                smiles = np.concatenate((smiles, np.load(os.path.join(input_dir, f"{i}_{index}_smiles.npy"))))
            else:
                smiles = np.load(os.path.join(input_dir, f"{i}_{index}_smiles.npy"))
            if X is not None:
                X = np.concatenate((X, np.load(os.path.join(input_dir, f"{i}_{index}_Xs.npy"))), axis=0)
            else:
                X = np.load(os.path.join(input_dir, f"{i}_{index}_Xs.npy"))
            if y is not None:
                y = np.concatenate((y, np.load(os.path.join(input_dir, f"{i}_{index}_ys.npy"))))
            else: 
                y = np.load(os.path.join(input_dir, f"{i}_{index}_ys.npy"))
            index += 1

        start_inds = [0]
        for j, s in enumerate(smiles):
            if s == smiles[start_inds[-1]]:
                continue
            start_inds.append(j)
        for j in range(len(start_inds)-1):
            assert (start_inds[j+1]-start_inds[j]) == (start_inds[1]-start_inds[0])

        d = start_inds[1]
        n = len(start_inds)            
        num_p = (n+p_size-1)//p_size
        print(f"{i}_{index} splitting {n} trees into {num_p} {p_size}-sized tree partitions")            
        for k in tqdm(range(num_p)):
            start_ind = start_inds[k*p_size]                        
            end_ind = start_ind + p_size*d
            node_mask_k = node_mask[start_ind:end_ind]
            X_k = X[num_nodes*start_ind:num_nodes*end_ind]
            y_k = y[num_nodes*start_ind:num_nodes*end_ind]
            smiles_k = smiles[start_ind:end_ind]
            np.save(os.path.join(args.out_dir, f"{i}_{k}_Xs.npy"), X_k)
            np.save(os.path.join(args.out_dir, f"{i}_{k}_ys.npy"), y_k)
            np.save(os.path.join(args.out_dir, f"{i}_{k}_smiles.npy"), smiles_k)
            np.save(os.path.join(args.out_dir, f"{i}_{k}_node_masks.npy"), node_mask_k)
    
        used_is.append(str(i))
        i += 1



if __name__ == "__main__":
    args = get_args()
    main(args)
