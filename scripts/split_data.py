import os
import numpy as np
from scipy import sparse
import shutil
from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method('fork')


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument("--in-dir", type=str)
    parser.add_argument("--partition_size", type=int, default=1)
    parser.add_argument("--gnn-datasets", type=int, nargs='+')
    parser.add_argument("--out-dir", type=str)    
    return parser.parse_args()


def load_npz(input_dir, index_str, suffix):
    return sparse.load_npz(os.path.join(input_dir, f"{index_str}_{suffix}.npz")).todense()


def load_npy(input_dir, index_str, suffix):
    return np.load(os.path.join(input_dir, f"{index_str}_{suffix}.npy"))


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
        max_ind = max(indices)+1
    else:
        max_ind = 1000000
    while i < max_ind:
        if i not in args.gnn_datasets:
            i += 1
            continue
        does_exist = os.path.exists(os.path.join(input_dir, f"{i}_edge_index.npy"))        
        if not does_exist:
            i += 1
            continue
        # copy edge_index
        os.makedirs(args.out_dir, exist_ok=True)
        shutil.copyfile(os.path.join(input_dir, f"{i}_edge_index.npy"), os.path.join(args.out_dir, f"{i}_edge_index.npy"))
        edge_index = np.load(os.path.join(input_dir, f"{i}_edge_index.npy"))
        num_nodes = edge_index.max()+1
        index = 0                
        smiles, node_mask, X, y = None, None, None, None
        while os.path.exists(os.path.join(input_dir, f"{i}_{index}_node_masks.npz")):
            # # y = np.load(os.path.join(input_dir, f"{i}_{index}_ys.npy"))            
            # print(f"{i}_{index}")
            # node_mask_path = os.path.join(input_dir, f"{i}_{index}_node_masks.npz")
            # smiles_path = os.path.join(input_dir, f"{i}_{index}_smiles.npy")
            # Xs_path = os.path.join(input_dir, f"{i}_{index}_Xs.npz")
            # ys_path = os.path.join(input_dir, f"{i}_{index}_ys.npz")
            # if node_mask is not None:
            #     node_mask = np.concatenate((node_mask, sparse.load_npz(node_mask_path).todense()), axis=0)
            # else:
            #     node_mask = sparse.load_npz(node_mask_path).todense()
            # if smiles is not None:
            #     smiles = np.concatenate((smiles, np.load(smiles_path)))
            # else:
            #     smiles = np.load(smiles_path)
            # if X is not None:
            #     X = np.concatenate((X, sparse.load_npz(Xs_path).todense()), axis=0)
            # else:
            #     X = sparse.load_npz(Xs_path).todense()
            # if y is not None:
            #     y = np.concatenate((y, sparse.load_npz(ys_path).todense()))
            # else: 
            #     y = sparse.load_npz(ys_path).todense()
            index += 1
        with mp.Pool(50) as p:
            res = p.starmap(load_npz, [(input_dir,f"{i}_{ind}","Xs") for ind in tqdm(range(index), desc="loading X")])
        X = np.concatenate(res, axis=0)
        with mp.Pool(50) as p:
            res = p.starmap(load_npz, [(input_dir,f"{i}_{ind}","ys") for ind in tqdm(range(index), desc="loading y")])            
        y = np.concatenate(res, axis=0)            
        with mp.Pool(50) as p:
            res = p.starmap(load_npz, [(input_dir,f"{i}_{ind}","node_masks") for ind in tqdm(range(index), desc="loading node_mask")])             
        node_mask = np.concatenate(res, axis=0)                            
        with mp.Pool(50) as p:
            res = p.starmap(load_npy, [(input_dir,f"{i}_{ind}","smiles") for ind in tqdm(range(index), desc="loading smiles")])                         
        smiles = np.concatenate(res, axis=0)   

        # mask to node_mask.sum() < len(node_mask)
        assert node_mask.shape[0]*node_mask.shape[1] == y.shape[0]
        node_mask.sum(axis=-1) == node_mask.shape[1]

        start_inds = []
        prev_sum = node_mask[0].sum()
        for j, nm in enumerate(node_mask):
            assert nm.sum() >= prev_sum
            if nm.sum() == prev_sum:            
                start_inds.append(j)
        for j in range(len(start_inds)-1):
            if (start_inds[j+1]-start_inds[j]) != (start_inds[1]-start_inds[0]):
                breakpoint()

        if len(start_inds) == 1:
            d = len(smiles)
        else:
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
            X_k = sparse.csr_array(X_k)
            y_k = sparse.csr_array(y_k)
            node_mask_k = sparse.csr_array(node_mask_k)
            sparse.save_npz(os.path.join(args.out_dir, f"{i}_{k}_Xs.npz"), X_k)
            sparse.save_npz(os.path.join(args.out_dir, f"{i}_{k}_ys.npz"), y_k)
            np.save(os.path.join(args.out_dir, f"{i}_{k}_smiles.npy"), smiles_k)
            sparse.save_npz(os.path.join(args.out_dir, f"{i}_{k}_node_masks.npz"), node_mask_k)

        used_is.append(str(i))
        i += 1



if __name__ == "__main__":
    args = get_args()
    main(args)
