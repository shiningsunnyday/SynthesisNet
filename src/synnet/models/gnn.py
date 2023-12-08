from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton, SkeletonSet
from synnet.models.mlp import GNN
from synnet.models.rt1 import _fetch_molembedder

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import torch

import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import logging
from pathlib import Path
from synnet.models.common import get_args

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
MODEL_ID = Path(__file__).stem


class PtrDataset(Dataset):
    def __init__(self, ptrs, **kwargs):
        super().__init__(**kwargs)
        self.ptrs = ptrs
        self.edge_index = {}
        for ptr in ptrs:
            _, e, _ = ptr
            if e in self.edge_index:
                continue
            edges = np.load(e)
            self.edge_index[e] = np.concatenate((edges, edges[::-1]), axis=-1)

    

    def __getitem__(self, idx):
        base, e, index = self.ptrs[idx]
        node_mask = sparse.load_npz(base+'_node_masks.npz').toarray()[index]
        num_nodes = self.edge_index[e].max()+1
        X = sparse.load_npz(base+'_Xs.npz').toarray()
        X = X.reshape(-1, num_nodes, X.shape[-1])[index]
        y = sparse.load_npz(base+'_ys.npz').toarray()
        y = y.reshape(-1, num_nodes, y.shape[-1])[index]        
        key_val = e.split('/')[-1].split('_')[0]+''.join(list(map(str, node_mask)  ))
        data = (
            torch.tensor(self.edge_index[e], dtype=torch.int64),
            np.array([key_val for _ in node_mask]), # index
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return Data(edge_index=data[0], key=data[1], x=data[2], y=data[3])
    
    
    def __len__(self):
        return len(self.ptrs)



def gather_ptrs(input_dir, include_is=[], ratio=[8,1,1]):
    used_is = []
    i = 0
    train_dataset_ptrs = []
    val_dataset_ptrs = []
    test_dataset_ptrs = []    
    train_frac = ratio[0]/sum(ratio)
    val_frac = (ratio[0]+ratio[1])/sum(ratio)
    while i < 100:
        if i not in include_is:
            i += 1
            continue
        does_exist = os.path.exists(os.path.join(input_dir, f"{i}_edge_index.npy"))        
        if not does_exist:
            i += 1
            continue

        index = 0        
        while os.path.exists(os.path.join(input_dir, f"{i}_{index}_node_masks.npz")):
            # y = np.load(os.path.join(input_dir, f"{i}_{index}_ys.npy"))
            node_mask = sparse.load_npz(os.path.join(input_dir, f"{i}_{index}_node_masks.npz"))
            smiles = np.load(os.path.join(input_dir, f"{i}_{index}_smiles.npy"))
            start_inds = [0]
            for j, s in enumerate(smiles):
                if s == smiles[start_inds[-1]]:
                    continue
                start_inds.append(j)               
            start_inds = np.where(node_mask.sum(axis=-1) == node_mask.sum(axis=-1).min())[0] # each distinct tree
            n = len(start_inds)
            print(f"splitting {n} trees into {ratio[0]}-{ratio[1]}-{ratio[2]} for skeleton {i}")
            train_ind = start_inds[int(train_frac*n)] if int(train_frac*n)<n else n
            val_ind = start_inds[int(val_frac*n)] if int(val_frac*n)<n else n
            for j in range(train_ind):
                train_dataset_ptrs.append((os.path.join(input_dir, f"{i}_{index}"), 
                os.path.join(input_dir, f"{i}_edge_index.npy"), j))
            for j in range(train_ind, val_ind):
                val_dataset_ptrs.append((os.path.join(input_dir, f"{i}_{index}"), 
                os.path.join(input_dir, f"{i}_edge_index.npy"), j))
            for j in range(val_ind, n):
                test_dataset_ptrs.append((os.path.join(input_dir, f"{i}_{index}"), 
                os.path.join(input_dir, f"{i}_edge_index.npy"), j))                
            index += 1     
        used_is.append(str(i))
        i += 1    
    return train_dataset_ptrs, val_dataset_ptrs, test_dataset_ptrs, used_is



def load_lazy_dataloaders(args):
    input_dir = args.gnn_input_feats
    if not args.gnn_datasets:
        files = os.listdir(input_dir)        
        indices = [f.split('_')[0] for f in files]
        indices = list(map(int, set(indices)))
        indices = sorted(indices)
        print(f"gnn-datasets has been set to {indices}")
        setattr(args, 'gnn_datasets', indices)    
    
    train_dataset_ptrs, val_dataset_ptrs, test_dataset_ptrs, used_is = gather_ptrs(input_dir, args.gnn_datasets)        
    dataset_train = PtrDataset(train_dataset_ptrs)
    dataset_valid = PtrDataset(val_dataset_ptrs)
    dataset_test = PtrDataset(test_dataset_ptrs)
    prefetch_factor = args.prefetch_factor if args.prefetch_factor else None
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.ncpu, shuffle=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    valid_dataloader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.ncpu, prefetch_factor=prefetch_factor, persistent_workers=True)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.ncpu, prefetch_factor=prefetch_factor, persistent_workers=True)
    return train_dataloader, valid_dataloader, test_dataloader, ','.join(used_is)


def load_split_dataloaders(args):
    input_dir = {}
    used_is = {}
    
    for split in ['train', 'valid', 'test']:
        input_dir[split] = Path(args.gnn_input_feats).parent / (Path(args.gnn_input_feats).name+f'_{split}')        

    if not args.gnn_datasets:
        files = os.listdir(input_dir)        
        indices = [f.split('_')[0] for f in files]
        indices = list(map(int, set(indices)))
        indices = sorted(indices)
        print(f"gnn-datasets has been set to {indices}")
        setattr(args, 'gnn_datasets', indices)    

    train_dataset_ptrs, _, _, used_is['train'] = gather_ptrs(input_dir['train'], args.gnn_datasets, [1,0,0])
    _, val_dataset_ptrs, _, used_is['valid'] = gather_ptrs(input_dir['valid'], args.gnn_datasets, [0,1,0])
    _, _, test_dataset_ptrs, used_is['test'] = gather_ptrs(input_dir['test'], args.gnn_datasets, [0,0,1])        
    if not (used_is['train'] == used_is['valid'] == used_is['test']):
        breakpoint()
    dataset_train = PtrDataset(train_dataset_ptrs)
    dataset_valid = PtrDataset(val_dataset_ptrs)
    dataset_test = PtrDataset(test_dataset_ptrs)
    if dataset_test.__len__() > dataset_train.__len__(): # TODO: one-time hack, get rid
        dataset_test, dataset_train = dataset_train, dataset_test
    prefetch_factor = args.prefetch_factor if args.prefetch_factor else None
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.ncpu, shuffle=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    valid_dataloader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.ncpu, prefetch_factor=prefetch_factor, persistent_workers=True)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.ncpu, prefetch_factor=prefetch_factor, persistent_workers=True)
    return train_dataloader, valid_dataloader, test_dataloader, ','.join(used_is)    


def load_dataloaders(args):
    input_dir = args.gnn_input_feats
    i = 0
    datasets_train = []
    datasets_valid = []
    datasets_test = []
    used_is = []
    if not args.gnn_datasets:
        files = os.listdir(input_dir)        
        indices = [f.split('_')[0] for f in files]
        indices = list(map(int, set(indices)))
        indices = sorted(indices)
        indices = ' '.join(list(map(str, indices)))
        print(f"gnn-datasets has been set to {indices}")
        setattr(args, 'gnn-datasets', indices)
    while i < 100:
        if i not in args.gnn_datasets:
            i += 1
            continue
        if not os.path.exists(os.path.join(input_dir, f"{i}_edge_index.npy")):
            i += 1
            continue

        edge_index = np.load(os.path.join(input_dir, f"{i}_edge_index.npy"))
        node_masks, Xs, ys = [], [], []
        smiles = []
        index = 0        
        while os.path.exists(os.path.join(input_dir, f"{i}_{index}_node_masks.npz")):
            node_masks.append(sparse.load_npz(os.path.join(input_dir, f"{i}_{index}_node_masks.npz")))
            num_nodes = edge_index.max()+1
            X = sparse.load_npz(os.path.join(input_dir, f"{i}_{index}_Xs.npz"))
            y = sparse.load_npz(os.path.join(input_dir, f"{i}_{index}_ys.npz"))
            smile = np.load(os.path.join(input_dir, f"{i}_{index}_smiles.npy"))
            smiles.append(smile)
            X = X.reshape(-1, num_nodes, X.shape[-1])
            y = y.reshape(-1, num_nodes, y.shape[-1])
            Xs.append(X)
            ys.append(y)
            print(f"loaded {i}_{index}")
            index += 1
        node_masks = np.concatenate(node_masks, axis=0)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        smiles = np.concatenate(smiles)[:, None]
        start_inds = np.where(node_masks.sum(axis=-1) == node_masks.sum(axis=-1).min())[0] # each distinct tree
        n = len(start_inds)
        print(f"splitting {n} trees into 80-10-10 with {index} partitions for skeleton {i}")
        train_ind, val_ind = start_inds[int(0.8*n)], start_inds[int(0.9*n)]
        X_train, y_train, node_mask_train, smiles_train = X[:train_ind], y[:train_ind], node_masks[:train_ind], smiles[:train_ind]
        X_valid, y_valid, node_mask_valid, smiles_valid = X[train_ind+1:val_ind], y[train_ind+1:val_ind], node_masks[train_ind+1: val_ind], smiles[train_ind+1: val_ind]
        X_test, y_test, node_mask_test, smiles_test = X[val_ind+1:], y[val_ind+1:], node_masks[val_ind+1:], smiles[val_ind+1:]
        # X_train, X_valid_test, node_mask_train, node_mask_valid_test, y_train, y_valid_test = train_test_split(X, node_masks, y, test_size=0.2, random_state=42)
        # X_valid, X_test, node_mask_valid, node_mask_test, y_valid, y_test = train_test_split(X_valid_test, node_mask_valid_test, y_valid_test, test_size=0.5, random_state=42)            

        dataset_train = torch.utils.data.TensorDataset(
            torch.tensor([edge_index for _ in range(X_train.shape[0])], dtype=torch.int64),
            torch.Tensor(node_mask_train),
            torch.Tensor(X_train),
            torch.Tensor(y_train)
        )
        dataset_valid = torch.utils.data.TensorDataset(
            torch.tensor([edge_index for _ in range(X_valid.shape[0])], dtype=torch.int64),
            torch.Tensor(node_mask_valid),
            torch.Tensor(X_valid),
            torch.Tensor(y_valid)
        )     
        dataset_test = torch.utils.data.TensorDataset(
            torch.tensor([edge_index for _ in range(X_test.shape[0])], dtype=torch.int64),
            torch.Tensor(node_mask_test),
            torch.Tensor(X_test),
            torch.Tensor(y_test)
        )
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
        datasets_test.append(dataset_test)        
        used_is.append(str(i))
        i += 1
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)
    dataset_test = torch.utils.data.ConcatDataset(datasets_test)
    train_dataloader = DataLoader([Data(edge_index=data[0], x=data[2], y=data[3]) for data in dataset_train], batch_size=args.batch_size, num_workers=args.ncpu, shuffle=True)
    valid_dataloader = DataLoader([Data(edge_index=data[0], x=data[2], y=data[3]) for data in dataset_valid], batch_size=args.batch_size, num_workers=args.ncpu)
    test_dataloader = DataLoader([Data(edge_index=data[0], x=data[2], y=data[3]) for data in dataset_test], batch_size=args.batch_size, num_workers=args.ncpu)
    return train_dataloader, valid_dataloader, test_dataloader, ','.join(used_is)


def main(args):
    if args.feats_split:
        train_dataloader, valid_dataloader, _, used_is = load_split_dataloaders(args)
    elif args.lazy_load:
        train_dataloader, valid_dataloader, _, used_is = load_lazy_dataloaders(args)
    else:
        train_dataloader, valid_dataloader, _, used_is = load_dataloaders(args)
    input_dim = 2*2048+91
    out_dim = 256+91
    molembedder = _fetch_molembedder(args)
    gnn = GNN(
        c_in=input_dim,
        c_out=out_dim,
        c_hidden=1200,
        layer_name=args.gnn_layer, # Transformer
        num_layers=5,
        dp_rate=args.gnn_dp_rate, # 0.5
        task="regression",
        loss=args.gnn_loss, # mse+cross_entropy
        valid_loss=args.gnn_valid_loss, # nn_accuracy+cross_entropy
        optimizer="adam",
        learning_rate=args.lr,
        val_freq=1,
        molembedder=molembedder,
        ncpu=args.ncpu,
        X=args.mol_embedder_file,
        datasets=used_is
    )

    # Set up Trainer
    save_dir = Path("results/logs/") / MODEL_ID
    save_dir.mkdir(exist_ok=True, parents=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    csv_logger = pl_loggers.CSVLogger(tb_logger.log_dir, name="", version="")
    logger.info(f"Log dir set to: {tb_logger.log_dir}")

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",
    #     dirpath=tb_logger.log_dir,
    #     filename="ckpts.{epoch}-{val_loss:.2f}",
    #     save_weights_only=False,
    # )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=3)
    tqdm_callback = TQDMProgressBar(refresh_rate=int(len(train_dataloader) * 0.05))

    max_epochs = args.epoch if not args.debug else 100
    # Create trainer
    if args.cuda > -1:
        gpu_kwargs = {'accelerator': 'gpu'}
        gpu_kwargs = {'devices': [args.cuda]}
    else:        
        gpu_kwargs = {'accelerator': 'cpu'}
    trainer = pl.Trainer(        
        max_epochs=max_epochs,
        callbacks=[
            # checkpoint_callback, 
            tqdm_callback],
        logger=[tb_logger, csv_logger],
        fast_dev_run=args.fast_dev_run,
        use_distributed_sampler=False,
        **gpu_kwargs
    )

    logger.info(f"Start training")
    trainer.fit(gnn, train_dataloader, valid_dataloader)
    logger.info(f"Training completed.")



if __name__ == "__main__":
    args = get_args()
    main(args)