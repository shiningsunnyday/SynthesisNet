from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton, SkeletonSet
from synnet.models.mlp import GNN
from synnet.models.rt1 import _fetch_molembedder

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch


import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
import torch
import logging
from pathlib import Path
from synnet.models.common import get_args

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
MODEL_ID = Path(__file__).stem


def load_dataloaders(input_dir, args):
    i = 0
    datasets_train = []
    datasets_valid = []
    datasets_test = []
    while True:
        try:
            edge_index = np.load(os.path.join(input_dir, f"{i}_edge_index.npy"))
            node_masks = np.load(os.path.join(input_dir, f"{i}_node_masks.npy"))
            num_nodes = edge_index.max()+1
            X = np.load(os.path.join(input_dir, f"{i}_Xs.npy"))
            y = np.load(os.path.join(input_dir, f"{i}_ys.npy"))
            X = X.reshape(-1, num_nodes, X.shape[-1])     
            y = y.reshape(-1, num_nodes, y.shape[-1])
            
            start_inds = np.where(node_masks.sum(axis=-1) == node_masks.sum(axis=-1).min())[0] # each distinct tree
            n = len(start_inds)
            train_ind, val_ind = start_inds[int(0.8*n)], start_inds[int(0.9*n)]
            X_train, y_train, node_mask_train = X[:train_ind], y[:train_ind], node_masks[:train_ind]
            X_valid, y_valid, node_mask_valid = X[train_ind+1:val_ind], y[train_ind+1:val_ind], node_masks[train_ind+1:val_ind]
            X_test, y_test, node_mask_test = X[val_ind+1:], y[val_ind+1:], node_masks[val_ind+1:]
            # X_train, X_valid_test, node_mask_train, node_mask_valid_test, y_train, y_valid_test = train_test_split(X, node_masks, y, test_size=0.2, random_state=42)
            # X_valid, X_test, node_mask_valid, node_mask_test, y_valid, y_test = train_test_split(X_valid_test, node_mask_valid_test, y_valid_test, test_size=0.5, random_state=42)            
        except:
            break
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
        break
        i += 1
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)
    dataset_test = torch.utils.data.ConcatDataset(datasets_test)
    train_dataloader = DataLoader([Data(edge_index=data[0], x=data[2], y=data[3]) for data in dataset_train], batch_size=args.batch_size, num_workers=args.ncpu, shuffle=True)
    valid_dataloader = DataLoader([Data(edge_index=data[0], x=data[2], y=data[3]) for data in dataset_valid], batch_size=args.batch_size, num_workers=args.ncpu)
    test_dataloader = DataLoader([Data(edge_index=data[0], x=data[2], y=data[3]) for data in dataset_test], batch_size=args.batch_size, num_workers=args.ncpu)
    return train_dataloader, valid_dataloader, test_dataloader



if __name__ == "__main__":
    args = get_args()
    train_dataloader, valid_dataloader, _ = load_dataloaders(args.gnn_input_feats, args)
    input_dim = 4097
    out_dim = 257
    molembedder = _fetch_molembedder(args)
    gnn = GNN(
        c_in=input_dim,
        c_out=out_dim,
        c_hidden=1200,
        num_layers=5,
        dp_rate=0.5,
        task="regression",
        loss="mse",
        valid_loss="faiss-knn",
        optimizer="adam",
        learning_rate=3e-4,
        val_freq=1,
        molembedder=molembedder,
        ncpu=args.ncpu,
        X=args.mol_embedder_file if args.mol_embedder_file else None
    )

    # Set up Trainer
    save_dir = Path("results/logs/") / MODEL_ID
    save_dir.mkdir(exist_ok=True, parents=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    csv_logger = pl_loggers.CSVLogger(tb_logger.log_dir, name="", version="")
    logger.info(f"Log dir set to: {tb_logger.log_dir}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_logger.log_dir,
        filename="ckpts.{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=3)
    tqdm_callback = TQDMProgressBar(refresh_rate=int(len(train_dataloader) * 0.05))

    max_epochs = args.epoch if not args.debug else 100
    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, tqdm_callback],
        logger=[tb_logger, csv_logger],
        fast_dev_run=args.fast_dev_run,
        use_distributed_sampler=False
    )

    logger.info(f"Start training")
    trainer.fit(gnn, train_dataloader, valid_dataloader)
    logger.info(f"Training completed.")
