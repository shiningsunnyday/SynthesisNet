from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from synnet.data_generation.syntrees import MorganFingerprintEncoder
from synnet.models.mlp import MLP
import torch.nn as nn
import time
import logging
import json
import rdkit.Chem as Chem

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class Predictor(nn.Module):
    def __init__(self, out_dim, in_dim=2048, dim=512,
                 dropout_rate=0.3):
        super(Predictor, self).__init__()
        self.in_dim = in_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(in_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(dim,dim)
        # self.bn2 = nn.BatchNorm1d(dim)
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim,out_dim)

    def forward(self,x, y=None, loss_fn =nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        # x = self.dropout1(F.elu(self.fc1(x)))
        # x = self.dropout2(F.elu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        if y is not None :
            return loss_fn(x, y)
        else:
            return x
            
    

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/pre-process/syntrees/synthetic-trees.json.gz",
        help="Input file for the generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--skeleton-file",
        type=str,
        default="results/viz/skeletons.pkl",
        help="Input file for the skeletons of syntree-file",
    )   
    parser.add_argument(
        "--work-dir",
        type=str,
        default=f"results/logs/recognizer/{time.time()}",
        help="Input file for the generated synthetic trees (*.json.gz)",
    )   
    # Training args
    parser.add_argument("--max_epochs", type=int, default=10)     
    parser.add_argument("--nbits", type=int, default=2048)   
    parser.add_argument("--num_per_class", type=int, default=100)        
    parser.add_argument("--batch_size", type=int, default=32)    
    parser.add_argument("--fp_radii", type=int, default=2)    
    parser.add_argument("--hidden_dim", type=int, default=512) 
    parser.add_argument('--cuda', type=int, default=-1)
    return parser.parse_args()


def skeleton2graph(skeleton):
    graph = nx.MultiDiGraph()
    count = {}
    lookup = {}
    for n in skeleton.nodes:
        name = n.smiles
        if n.smiles in count:
            name += f":{count[n.smiles]}"
        graph.add_node(name)
        count[n.smiles] = count.get(n.smiles, 0)+1
        lookup[n] = name
    for e in skeleton.edges:
        graph.add_edge(lookup[skeleton.nodes[e[0]]], lookup[skeleton.nodes[e[1]]])
    return graph


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.work_dir, exist_ok=True)  
    same = False
    config_path = os.path.join(args.work_dir, 'config.json')
    if os.path.exists(config_path):
        same = True
        args_dict = json.load(open(config_path, 'r'))
        for k in args_dict:
            if args_dict[k] != getattr(args, k):
                same = False 
    
    skeletons = pickle.load(open(args.skeleton_file, 'rb'))          
    nbits = args.nbits    
    num_per_class = args.num_per_class
    skeletons = {k:skeletons[k] for k in skeletons if len(skeletons[k]) >= num_per_class}
    print(f"{len(skeletons)} classes with >= {num_per_class} per class")
    num_classes = len(skeletons)
    setattr(args, 'num_classes', num_classes)    
    if same:
        train_dataset = pickle.load(open(os.path.join(args.work_dir, 'train_dataset.pkl'), 'rb'))
        valid_dataset = pickle.load(open(os.path.join(args.work_dir, 'valid_dataset.pkl'), 'rb'))
    else:      
        encoder = MorganFingerprintEncoder(args.fp_radii, nbits)
        lookup = {}
        for i, k in enumerate(skeletons):            
            for j, st in enumerate(skeletons[k]):                
                try: feats = encoder.encode(st.root.smiles).flatten()                    
                except: continue
                if st.root.smiles not in lookup:
                    lookup[st.root.smiles] = {'labels': np.zeros((len(skeletons),)), 'features': feats}                
                lookup[st.root.smiles]['labels'][i] = 1
        features, labels = [], []
        for smi in lookup:
            features.append(lookup[smi]['features'])
            label = lookup[smi]['labels']
            label = label/label.sum()
            labels.append(label)

        # X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33, stratify=labels)
        X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33)
        print("valid unique counts:", np.unique(y_valid, return_counts=True))
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train),
            torch.Tensor(y_train),
        )    
        valid_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_valid),
            torch.Tensor(y_valid),
        )
        pickle.dump(train_dataset, open(os.path.join(args.work_dir, 'train_dataset.pkl'), 'wb+'))
        pickle.dump(valid_dataset, open(os.path.join(args.work_dir, 'valid_dataset.pkl'), 'wb+'))
        json.dump(args.__dict__, open(config_path, 'w+'))

    mlp = MLP(
        input_dim=args.nbits,
        output_dim=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=5,
        dropout=0.5,
        num_dropout_layers=1,
        task="classification",
        loss="cross_entropy",
        valid_loss="accuracy",
        optimizer="adam",
        learning_rate=1e-4,
        val_freq=1,
        ncpu=0
    )            
    breakpoint()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)    
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)    
    
    save_dir = args.work_dir
    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    csv_logger = pl_loggers.CSVLogger(save_dir, name="", version="")
    tqdm_callback = TQDMProgressBar(refresh_rate=int(len(train_dataloader) * 0.05))

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=save_dir,
        filename="ckpts.{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )  
    if args.cuda > -1:
        gpu_kwargs = {'accelerator': 'gpu'}
        gpu_kwargs = {'devices': [args.cuda]}
    else:        
        gpu_kwargs = {'accelerator': 'cpu'}      
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, tqdm_callback],
        logger=[tb_logger, csv_logger],
        **gpu_kwargs
    )
    logger.info(f"Start training")
    trainer.fit(mlp, train_dataloader, valid_dataloader)
    logger.info(f"Training completed.")    
    