from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from synnet.data_generation.syntrees import MorganFingerprintEncoder
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.models.mlp import MLP
from synnet.utils.data_utils import Skeleton
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
from sklearn.manifold import TSNE

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
    parser.add_argument(
        "--ckpt",
        help="If given, load ckpt for analysis"
    )
    parser.add_argument("--datasets", type=int, nargs='+')    
    # Training args
    parser.add_argument("--max_epochs", type=int, default=3)     
    parser.add_argument("--nbits", type=int, default=2048)   
    parser.add_argument("--num_per_class", type=int, default=100)        
    parser.add_argument("--batch_size", type=int, default=32)    
    parser.add_argument("--fp_radii", type=int, default=2)    
    parser.add_argument("--hidden_dim", type=int, default=512) 
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=-1)
    # Analysis args
    parser.add_argument("--top_k", type=int, help="Number of skeleton classes to visualize", default=4)
    parser.add_argument("--vis_class_criteria", help="Which skeletons to visualize", choices=['popularity', 'size_small', 'size_large'])
    parser.add_argument("--max_vis_per_class", type=int, default=30)
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


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)  
    same = False
    config_path = os.path.join(args.work_dir, 'config.json')
    if os.path.exists(config_path):
        same = True
        args_dict = json.load(open(config_path, 'r'))
        for k in args_dict:
            if k == 'work_dir':
                continue
            if hasattr(args, k) and args_dict[k] != getattr(args, k):
                same = False 
    
    all_skeletons = pickle.load(open(args.skeleton_file, 'rb'))          
    nbits = args.nbits    
    num_per_class = args.num_per_class
    os.makedirs(os.path.join(args.work_dir, 'sks/'), exist_ok=True)
    skeletons = {}    
    for index, k in enumerate(all_skeletons):
        if args.datasets and index not in args.datasets:
            continue
        if len(all_skeletons[k]) >= num_per_class:
            sk = Skeleton(k, index=index)
            ind = len(skeletons)
            path = os.path.join(os.path.join(args.work_dir, 'sks/'), f'{index}_{ind}.png')
            sk.visualize(path=path)
            skeletons[k] = all_skeletons[k]

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
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)
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

    if args.ckpt:
        mlp = load_mlp_from_ckpt(args.ckpt)
    else:
        mlp = MLP(
            input_dim=args.nbits,
            output_dim=args.num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task="classification",
            loss="cross_entropy",
            valid_loss="multi_class_accuracy",
            optimizer="adam",
            learning_rate=1e-4,
            val_freq=1,
            ncpu=args.num_workers
        )      
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)    
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)    
        
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

    # Analysis
    if args.vis_class_criteria == 'popularity':
        vis_class_criteria = lambda i:len(all_skeletons[list(all_skeletons)[i]])
    elif args.vis_class_criteria == 'size_small':
        vis_class_criteria = lambda i:-(len(list(all_skeletons)[i].chemicals)+len(list(all_skeletons)[i].reactions))
    elif args.vis_class_criteria == 'size_large':
        vis_class_criteria = lambda i:len(list(all_skeletons)[i].chemicals)+len(list(all_skeletons)[i].reactions)
    else:
        raise NotImplementedError
    mlp.eval()
    cmap = plt.get_cmap('tab10')
    top_indices = sorted(range(len(all_skeletons)), key=vis_class_criteria)[-args.top_k:]
    val_feats = []
    val_indices = []
    val_max_count = {}
    for X_val, y_val in valid_dataset:
        if y_val.sum().item() > 1:
            continue
        feats = mlp(X_val[None], return_hidden=True)
        ind = y_val.argmax().item()
        if ind not in top_indices:
            continue
        if val_max_count.get(ind, 0) >= 30:
            continue
        val_max_count[ind] = val_max_count.get(ind, 0)+1
        val_indices.append(ind)
        val_feats.append(feats)
    train_feats = []
    train_indices = []
    train_max_count = {}        
    for X_train, y_train in train_dataset:
        if y_train.sum().item() > 1:
            continue
        feats = mlp(X_train[None], return_hidden=True)
        ind = y_train.argmax().item()
        if ind not in top_indices:
            continue
        if train_max_count.get(ind, 0) >= args.max_vis_per_class:
            continue
        train_max_count[ind] = train_max_count.get(ind, 0)+1
        train_indices.append(ind)
        train_feats.append(feats)        
    val_feats = torch.cat(val_feats, dim=0)
    val_feats = val_feats.detach().cpu().numpy()
    train_feats = torch.cat(train_feats, dim=0)
    train_feats = train_feats.detach().cpu().numpy()    
    tsne = TSNE(n_components=2, verbose=1, random_state=0, init='pca')    
    X_val = tsne.fit_transform(val_feats)   
    X_train = tsne.fit_transform(train_feats)   
    fig = plt.Figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.set_title("Hidden feats (val set)")
    for ind in top_indices:
        mask = np.array(val_indices) == ind
        popularity = len(top_indices)-top_indices.index(ind)
        ax.scatter(X_val[mask,0],X_val[mask,1],c=cmap(popularity-1),label=f"{args.vis_class_criteria}={popularity}")
    ax.legend(title="Skeleton class")
    ax = fig.add_subplot(1,2,2)
    ax.set_title("Hidden feats (train set)")
    for ind in top_indices:
        mask = np.array(train_indices) == ind
        popularity = len(top_indices)-top_indices.index(ind)
        ax.scatter(X_train[mask,0],X_train[mask,1],c=cmap(popularity-1),label=f"{args.vis_class_criteria}={popularity}")
    ax.legend(title="Skeleton class")    
    fig.savefig(os.path.join(args.work_dir, 'tsne.png'))
        


if __name__ == "__main__":
    args = get_args()
    breakpoint()
    main(args)
