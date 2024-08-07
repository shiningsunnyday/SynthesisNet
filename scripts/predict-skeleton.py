from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
import numpy as np
from tqdm import tqdm
from synnet.data_generation.preprocessing import BuildingBlockFileHandler, ReactionTemplateFileHandler
from synnet.visualize.drawers import MolDrawer, RxnDrawer
from synnet.visualize.visualizer import SkeletonVisualizer
from synnet.visualize.writers import SynTreeWriter, SkeletonPrefixWriter
from synnet.data_generation.syntrees import MorganFingerprintEncoder
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.models.mlp import MLP, nn_search_list
from synnet.utils.data_utils import Skeleton
import torch.nn as nn
import time
import logging
import json
import rdkit.Chem as Chem
from sklearn.neighbors import KNeighborsClassifier
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


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


from scipy.spatial.distance import cosine


def nearest_neighbors(fps, fp, top_k=5):    
    fp = torch.tensor(fp)
    inds = []
    dists = []
    for k in range(5):
        ind = nn_search_list(fp, fps, k+1)
        ind = ind.item()
        inds.append(ind)    
        dist = cosine(fps[ind], fp)
        dists.append(dist)
    return dists, inds



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
            if k in ['work_dir', 'ckpt', 'top_k', 'vis_class_criteria', 'max_vis_per_class']:
                continue
            if hasattr(args, k) and args_dict[k] != getattr(args, k):                
                same = False 
    
    all_skeletons = pickle.load(open(args.skeleton_file, 'rb'))          
    nbits = args.nbits    
    num_per_class = args.num_per_class
    os.makedirs(os.path.join(args.work_dir, 'sks/'), exist_ok=True)
    skeletons = {}    
    datasets = []
    for index, k in enumerate(all_skeletons):
        if args.datasets and index not in args.datasets:
            continue
        if len(all_skeletons[k]) >= num_per_class:
            sk = Skeleton(k, index=index)
            datasets.append(index)
            ind = len(skeletons)
            path = os.path.join(os.path.join(args.work_dir, 'sks/'), f'{index}_{ind}.png')
            sk.visualize(path=path)
            skeletons[k] = all_skeletons[k]
    setattr(args, 'datasets', datasets)
    print(f"{len(skeletons)} classes with >= {num_per_class} per class")
    num_classes = len(skeletons)
    setattr(args, 'num_classes', num_classes)    
    encoder = MorganFingerprintEncoder(args.fp_radii, nbits)
    lookup = {}
    for i, k in enumerate(skeletons):            
        for j, st in enumerate(skeletons[k]):                
            try: 
                feats = encoder.encode(st.root.smiles).flatten()                    
            except: 
                continue
            if st.root.smiles not in lookup:
                lookup[st.root.smiles] = {'labels': np.zeros((len(skeletons),)), 
                                          'features': feats,
                                          'st': st,
                                          'index': i}
            lookup[st.root.smiles]['labels'][i] = 1
    features, labels = [], []
    smiles, sts = [], []
    indices = []
    fps = []
    for smi in lookup:
        features.append(lookup[smi]['features'])
        label = lookup[smi]['labels']
        st = lookup[smi]['st']
        fp = encoder.encode(smi).flatten()
        index = lookup[smi]['index']
        label = label/label.sum()
        labels.append(label)
        smiles.append(smi)
        sts.append(st)
        indices.append(index)
        fps.append(fp)
    fps = torch.tensor(fps)
    if same:
        train_dataset = pickle.load(open(os.path.join(args.work_dir, 'train_dataset.pkl'), 'rb'))
        valid_dataset = pickle.load(open(os.path.join(args.work_dir, 'valid_dataset.pkl'), 'rb'))
        # train_smiles = pickle.load(open(os.path.join(args.work_dir, 'train_smiles.pkl'), 'rb'))
        # valid_smiles = pickle.load(open(os.path.join(args.work_dir, 'valid_smiles.pkl'), 'rb'))
    else:          

        # X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33, stratify=labels)
        
        X_train, X_valid, y_train, y_valid, smiles_train, smiles_valid = train_test_split(features, labels, smiles, test_size=0.33, random_state=42)
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
        pickle.dump(smiles_train, open(os.path.join(args.work_dir, 'train_smiles.pkl'), 'wb+'))
        pickle.dump(smiles_valid, open(os.path.join(args.work_dir, 'valid_smiles.pkl'), 'wb+'))
        json.dump(args.__dict__, open(config_path, 'w+'))

    if args.ckpt:
        mlp = load_mlp_from_ckpt(args.ckpt)
        knn = pickle.load(open(os.path.join(args.work_dir, 'knn.pkl'), 'rb'))        
        # Analysis
        if args.vis_class_criteria == 'popularity':
            vis_class_criteria = lambda i:len(all_skeletons[list(all_skeletons)[i]])
        elif args.vis_class_criteria == 'size_large':
            vis_class_criteria = lambda i:len(list(all_skeletons)[i].chemicals)+len(list(all_skeletons)[i].reactions)        
        else:
        # elif args.vis_class_criteria == 'size_small':
            vis_class_criteria = lambda i:-(len(list(all_skeletons)[i].chemicals)+len(list(all_skeletons)[i].reactions))            
        mlp.eval()
        cmap = plt.get_cmap('tab10')
        top_indices = sorted(range(len(all_skeletons)), key=vis_class_criteria)[-args.top_k:]

        skeleton_path = 'results/viz/skeletons-valid.pkl'
        valid_skeletons = pickle.load(open(skeleton_path, 'rb'))         
        valid_conf = {}
        for ind in range(len(valid_skeletons)):
            if ind not in args.datasets:
                continue
            st = list(valid_skeletons)[ind]
            index = args.datasets.index(ind)
            for st in tqdm(valid_skeletons[st][:10]):
                smi = st.root.smiles
                fp = encoder.encode(smi)
                pred = mlp(torch.Tensor(fp))                 
                # probs = knn.predict_proba(fp)          
                # only indices same as fp
                mask = np.array(indices) == index
                masked_inds = np.argwhere(mask).flatten()
                fp = fp.flatten()
                dists, kneis = nearest_neighbors(fps[mask], fp)
                knei_inds = [masked_inds[knei] for knei in kneis]
                knei_inds = [(knei_ind, dist) for knei_ind, dist in zip(knei_inds, dists)]
                knei_inds = sorted(knei_inds, key=lambda x: x[1])
                knei_inds = [k[0] for k in knei_inds]
                prob = pred[..., index].item()                
                data = {'smi': smi, 'prob_mlp': prob, 'st': st, 'ind': ind, 
                        'kneis': knei_inds, 'dist_knn': dists}
                valid_conf[ind] = valid_conf.get(ind, []) + [data]
        
        rxns = ReactionTemplateFileHandler().load('data/assets/reaction-templates/hb.txt')
        skviz = lambda sk, *pargs: SkeletonVisualizer(sk, 'results/viz/', *pargs).with_drawings(mol_drawer=MolDrawer, rxn_drawer=RxnDrawer)
        for ind in valid_conf:
            valid_conf[ind] = sorted(valid_conf[ind], key=lambda x: x['prob_mlp'])
            version = None
            for k in range(1,min(6, len(valid_conf[ind]))):
                smi = valid_conf[ind][-k]['smi']
                prob = valid_conf[ind][-k]['prob_mlp']
                dist_knn = valid_conf[ind][-k]['dist_knn']                
                kneis = valid_conf[ind][-k]['kneis']               
                st = valid_conf[ind][-k]['st']
                sk = Skeleton(st, ind)
                sk.attribute_rxns(rxns)
                viz = skviz(sk, version)
                if version is None:
                    version = viz.version
                sk.mask = np.arange(len(sk.mask))
                mermaid_txt = viz.write(node_mask=sk.mask)
                mask_str = ''.join(map(str,sk.mask))
                outfile = viz.path / f"{sk.index}_{mask_str}_top={k}_prob={prob}.md" 
                title = f"{smi}_{prob}"
                SynTreeWriter(prefixer=SkeletonPrefixWriter(title)).write(mermaid_txt).to_file(outfile)
                for i in range(len(kneis)):
                    knei = kneis[i]
                    dist = dist_knn[i]
                    index = indices[knei]
                    sk = Skeleton(sts[knei], ind)
                    sk.attribute_rxns(rxns)
                    viz = skviz(sk, version)
                    sk.mask = np.arange(len(sk.mask))
                    mermaid_txt = viz.write(node_mask=sk.mask)
                    mask_str = ''.join(map(str,sk.mask))
                    outfile = viz.path / f"{sk.index}_{mask_str}_top={k}_dist={dist}_{i+1}th_neighbor.md" 
                    title = f"{smi}_{dist}_{i+1}'th neighbor"
                    SynTreeWriter(prefixer=SkeletonPrefixWriter(title)).write(mermaid_txt).to_file(outfile)                     


        val_feats = []
        val_indices = []
        val_max_count = {}
        conf = {}
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        for X_val, y_val in tqdm(valid_dataloader, "val dataset"):
            X_val = X_val[y_val.sum(axis=-1) == 1]
            y_val = y_val[y_val.sum(axis=-1) == 1]
            feats = mlp(X_val, return_hidden=True)
            inds = y_val.argmax(axis=-1)
            for i in range(inds.shape[0]):
                ind = inds[i].item()
                if ind not in top_indices:
                    continue
                if val_max_count.get(ind, 0) >= args.max_vis_per_class:
                    continue
                val_max_count[ind] = val_max_count.get(ind, 0)+1
                val_indices.append(ind)
                feat = feats[i]
                val_feats.append(feat)
                                    
        train_feats = []
        train_indices = []
        train_max_count = {}    
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)              
        for X_train, y_train in tqdm(train_dataloader, "train dataset"):
            X_train = X_train[y_train.sum(axis=-1) == 1]
            y_train = y_train[y_train.sum(axis=-1) == 1]            
            feats = mlp(X_train, return_hidden=True)
            inds = y_train.argmax(axis=-1)
            for i in range(inds.shape[0]):
                ind = inds[i].item()
                if ind not in top_indices:
                    continue
                if train_max_count.get(ind, 0) >= args.max_vis_per_class:
                    continue
                train_max_count[ind] = train_max_count.get(ind, 0)+1
                train_indices.append(ind)
                feat = feats[i]
                train_feats.append(feat)        
        val_feats = torch.stack(val_feats, dim=0)
        val_feats = val_feats.detach().cpu().numpy()
        train_feats = torch.stack(train_feats, dim=0)
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
        fig.savefig(os.path.join(args.work_dir, f'tsne-{args.top_k}.png'))
        print(os.path.abspath(os.path.join(args.work_dir, f'tsne-{args.top_k}.png')))

    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, [label.argmax() for label in labels])
        path = os.path.join(args.work_dir, 'knn.pkl')
        pickle.dump(open(path, 'wb+'))
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
        


if __name__ == "__main__":
    args = get_args()
    main(args)
