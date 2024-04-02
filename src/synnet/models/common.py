"""Common methods and params shared by all models.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from scipy import sparse

from synnet.models.mlp import MLP, GNN


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/featurized/Xy", help="Directory with X,y data.")
    parser.add_argument("--results-log", default="results/logs/")
    parser.add_argument("--skeleton-dir", type=str, help="Directory with X,y data.")    

    # for training gnn
    parser.add_argument(
        "--gnn-input-feats",
        type=str,
        help="Where to load featurized data to train GNN",
    )
    parser.add_argument("--pe", choices=['sin'], help='pos encoding')
    parser.add_argument(
        "--feats-split",
        action='store_true',
        help="add _{train|valid|test} suffix to gnn-input-feats",
    )
        
    parser.add_argument("--gnn-datasets", type=int, nargs='+')
    parser.add_argument("--lazy_load", action='store_true')
    parser.add_argument("-f", "--featurize", type=str, default="fp", help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default="hb", help="Choose from ['hb', 'pis']")
    parser.add_argument("--radius", type=int, default=2, help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096, help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--out_dim", type=int, default=256, help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=16, help="Number of cpus")
    parser.add_argument("--prefetch_factor", type=int, default=0, help="Number of batches to prefetch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epoch", type=int, default=2000, help="Maximum number of epoches.")
    parser.add_argument(
        "--ckpt-file",
        type=str,
        default=None,
        help="Checkpoint file. If provided, load and resume training.",
    )
    parser.add_argument(
        "--mol-embedder-file",
        type=str,
        help="Load mol embedder with this.",
    )    
    parser.add_argument("-v", "--version", type=int, default=1, help="Version")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--fast-dev-run", default=False, action="store_true")

    # training args
    parser.add_argument('--lr',type=float, default=3e-4)
    parser.add_argument('--cuda', type=int, default=-1)

    # gnn args
    parser.add_argument('--gnn-layer', default='GCN')
    parser.add_argument('--gnn-dp-rate', default=0.5, type=float)
    parser.add_argument('--gnn-valid-loss', default='nn_accuracy')
    parser.add_argument('--gnn-loss', default='mse')


    return parser.parse_args()


def xy_to_dataset(
    X_file: str, y_file: str, task: str = "regression", n: Union[int, float] = 1.0, **kwargs
):
    """Loads featurized X,y `*.npz`-data into a `DataLoader`"""
    X = sparse.load_npz(X_file)
    y = sparse.load_npz(y_file)
    # Filer?
    if isinstance(n, int):
        n = min(n, X.shape[0])  # ensure n does not exceed size of dataset
        X = X[:n]
        y = y[:n]
    elif isinstance(n, float) and n < 1.0:
        xn = X.shape[0] * n
        yn = X.shape[0] * n
        X = X[:xn]
        y = y[:yn]
    else:
        pass  #
    X = np.atleast_2d(np.asarray(X.todense()))
    y = (
        np.atleast_2d(np.asarray(y.todense()))
        if task == "regression"
        else np.asarray(y.todense()).squeeze()
    )
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X),
        torch.Tensor(y),
    )
    return dataset    


def xy_to_dataloader(
    X_file: str, y_file: str, task: str = "regression", n: Union[int, float] = 1.0, **kwargs
):
    dataset = xy_to_dataset(X_file, y_file, task, n, **kwargs)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def load_mlp_from_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference."""
    try:
        model = MLP.load_from_checkpoint(ckpt_file, map_location='cpu')
    except TypeError:
        model = _load_mlp_from_iclr_ckpt(ckpt_file)
    return model.eval()


def load_gnn_from_ckpt(ckpt_file: str):
    """Load a gnn from a checkpoint"""
    model = GNN.load_from_checkpoint(ckpt_file, map_location='cpu').model
    return model


def find_best_model_ckpt(path: str, version=None, key="val_loss") -> Union[Path, None]:
    """Find checkpoint with lowest val_loss.

    Poor man's regex:
    somepath/act/ckpts.epoch=70-val_loss=0.03.ckpt
                                         ^^^^--extract this as float
    """
    ckpts = Path(path).rglob("*.ckpt")
    best_model_ckpt = None
    lowest_loss = 10_000  # ~ math.inf    
    for file in ckpts:
        if version is not None:
            if str(file).find(f"version_{version}") != -1:
                return file
        else:
            stem = file.stem
            val_loss = float(stem.split(f"{key}=")[-1])
            if val_loss < lowest_loss:
                best_model_ckpt = file
                lowest_loss = val_loss
    if version is None:
        raise AssertionError(f"{path} {version} has no ckpt")
    return best_model_ckpt


def _load_mlp_from_iclr_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference.
    Info: hparams were not saved, so we specify the ones needed for inference again."""
    model = Path(ckpt_file).parent.name  # assume "<dirs>/<model>/<file>.ckpt"
    if model == "act":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=3 * 4096,
            output_dim=4,
            hidden_dim=1000,
            num_layers=5,
            task="classification",
            dropout=0.5,
        )
    elif model == "rt1":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=3 * 4096,
            output_dim=256,
            hidden_dim=1200,
            num_layers=5,
            task="regression",
            dropout=0.5,
        )
    elif model == "rxn":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=4 * 4096,
            output_dim=91,
            hidden_dim=3000,
            num_layers=5,
            task="classification",
            dropout=0.5,
        )
    elif model == "rt2":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=4 * 4096 + 91,
            output_dim=256,
            hidden_dim=3000,
            num_layers=5,
            task="regression",
            dropout=0.5,
        )

    else:
        raise ValueError
    return model.eval()


if __name__ == "__main__":
    import json

    args = get_args()
    print("Default Arguments are:")
    print(json.dumps(args.__dict__, indent=2))
