from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from synnet.models.mlp import MLP
from synnet.MolEmbedder import MolEmbedder


def load_mlp_from_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference."""
    try:
        model = MLP.load_from_checkpoint(ckpt_file, map_location='cpu')
    except TypeError:
        model = _load_mlp_from_iclr_ckpt(ckpt_file)
    return model.eval()


def find_best_model_ckpt(path: str, version=None, key="val_loss") -> Union[Path, None]:
    """Find checkpoint with lowest val_loss.

    Poor man's regex:
    somepath/act/ckpts.epoch=70-val_loss=0.03.ckpt
                                         ^^^^--extract this as float
    """
    ckpts = Path(path).rglob("*.ckpt")
    best_model_ckpt = None
    lowest_loss = 10_000  # ~ math.inf
    ckpts = list(ckpts)
    for file in ckpts:
        if version is not None and f"version_{version}" not in str(file.parent):
            continue
        stem = file.stem
        try:
            val_loss = float(stem.split(f"{key}=")[-1])
        except:
            assert best_model_ckpt is None
            best_model_ckpt = file
            continue
        if val_loss < lowest_loss:
            best_model_ckpt = file
            lowest_loss = val_loss
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
            map_location='cpu'
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
            map_location='cpu'
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
            map_location='cpu'
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
            map_location='cpu'
        )

    else:
        raise ValueError
    return model.eval()


class MLP(pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_dropout_layers: int = 1,
        task: str = "classification",
        loss: str = "cross_entropy",
        valid_loss: str = "accuracy",
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        val_freq: int = 10,
        ncpu: int = 16,
        molembedder: MolEmbedder = None,
        X = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="molembedder")
        self.loss = loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ncpu = ncpu
        self.val_freq = val_freq
        self.X = torch.Tensor(np.load(X)) if (X is not None) else (molembedder.embeddings if molembedder is not None else None)
        # self.nn = NearestNeighbors(n_neighbors=1)
        # self.nn.fit(molembedder.embeddings)
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.ReLU())

        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.BatchNorm1d(hidden_dim))
            modules.append(nn.ReLU())
            if i > num_layers - 3 - num_dropout_layers:
                modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*modules)

    def forward(self, x, return_hidden=False):
        """Forward step for inference only."""
        y_hat = self.layers(x)
        if hasattr(self, 'final_layer'):
            if return_hidden:
                return y_hat
            else:
                y_hat = self.final_layer(y_hat)
        if (
            self.hparams.task == "classification"
        ):  # during training, `cross_entropy` loss expects raw logits
            y_hat = F.softmax(y_hat, dim=-1)
        return y_hat
