"""
Multi-layer perceptron (MLP) class.
"""
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric.nn as geom_nn
from collections import defaultdict

# from synnet.MolEmbedder import MolEmbedder
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


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
        # molembedder: MolEmbedder = None,
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
        self.molembedder = molembedder
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

    def forward(self, x):
        """Forward step for inference only."""
        y_hat = self.layers(x)
        if (
            self.hparams.task == "classification"
        ):  # during training, `cross_entropy` loss expects raw logits
            y_hat = F.softmax(y_hat, dim=-1)
        return y_hat

    def training_step(self, batch, batch_idx):
        """The complete training loop."""
        x, y = batch
        y_hat = self.layers(x)
        if self.loss == "cross_entropy":
            try:
                loss = F.cross_entropy(y_hat, y.long())
            except:
                breakpoint()
        elif self.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.loss == "l1":
            loss = F.l1_loss(y_hat, y)
        elif self.loss == "huber":
            loss = F.huber_loss(y_hat, y)
        else:
            raise ValueError("Unsupported loss function '%s'" % self.loss)
        self.log(f"train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """The complete validation loop."""
        if self.trainer.current_epoch % self.val_freq != 0:
            return None
        x, y = batch
        y_hat = self.layers(x)
        if self.valid_loss == "cross_entropy":
            loss = F.cross_entropy(y_hat, y.long())
        elif self.valid_loss == "accuracy":
            y_hat = torch.argmax(y_hat, axis=1)
            accuracy = (y_hat == y).sum() / len(y)
            loss = 1 - accuracy
        elif self.valid_loss[:11] == "nn_accuracy":
            # NOTE: Very slow!
            # Performing the knn-search can easily take a couple of minutes,
            # even for small datasets.
            y = nn_search_list(y.detach().cpu(), self.X)
            y_hat = nn_search_list(y_hat.detach().cpu(), self.X)
            accuracy = (y_hat == y).sum() / len(y)
            loss = 1 - accuracy
        elif self.valid_loss == "faiss-knn":
            index = self.molembedder.index
            device = index.getDevice() if hasattr(index, "getDevice") else "cpu"
            import faiss.contrib.torch_utils  # https://github.com/facebookresearch/faiss/issues/561

            # Normalize query vectors
            y_normalized = y / torch.linalg.norm(y, dim=1, keepdim=True)
            ypred_normalized = y_hat / torch.linalg.norm(y_hat, dim=1, keepdim=True)
            # kNN search
            k = 1
            _, ind_y = index.search(y_normalized.to(device), k)
            _, ind_ypred = index.search(ypred_normalized.to(device), k)
            accuracy = (ind_y == ind_ypred).sum() / len(y)
            loss = 1 - accuracy
        elif self.valid_loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.valid_loss == "l1":
            loss = F.l1_loss(y_hat, y)
        elif self.valid_loss == "huber":
            loss = F.huber_loss(y_hat, y)
        else:
            raise ValueError("Unsupported loss function '%s'" % self.valid_loss)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """Define Optimerzers and LR schedulers."""
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs,
    ):
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer_by_name = {"GCN": geom_nn.GCNConv, 
                             "GAT": geom_nn.GATConv, 
                             "GraphConv": geom_nn.GraphConv,
                             "Transformer": geom_nn.TransformerConv,
                             "GIN": geom_nn.GINConv}
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, data):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        x = data.x
        edge_index = data.edge_index
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class GNN(pl.LightningModule):
    def __init__(
        self,     
        task: str = "classification",
        loss: str = "cross_entropy",
        valid_loss: str = "accuracy",
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        val_freq: int = 10,
        ncpu: int = 16,
        # molembedder: MolEmbedder = None,
        X = None,
        datasets = "",
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore="molembedder")
        self.loss = loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ncpu = ncpu
        self.val_freq = val_freq
        # self.molembedder = molembedder
        self.X = torch.Tensor(np.load(X)) if (X is not None) else molembedder.embeddings
        self.valid_steps_y = []
        self.valid_steps_y_hat = []
        self.model = GNNModel(**model_kwargs)

    def forward(self, data, mask):
        """Forward step for inference only."""
        x, edge_index = data.x, data.edge_index
        y_hat = self.model(x, edge_index)
        if (
            self.hparams.task == "classification"
        ):  # during training, `cross_entropy` loss expects raw logits
            y_hat = F.softmax(y_hat, dim=-1)
        return y_hat

    def training_step(self, batch, batch_idx):
        """The complete training loop."""
        data = batch
        y_hat = self.model(data)
        y = data.y

        # only building blocks
        mask_bb = (y[:, :256] != 0).any(axis=-1)
        y_bb = y[mask_bb, :256]
        y_hat_bb = y_hat[mask_bb, :256]

        # only reactions
        mask_rxn = (y[:, 256:] != 0).any(axis=-1)
        y_rxn = y[mask_rxn, 256:]        
        y_hat_rxn = y_hat[mask_rxn, 256:]

        loss = torch.Tensor([0.]).to(y_rxn.device)
        if "cross_entropy" in self.loss:
            if y_rxn.shape[0]:
                ce_loss = F.cross_entropy(y_hat_rxn, y_rxn)   
                self.log(f"train_ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                loss += ce_loss               
        if "mse" in self.loss:
            if y_bb.shape[0]:
                mse_loss = F.mse_loss(y_hat_bb, y_bb)       
                self.log(f"train_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                loss += mse_loss 

        self.log(f"train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """The complete validation loop."""
        if self.trainer.current_epoch % self.val_freq != 0:
            return None
        data = batch
        y = data.y
        data_key = np.array([k for data_key in data.key for k in data_key])
        y_hat = self.model(data)
        # only building blocks
        mask_bb = (y[:, :256] != 0).any(axis=-1)
        y_bb = y[mask_bb, :256]
        y_hat_bb = y_hat[mask_bb, :256]
        # only reactions
        mask_rxn = (y[:, 256:] != 0).any(axis=-1)        
        y_rxn = y[mask_rxn, 256:]        
        y_hat_rxn = y_hat[mask_rxn, 256:]

        
        # turn into numpy for logging
        mask_rxn = mask_rxn.cpu().numpy()
        mask_bb = mask_bb.cpu().numpy()

        if "cross_entropy" in self.valid_loss:
            if y_rxn.shape[0]:
                ce_loss = F.cross_entropy(y_hat_rxn, y_rxn)
                self.log("val_cross_entropy_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                batch_ce_loss = F.cross_entropy(y_hat_rxn, y_rxn, reduce=False)
                loss_by_key = defaultdict(list)
                for key, loss in zip(data_key[mask_rxn], batch_ce_loss):
                    loss_by_key[key].append(loss.item())
                for k, v in loss_by_key.items():
                    self.log(f"val_cross_entropy_loss_{k}", np.mean(v), batch_size=len(v), on_step=False, on_epoch=True, prog_bar=True, logger=True)                    
                                    
        if "accuracy" in self.valid_loss:
            if y_rxn.shape[0]:
                breakpoint()
                y_hat, y = y_hat_rxn, y_rxn
                y_hat = torch.argmax(y_hat, axis=1)
                y = torch.argmax(y, axis=1)
                accuracy = (y_hat == y).sum() / len(y)
                acc_loss = (1 - accuracy)
                self.log("val_accuracy_loss", acc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                acc_by_key = defaultdict(list)
                for key, correct in zip(data_key[mask_rxn], (y_hat == y)):
                    acc_by_key[key].append(correct.item())
                for k, v in acc_by_key.items():
                    self.log(f"val_accuracy_{k}", np.mean(v), batch_size=len(v), on_step=False, on_epoch=True, prog_bar=True, logger=True)                    
                             
                
        if "nn_accuracy" in self.valid_loss:
            if y_bb.shape[0]:
                y = nn_search_list(y_bb, torch.as_tensor(self.X, dtype=torch.float32).to(y.device))
                y_hat = nn_search_list(y_hat_bb, torch.as_tensor(self.X, dtype=torch.float32).to(y.device))            
                accuracy = (y_hat == y).sum() / len(y)
                nn_acc_loss = (1 - accuracy)
                self.log("val_nn_accuracy_loss", nn_acc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                acc_by_key = defaultdict(list)
                for key, correct in zip(data_key[mask_bb], (y_hat == y)):
                    acc_by_key[key].append(correct.item())
                for k, v in acc_by_key.items():
                    self.log(f"val_nn_accuracy_{k}", np.mean(v), batch_size=len(v), on_step=False, on_epoch=True, prog_bar=True, logger=True)                    
                                      
        if "faiss-knn" in self.valid_loss:
            index = self.molembedder.index
            device = index.getDevice() if hasattr(index, "getDevice") else "cpu"
            import faiss.contrib.torch_utils  # https://github.com/facebookresearch/faiss/issues/561

            # Normalize query vectors
            y_normalized = y / torch.linalg.norm(y, dim=1, keepdim=True)
            ypred_normalized = y_hat / torch.linalg.norm(y_hat, dim=1, keepdim=True)
            # kNN search
            k = 1
            _, ind_y = index.search(y_normalized.to(device), k)
            _, ind_ypred = index.search(ypred_normalized.to(device), k)
            accuracy = (ind_y == ind_ypred).sum() / len(y)
            knn_loss = (1 - accuracy)
            self.log("val_knn_loss", knn_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        
        # self.valid_steps_y.append(y)
        # self.valid_steps_y_hat.append(y_hat)
    

    
    # def on_validation_epoch_end(self):
    #     # outs is a list of whatever you returned in `validation_step`
    #     y = torch.cat(self.valid_steps_y, axis=0)
    #     y_hat = torch.cat(self.valid_steps_y_hat, axis=0)
    #     self.valid_steps_y = []
    #     self.valid_steps_y_hat = []
    #     index = self.molembedder.index
    #     device = index.getDevice() if hasattr(index, "getDevice") else "cpu"
    #     import faiss.contrib.torch_utils  # https://github.com/facebookresearch/faiss/issues/561
    #     # Normalize query vectors
    #     y_normalized = y / torch.linalg.norm(y, dim=1, keepdim=True)
    #     ypred_normalized = y_hat / torch.linalg.norm(y_hat, dim=1, keepdim=True)
    #     # kNN search
    #     k = 1
    #     _, ind_y = index.search(y_normalized.to(device), k)
    #     _, ind_ypred = index.search(ypred_normalized.to(device), k)
    #     ind_y = ind_y.flatten().detach().cpu().numpy()
    #     ind_ypred = ind_ypred.flatten().detach().cpu().numpy()
    #     accuracy = ((ind_y == ind_ypred).sum() + 0.0) / len(ind_ypred)
    #     loss = 1 - accuracy     
    #     print(ind_y)
    #     print(ind_ypred)
    #     print(f"accuracy: {accuracy}")        
    #     if loss.item() > 1:
    #         breakpoint()
    #     self.log("val_loss", loss)


    def configure_optimizers(self):
        """Define Optimerzers and LR schedulers."""
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


def nn_search_list(y, X, top_k=1):
    def cosine_neighbors(x, y):
        y_T = y.T
        norm = torch.norm(x, dim=-1)[None]        
        if len(norm.shape) == 2:
            norm_T = norm.T
        else:
            norm_T = norm
        sims = ((x @ y_T)/(norm_T @ torch.norm(y, dim=-1)[None]))
        return torch.kthvalue(sims, sims.shape[-1]+1-top_k, axis=-1, keepdims=True).indices
    # ind_2 = molembedder.kdtree.query(y, k=1, return_distance=False)  # (n_samples, 1)
    ind_3 = cosine_neighbors(y, X)
    # assert (ind_2 == ind_3).all()
    return ind_3


if __name__ == "__main__":
    pass
