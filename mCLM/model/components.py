import torch
from torch import nn
from typing import Callable, Any, Union, List, Tuple, Dict, Optional
import hydra
import lightning as L
import numpy as np
import torch.nn.functional as F
import torch_geometric
import torchmetrics
import warnings
from torch_geometric.nn import (
    global_add_pool,
    GINEConv,
    JumpingKnowledge,
)
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    """
    A simple MLP
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: List[int] = (1024, 1024),
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = output_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], self.latent_dim))

    def forward(self, x) -> torch.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def forward_embedding(self, x) -> torch.Tensor:
        for i, layer in enumerate(self.network[:-1]):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class GNNMolEncoder(nn.Module):
    """Encoder class for molecule data"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim_graph: int,
        hidden_dim_ffn: int,
        num_mp_layers: int,
        num_readout_layers: int,
        out_channels: int,
        dropout: float,
        aggr: str = "mean",
        jk: str = "cat",
        mol_features_size: int = 0,
    ):
        """
        Args:
            node_dim (int): Input dimension for molecule node feature.
            edge_dim (int): Input dimension for molecule edge feature.
            hidden_dim_graph (int): Hidden dimension for graph.
            hidden_dim_ffn (int): Hidden dimension for classifier.
            num_mp_layers (int): Number of layers for neural network h for GINEConv.
            num_readout_layers (int): Number of layers for neural network classifier.
            out_channels (int): Size of output channels for classifier.
            dropout (float): Dropout ratio for neural network h and classifier.
            aggr (str, optional): How is graph pooling done. Defaults to "mean".
            jk (str, optional): How is jumping knowledge mode to use. Defaults to "cat".
            mol_features_size (int, optional): Size of molecular features. Defaults to 0.

        """
        super().__init__()

        # graph encoder
        self.node_encoder = nn.Linear(node_dim, hidden_dim_graph)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim_graph)
        self.out_channels = out_channels

        self.convs = nn.ModuleList()
        for _ in range(num_mp_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim_graph, 2 * hidden_dim_graph),
                nn.BatchNorm1d(2 * hidden_dim_graph),
                nn.ReLU(inplace=True),
                nn.Linear(2 * hidden_dim_graph, hidden_dim_graph),
                nn.BatchNorm1d(hidden_dim_graph),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            conv = GINEConv(mlp, train_eps=True)
            self.convs.append(conv)

        self.jk_mode = jk
        if self.jk_mode == "none":
            self.jk = None
        else:
            self.jk = JumpingKnowledge(
                mode=self.jk_mode, channels=hidden_dim_graph, num_layers=num_mp_layers
            )

        # global pooling
        self.aggr = aggr
        if aggr == "mean":
            self.global_pool = global_mean_pool
        elif aggr == "sum":
            self.global_pool = global_add_pool

        # classifier
        self.classifier = nn.ModuleList()

        if self.jk_mode == "none":
            hidden_channels_mol = hidden_dim_graph
        elif self.jk_mode == "cat":
            hidden_channels_mol = hidden_dim_graph * (num_mp_layers + 1)
        else:
            raise NotImplementedError

        ffn_hidden_size = (
            int(hidden_dim_ffn)
            if hidden_dim_ffn is not None
            else int(hidden_channels_mol)
        )

        for layer in range(num_readout_layers):
            input_dim = hidden_channels_mol if layer == 0 else ffn_hidden_size
            mlp = nn.Sequential(
                nn.Linear(input_dim, ffn_hidden_size),
                nn.BatchNorm1d(ffn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            self.classifier.append(mlp)

        # last layer (classifier)
        input_dim_classifier = (
            hidden_channels_mol if num_readout_layers == 0 else ffn_hidden_size
        )
        self.classifier.append(
            nn.Linear(input_dim_classifier, self.out_channels),
        )

    def compute_message_passing(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        list_graph_encodings = []

        x_encoded = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        if self.jk_mode != "none":
            list_graph_encodings.append(x_encoded)
        for conv in self.convs:
            x_encoded = conv(x_encoded, edge_index, edge_attr)
            if self.jk_mode != "none":
                list_graph_encodings.append(x_encoded)

        if self.jk_mode != "none":
            x_encoded = self.jk(list_graph_encodings)

        out = self.global_pool(x_encoded, batch)  # [batch_size, hidden_channels]
        return out

    def compute_readout(
        self, graph_repr: torch.Tensor, restrict_output_layers: int = 0
    ) -> torch.Tensor:
        out = graph_repr
        for mlp in self.classifier[
            : None if restrict_output_layers == 0 else restrict_output_layers
        ]:
            out = mlp(out)
        return out

    def forward(
        self, data: torch_geometric.data.Data, restrict_output_layers: int = 0
    ) -> torch.Tensor:
        graph_repr = self.compute_message_passing(data)
        out = self.compute_readout(graph_repr, restrict_output_layers)
        return out

