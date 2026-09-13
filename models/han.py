from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv


class HANNodeClassifier(nn.Module):
    def __init__(
        self,
        data,
        target_node_type,
        hidden_dim,
        out_dim,
        heads,
        dropout,
    ):
        super().__init__()

        self.target_node_type = target_node_type
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.metadata = data.metadata()

        self.input_proj = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()

        in_channels_dict = {}

        for ntype in data.node_types:
            has_x = hasattr(data[ntype], "x") and data[ntype].x is not None

            if has_x:
                x = data[ntype].x
                if x.dim() != 2:
                    raise ValueError(f"Node type '{ntype}' has invalid x shape: {tuple(x.shape)}")
                in_dim = x.size(-1)
                self.input_proj[ntype] = nn.Linear(in_dim, hidden_dim)
            else:
                num_nodes = data[ntype].num_nodes
                if num_nodes is None:
                    raise ValueError(f"Node type '{ntype}' has no x and no num_nodes.")
                self.embeddings[ntype] = nn.Embedding(num_nodes, hidden_dim)

            in_channels_dict[ntype] = hidden_dim

        self.conv1 = HANConv(
            in_channels=in_channels_dict,
            out_channels=hidden_dim,
            metadata=self.metadata,
            heads=heads,
            dropout=dropout,
        )

        self.conv2 = HANConv(
            in_channels={ntype: hidden_dim for ntype in data.node_types},
            out_channels=hidden_dim,
            metadata=self.metadata,
            heads=heads,
            dropout=dropout,
        )

        self.classifier = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for _, mod in self.input_proj.items():
            mod.reset_parameters()

        for _, emb in self.embeddings.items():
            nn.init.xavier_uniform_(emb.weight)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.classifier.reset_parameters()

    def _build_input_x_dict(self, data, device):
        x_dict = {}

        for ntype in data.node_types:
            has_x = hasattr(data[ntype], "x") and data[ntype].x is not None

            if has_x:
                x = data[ntype].x.to(device)
                x_dict[ntype] = self.input_proj[ntype](x)
            else:
                if hasattr(data[ntype], "global_nids") and data[ntype].global_nids is not None:
                    idx = data[ntype].global_nids.to(device)
                else:
                    num_nodes = data[ntype].num_nodes
                    idx = torch.arange(num_nodes, device=device)

                x_dict[ntype] = self.embeddings[ntype](idx)

        return x_dict

    def forward(
        self,
        data,
        return_embeddings,
    ):
        device = next(self.parameters()).device
        x_dict = self._build_input_x_dict(data, device)
        edge_index_dict = {k: v.edge_index.to(device) for k, v in data.edge_items()}

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        logits = self.classifier(x_dict[self.target_node_type])

        if return_embeddings:
            return logits, x_dict
        return logits