# %%

import torch
from torch import nn
from torch_geometric.nn import GraphNorm


# %%
class InteractionBlock(nn.Module):

    def __init__(self, edge_dim, node_dim):
        super().__init__()

        self.filter_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, node_dim),
            nn.SiLU(),
        )

        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
        )

        self.graph_norm = GraphNorm(node_dim)

    def forward(self, H_curr, E, edge_index, batch_indices=None):

        filter_inputs = torch.concat(
            [E, H_curr[edge_index[0]], H_curr[edge_index[1]]], dim=1
        )
        filters = self.filter_mlp(filter_inputs)
        M = torch.zeros_like(H_curr)
        # on additionne les messages le long des lignes et dans notre convention receveur = edge_index[0]
        M.index_add_(dim=0, index=edge_index[0], source=(H_curr[edge_index[1]] * filters))
        return H_curr + self.message_mlp(self.graph_norm(M, batch=batch_indices))