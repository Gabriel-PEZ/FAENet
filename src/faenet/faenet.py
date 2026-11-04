# %%

import torch 
from torch import nn
from torch_geometric.nn import global_mean_pool
from .embeddings import EdgeEmbedding, MoleculeEmbedding
from .interaction_block import InteractionBlock
from .output_block import OutputBlock
from .graph_creation import SFA, rho1_inv, adjacency_matrix

# %%

class FAENet(nn.Module):

    def __init__(
            self,
            atom_embedding_dim,
            hidden_dim_node, 
            hidden_dim_edge,
            hidden_dim_output_block,
            node_dim,
            edge_dim,
            atomic_numbers_to_indices,
            properties_atomes_scaled,
            nb_gaussians_rbf,
            cutoff,
            num_iteration_block,
            jump_connection,

        ):
        """
        Inputs: 
            - atom_embedding_dim = taille de l'embedding pour numéro atomique, groupe et période
        """

        super().__init__()
        self.cutoff = cutoff
        self.jump_connection = jump_connection

        self.molecule_embedding = MoleculeEmbedding(
            atom_embedding_dim=atom_embedding_dim,
            hidden_dim=hidden_dim_node,
            node_dim=node_dim,
            atomic_numbers_to_indices=atomic_numbers_to_indices,
            properties_atomes_scaled=properties_atomes_scaled
        )

        self.edge_embedding = EdgeEmbedding(
            hidden_dim=hidden_dim_edge,
            edge_dim=edge_dim,
            nb_gaussians=nb_gaussians_rbf,
            cutoff=cutoff
        )

        self.list_interaction_blocks = nn.ModuleList(
            [InteractionBlock(edge_dim=edge_dim, node_dim=node_dim) for i in range(num_iteration_block)]
        )

        if jump_connection: 
            H_dim = num_iteration_block * node_dim
        else:
            H_dim = node_dim

        self.output_block = OutputBlock(
            H_dim=H_dim, 
            hidden_dim=hidden_dim_output_block, 
            node_dim=node_dim
        )

    def forward(self, molecule, sfa_generator=None):

        batch_indices = getattr(molecule, "batch", None)
        X_global = molecule.pos
        U_star = SFA(X_global, batch_indices=batch_indices, generator=sfa_generator)
        if batch_indices is None:
            t = X_global.mean(axis=0)
        else:
            t = global_mean_pool(x=X_global, batch=batch_indices)
        X_local = rho1_inv(X_global, t, U_star, batch_indices)
        _, _, edge_index = adjacency_matrix(X_local, self.cutoff, batch_indices)
        
        H = self.molecule_embedding(molecule) # H0 de tous les atomes de la molécule ou du batch
        E = self.edge_embedding(X_local, edge_index) # embedding des arêtes

        if self.jump_connection:
            H_jump_list = []

        for interaction_block in self.list_interaction_blocks:
            H = interaction_block(H, E, edge_index, batch_indices)
            if self.jump_connection:
                H_jump_list.append(H)

        if self.jump_connection:
            H_jump = torch.concat(H_jump_list, dim=1)
            return self.output_block(H_jump, H, batch_indices)
        else: 
            return self.output_block(H, H, batch_indices)