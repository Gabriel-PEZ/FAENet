# %%

import torch 
from torch import nn

from embeddings import EdgeEmbedding, MoleculeEmbedding
from interaction_block import InteractionBlock
from output_block import OutputBlock
from graph_creation import SFA, rho1_inv, adjacency_matrix

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

    def forward(self, molecule):

        X_global = molecule.pos
        U_star = SFA(X_global)
        X_local = rho1_inv(X_global, X_global.mean(axis=0), U_star)
        _, _, edge_index = adjacency_matrix(X_local, cutoff=self.cutoff)
        
        H = self.molecule_embedding(molecule) # on récupère le H0 de tous les atomes de la molécule
        E = self.edge_embedding(X_local, edge_index) # embedding des arêtes

        if self.jump_connection:
            H_jump_list = []

        for interaction_block in self.list_interaction_blocks:
            H = interaction_block(H, E, edge_index)
            if self.jump_connection:
                H_jump_list.append(H)

        if self.jump_connection:
            H_jump = torch.concat(H_jump_list, dim=1)
            return self.output_block(H_jump, H)
        else: 
            return self.output_block(H, H)