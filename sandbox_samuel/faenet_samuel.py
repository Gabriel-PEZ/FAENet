# %%

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from faenet.embeddings import EdgeEmbedding, MoleculeEmbedding
from faenet.interaction_block import InteractionBlock
from faenet.output_block import OutputBlock
from faenet.graph_creation import SFA, rho1_inv, adjacency_matrix


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
            framing = "sfa"

        ):
        """
        Inputs:
            - atom_embedding_dim = taille de l'embedding pour numéro atomique, groupe et période
        """

        super().__init__()
        self.cutoff = cutoff
        self.jump_connection = jump_connection
        self.framing = framing

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

    def gnn(self, H, X_local, edge_index, batch_indices):
        #Travaille avec un frame fixé
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

    def forward(self, molecule, sfa_generator=None):
        framing = self.framing
        batch_indices = getattr(molecule, "batch", None)
        X_global = molecule.pos
        _, _, edge_index = adjacency_matrix(X_global, self.cutoff, batch_indices)
        H = self.molecule_embedding(molecule)

        if batch_indices is None:
            t = X_global.mean(axis=0)
        else:
            t = global_mean_pool(x=X_global, batch=batch_indices)


        if framing == "sfa":
            U_star = SFA(X_global, batch_indices=batch_indices, generator=sfa_generator)
            X_local = rho1_inv(X_global, t, U_star, batch_indices)
            return self.gnn(H, X_local, edge_index, batch_indices)
        elif framing == None:
            X_local = X_global
            return self.gnn(H, X_local, edge_index, batch_indices)
        elif framing == "full":
            Y_sum, y_sum = 0, 0
            for k in range(8): #We compute all the possible orientation directions and then average over them
                U_star = SFA(X_global, batch_indices = batch_indices, k=k, generator=sfa_generator)
                X_local = rho1_inv(X_global, t, U_star, batch_indices)
                Y_k, y_k = self.gnn(H, X_local, edge_index, batch_indices)
                Y_sum += Y_k
                y_sum += y_k
            return Y_sum/8, y_sum/8
        else:
            raise ValueError(f"Unknown framing: {framing}")







