# %%

import torch
from torch import nn

# %%

class MoleculeEmbedding(nn.Module):

    def __init__(
            self, 
            atom_embedding_dim, 
            hidden_dim, 
            node_dim,
            atomic_numbers_to_indices,
            properties_atomes_scaled,
        ):

        super().__init__()
        self.embedding_dim = atom_embedding_dim
        indices = atomic_numbers_to_indices.values()
        self.atomic_numbers_embedding = nn.Embedding(
            num_embeddings=len({val[0] for val in indices}),
            embedding_dim=atom_embedding_dim
        )

        self.groups_embedding = torch.nn.Embedding(
            num_embeddings=len({val[1] for val in indices}),
            embedding_dim=atom_embedding_dim
        )

        self.periods_embedding = torch.nn.Embedding(
            num_embeddings=len({val[2] for val in indices}),
            embedding_dim=atom_embedding_dim
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(3 * atom_embedding_dim + properties_atomes_scaled.shape[1], hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.SiLU()
        )

        self.atomic_numbers_to_indices = atomic_numbers_to_indices
        self.register_buffer("properties_atomes_scaled", properties_atomes_scaled)

        atomic_numbers_to_indices_tensor = torch.zeros(max(atomic_numbers_to_indices.keys())+1, 3, dtype=int)
        for key, value in atomic_numbers_to_indices.items():
            # tenseur avec lignes de 0 et sinon [Z, G, P] pour la molécule de numéro atomique i
            # ex hydrogene numéro atomique 1, ligne = 1 [Z_h, G_h, P_h]
            atomic_numbers_to_indices_tensor[key, :] = torch.tensor(value)
        self.register_buffer("atomic_numbers_to_indices_tensor", atomic_numbers_to_indices_tensor)


    def forward(self, molecule):
    
        zgp_atomes = self.atomic_numbers_to_indices_tensor[molecule.z] # chaque ligne = Z_i, G_i, P_i de l'atome i pour la molécule
        Z_embedded = self.atomic_numbers_embedding(zgp_atomes[:,0])
        G_embedded = self.groups_embedding(zgp_atomes[:,1])
        P_embedded = self.periods_embedding(zgp_atomes[:,2])
        properties_embedded = self.properties_atomes_scaled[zgp_atomes[:,0]]
        V = torch.concat([Z_embedded, G_embedded, P_embedded, properties_embedded], dim=1)
        H = self.mlp(V)
        return H

# %%

class EdgeEmbedding(nn.Module):

    def __init__(self, hidden_dim, edge_dim, nb_gaussians, cutoff):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nb_gaussians + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
            nn.SiLU()
        )

        # on stocke les tenseurs non apprenables dans le buffer pour le switch automatique cpu/gpu
        self.register_buffer("centers", torch.linspace(0, cutoff, nb_gaussians))
        self.register_buffer("sigma", self.centers[1] - self.centers[0])

    def forward(self, X_local, edge_index):

        # distance entre tous les atomes reliés par une arête via la matrice A
        R = X_local[edge_index[0]] - X_local[edge_index[1]]
        # norme de toutes ces distances
        d_rbf = torch.norm(R, dim=1)

        RBF = torch.exp(
            -(1/(2 * self.sigma**2)) * (d_rbf.unsqueeze(1) - self.centers)**2
        )
        concat_R_RBF = torch.concat([R, RBF], dim=1)

        return self.mlp(concat_R_RBF)