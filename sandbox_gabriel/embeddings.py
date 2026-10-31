# %%

import torch
from torch import nn

# %%

class MoleculeEmbedding(nn.Module):

    def __init__(
            self, 
            embedding_dim, 
            hidden_dim, 
            output_dim,
            atomic_numbers_to_indices,
            properties_atomes_scaled,
        ):

        super().__init__()
        self.embedding_dim = embedding_dim
        indices = atomic_numbers_to_indices.values()
        self.atomic_numbers_embedding = nn.Embedding(
            num_embeddings=len({val[0] for val in indices}),
            embedding_dim=embedding_dim
        )

        self.groups_embedding = torch.nn.Embedding(
            num_embeddings=len({val[1] for val in indices}),
            embedding_dim=embedding_dim
        )

        self.periods_embedding = torch.nn.Embedding(
            num_embeddings=len({val[2] for val in indices}),
            embedding_dim=embedding_dim
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(3 * embedding_dim + 10, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU()
        )

        self.atomic_numbers_to_indices = atomic_numbers_to_indices
        self.register_buffer("properties_atomes_scaled", properties_atomes_scaled)


    def forward(self, molecule):
        atomic_numbers = molecule.z.tolist()
        atome_embeddings = []
        for atomic_number in atomic_numbers:

            idx_Z = self.atomic_numbers_to_indices[atomic_number][0]
            idx_G = self.atomic_numbers_to_indices[atomic_number][1]
            idx_P = self.atomic_numbers_to_indices[atomic_number][2]

            atome_embedding = torch.concat(
                (
                    self.atomic_numbers_embedding(torch.tensor(idx_Z)),
                    self.groups_embedding(torch.tensor(idx_G)),
                    self.periods_embedding(torch.tensor(idx_P)),
                    self.properties_atomes_scaled[idx_Z,:]
                )
            )

            atome_embeddings.append(atome_embedding)
        V = torch.stack(atome_embeddings)
        H = self.mlp(V)
        return H

# %%

class EdgeEmbedding(nn.Module):

    def __init__(self, hidden_dim, output_dim, nb_gaussians, cutoff):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nb_gaussians + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
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