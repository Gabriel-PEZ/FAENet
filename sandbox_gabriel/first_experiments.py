# %%
from pathlib import Path

import torch
from mendeleev import element
from torch import nn
from torch_geometric.datasets import QM9

print(Path.cwd())

# %%

dataset = QM9(root="../data/qm9")

# %%

molecule = dataset[2003]
print("Nom de la molécule :", molecule.name, "\n")
print("Numéros atomiques :\n", molecule.z, "\n") # Z du papier
print("Coordonnées 3D pour chaque atome :\n", molecule.pos) # X du papier

# %% 

print("Caractéristiques des atomes : ", molecule.x)
"""
Colonnes 
- 0 à 4 : type d'atome encodé en one-hot (H, C, N, O, F)
- 5 : Numéro atomique 
- 6 : Nombre de voisins liés chimiquement 
- 7 : Nombre d'hydrogènes liés
- 8 : Charge formelle
- 9 : Nombre d'électrons non appariés 
- 10 : Indique si l'atome est aromatique
"""

# %%

# Dans QM9, chaque liaison chimique est représentée dans les deux sens.
print(molecule.edge_index)
# Par exemple pour la molécule 2003 du dataset, l'atome 0 (O) est lié à l'atome 6 (C)


# %%

print(molecule.edge_attr)
"""
Matrice de vecteurs one-hot qui décrit l'arête située dans la colonne 
correspondante de edge_index.
- (1,0,0,0) : liaison simple
- (0,1,0,0) : liaison double 
- (0,0,1,0) : liaison triple
- (0,0,0,1) : liaison aromatique
"""

print(molecule.y) # regression targets

# %%

def rho1_inv(X_global, t, U_star):
    """
    Fonction pour exprimer X du repère global au repère local 
    """
    return (X_global - t) @ U_star

def rho1(X_local, t, U_star):
    """
    Fonction pour exprimer X du repère local au repère global 
    """
    return X_local@ U_star.t() + t

def SFA(X, k=None):

    t = X.mean(axis=0)
    X_c = X - t 
    Sigma = X_c.t() @ X_c
    _, eigenvectors = torch.linalg.eigh(Sigma)
    signs = torch.cartesian_prod(
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0])
    ).unsqueeze(1)
    U = signs * eigenvectors
    if k is not None:
        U_star = U[k]
    else:
        U_star = U[torch.randint(0, 7, size=(1,)).item()]
    return U_star

def adjacency_matrix(X_local, cutoff):

    D = torch.cdist(X_local, X_local)
    A = (D < cutoff).float()
    for i in range(X_local.shape[0]):
        A[i,i] = 0.0
    edge_index = torch.nonzero(A).t() # same shape que dans QM9 (2 x E)
    return A, D, edge_index

# %%
X = molecule.pos
t = X.mean(axis=0)
U_star1 = SFA(X,0)
U_star2 = SFA(X,5)

X_local = rho1_inv(X, t, U_star1)
X_global = rho1(X_local, t, U_star1)
assert torch.isclose(X, X_global).all().item() == True # pour voir si rho1 * rho1_inv = Id

# %%
X_local2 = rho1_inv(X, t, U_star2)
A, D, edge_index = adjacency_matrix(X_local, cutoff=6)
_, D2, _ = adjacency_matrix(X_local2, cutoff=6)

# test pour vérifier que la distance est la même entre 2 repères différents
assert torch.isclose(D, D2).all().item() == True

# %%
"""
Construction du tenseur des propriétés scalés pour tous les atomes du dataset
"""
# obtenir les numéros atomiques, groupes et périodes uniques 
unique_atomic_numbers = sorted(dataset.z.unique().tolist()) # numéros atomiques uniques de QM9
groups = []
periods = []
for z in unique_atomic_numbers: 
    groups.append(element(z).group_id)
    periods.append(element(z).period)
unique_groups = sorted(set(groups))
unique_periods = sorted(set(periods))

# dico pour le lookup embedding
atomic_numbers_to_idx = {k:v for k,v in zip(unique_atomic_numbers, range(len(unique_atomic_numbers)))}
groups_to_idx = {k:v for k,v in zip(unique_groups, range(len(unique_groups)))}
periods_to_idx = {k:v for k,v in zip(unique_periods, range(len(unique_periods)))}

properties_atomes = torch.zeros(len(unique_atomic_numbers), 10) 
for i, num in enumerate(unique_atomic_numbers):
    atome = element(num)
    properties_atomes[i, :] = torch.tensor([
        atome.atomic_radius,
        atome.atomic_volume,
        atome.density,
        atome.dipole_polarizability,
        atome.electron_affinity,
        atome.en_allen,
        atome.vdw_radius,
        atome.covalent_radius,
        atome.ionenergies[1],
        atome.ionenergies.get(2, torch.nan) # gère le cas hydrogène
    ])

mu_atomes = torch.nanmean(properties_atomes, dim=0) # permet de compute la moyenne sans prendre en compte des nan
sigma_atomes = torch.sqrt(
    torch.nanmean((properties_atomes - mu_atomes)**2, dim=0)
)

# on remplace les nan par la moyenne de la catégorie correspondante
properties_atomes_imputed = torch.where(
    torch.isnan(properties_atomes),
    mu_atomes,
    properties_atomes
) 

# scaled pour éviter les différences d'échelles dans le MLP 
properties_atomes_scaled = (properties_atomes_imputed - mu_atomes)/sigma_atomes


# %% 
"""
Classe pour l'embedding des noeuds (atomes d'une molécule)
"""

class MoleculeEmbedding(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, ouput_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.atomic_numbers_embedding = nn.Embedding(
            num_embeddings=len(unique_atomic_numbers),
            embedding_dim=embedding_dim

        )
        self.groups_embedding = torch.nn.Embedding(
            num_embeddings=len(unique_groups),
            embedding_dim=embedding_dim
        )

        self.periods_embedding = torch.nn.Embedding(
            num_embeddings=len(unique_periods),
            embedding_dim=embedding_dim
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(3 * embedding_dim + 10, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, ouput_dim),
            nn.SiLU()
        )

    def forward(self, molecule):
        atomic_numbers = molecule.z.tolist()
        atome_embeddings = []
        for atomic_number in atomic_numbers:
            atome = element(atomic_number)
            atome_embedding = torch.concat(
                (
                    self.atomic_numbers_embedding(torch.tensor(atomic_numbers_to_idx[atome.atomic_number])),
                    self.groups_embedding(torch.tensor(groups_to_idx[atome.group_id])),
                    self.periods_embedding(torch.tensor(periods_to_idx[atome.period])),
                    properties_atomes_scaled[atomic_numbers_to_idx[atome.atomic_number],:]
                )
            )

            atome_embeddings.append(atome_embedding)
        V = torch.stack(atome_embeddings)
        H = self.mlp(V)
        return H


testME = MoleculeEmbedding(
    embedding_dim=128,
    hidden_dim=400,
    ouput_dim=400
)

# %%

V = testME(molecule)
print(V.shape)

# %%

"""
Embedding des arêtes basés sur le graphe d'adjacence construit avec X_local après SFA
"""

# distance entre tous les atomes reliés via la matrice d'adjacence
R = X_local[edge_index[0]] - X_local[edge_index[1]]

# norme de toutes ces distances
d_rbf = torch.norm(R, dim=1)

# centre des RBF gaussiennes
rbf_gaussian_centers = torch.linspace(0, 6, 100)
sigma = rbf_gaussian_centers[1] - rbf_gaussian_centers[0]

RBF = torch.exp(-(1/(2 * sigma**2)) * (d_rbf.unsqueeze(1) - rbf_gaussian_centers)**2)
# 132 * 103 (132 arêtes et pour chaque arête vecteur de R3 pour position relative + 100RBF)
concat_R_RBF = torch.concat([R, RBF], dim=1)

# %%

class EdgeEmbedding(nn.Module):

    def __init__(self):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(103, 400),
            nn.SiLU(),
            nn.Linear(400, 400),
            nn.SiLU()
        )

    def forward(self, concat_R_RBF):

        return self.mlp(concat_R_RBF)


edge_embedding = EdgeEmbedding()
# %%

print(edge_embedding(concat_R_RBF).shape)

# %%
