# %%
from pathlib import Path

import torch
from mendeleev import element
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

# test embedding premier atome de la molécule
z_0 = molecule.z[0].item() # = numéro atomique du premier atome de la molécule
atome = element(z_0) # fonction de mendeleev pour récupérer les caractéristiques

unique_atomic_number = dataset.z.unique().tolist()

groups = []
periods = []
for z in unique_atomic_number: 
    groups.append(element(z).group_id)
    periods.append(element(z).period)
unique_groups = list(set(groups))
unique_periods = list(set(periods))

# table correspondance pour lignes de l'embedding
atomic_numbers_to_idx = {k:v for k,v in zip(unique_atomic_number, range(len(unique_atomic_number)))}
groups_to_idx = {k:v for k,v in zip(unique_groups, range(len(unique_groups)))}
periods_to_idx = {k:v for k,v in zip(unique_periods, range(len(unique_periods)))}

d = 128

atomic_numbers_embedding = torch.nn.Embedding(
    num_embeddings=len(unique_atomic_number),
    embedding_dim=d
)

groups_embedding = torch.nn.Embedding(
    num_embeddings=len(unique_groups),
    embedding_dim=d
)

periods_embedding = torch.nn.Embedding(
    num_embeddings=len(unique_periods),
    embedding_dim=d
)

# %%

# atome.metallic_radius, (H, C, O, N, F non métaux donc None pour metallic_radius)

carac_phys_0 = torch.tensor(
    [
        atome.atomic_radius,
        atome.atomic_volume,
        atome.density,
        atome.dipole_polarizability,
        atome.electron_affinity,
        atome.en_allen,
        atome.vdw_radius,
        atome.covalent_radius,
        # atome.metallic_radius if atome.metallic_radius is not None else 0.0,
        atome.ionenergies[1],
        atome.ionenergies.get(2, 0.0) # gère le cas hydrogène

    ]
)

atome_0_embedding_concat = torch.concat(
    (
        atomic_numbers_embedding(torch.tensor(atomic_numbers_to_idx[atome.atomic_number])),
        groups_embedding(torch.tensor(groups_to_idx[atome.group_id])),
        periods_embedding(torch.tensor(periods_to_idx[atome.period])),
        carac_phys_0
    )
)
# %%


# matrice des caractéristiques de l'ensemble des atomes disponibles dans QM9 

print(unique_atomic_number)
print(len(carac_phys_0))

# %%

statistics_atomes = torch.zeros(len(unique_atomic_number), len(carac_phys_0)) 
for i, num in enumerate(unique_atomic_number):
    atome = element(num)
    statistics_atomes[i, :] = torch.tensor([
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

mu_atomes = torch.nanmean(statistics_atomes, dim=0)# permet de compute la moyenne sans prendre en compte des nan
sigma_atomes = torch.sqrt(
    torch.nanmean((statistics_atomes - mu_atomes)**2, dim=0)
)

# %% 


