# %%
from pathlib import Path

import torch
from mendeleev import element
from torch_geometric.datasets import QM9

print(Path.cwd())

from embeddings import EdgeEmbedding, MoleculeEmbedding

# %%

dataset = QM9(root="../data/qm9")

# %%

molecule = dataset[2003]
print("Nom de la molécule :", molecule.name, "\n")

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

# dico pour le lookup embedding qui stocke numéro atomique : [idx_Z, idx_G, idx_P]
atomic_numbers_to_indices = {}
for i,z in enumerate(unique_atomic_numbers):
    atome = element(z)
    atomic_numbers_to_indices[z] = [i, unique_groups.index(atome.group_id), unique_periods.index(atome.period)]

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

molecule_embedding = MoleculeEmbedding(
    embedding_dim=128,
    hidden_dim=400,
    ouput_dim=400,
    atomic_numbers_to_indices=atomic_numbers_to_indices,
    properties_atomes_scaled=properties_atomes_scaled
)

edge_embedding = EdgeEmbedding(
    hidden_dim=400,
    output_dim=400,
    nb_gaussians=100,
    cutoff=6
)
# %%

print(molecule_embedding(molecule).shape)
print(edge_embedding(X_local, edge_index).shape)

# %%
