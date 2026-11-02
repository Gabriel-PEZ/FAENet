# %%
from pathlib import Path

import torch
from mendeleev import element
from torch import nn
from torch_geometric.datasets import QM9
from torch_geometric.nn import GraphNorm

print(Path.cwd())

from embeddings import EdgeEmbedding, MoleculeEmbedding
from graph_creation import SFA, rho1, rho1_inv, adjacency_matrix
from faenet import FAENet

# %%

dataset = QM9(root="../data/qm9")

# %%

molecule = dataset[2003]
print("Nom de la molécule :", molecule.name, "\n")

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
len(properties_atomes_scaled)

# %%

faenet = FAENet(
    atom_embedding_dim=128,
    hidden_dim_node=400,
    hidden_dim_edge=480,
    hidden_dim_output_block=200,
    node_dim=400,
    edge_dim=480,
    atomic_numbers_to_indices=atomic_numbers_to_indices,
    properties_atomes_scaled=properties_atomes_scaled,
    nb_gaussians_rbf=100,
    cutoff=6.0,
    num_iteration_block=5,
    jump_connection=True,
)
# %%

faenet(molecule)
# %%
