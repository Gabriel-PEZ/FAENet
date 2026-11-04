# %%
from pathlib import Path

import torch
from torch import nn
from mendeleev import element
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool

print(Path.cwd())

from embeddings import EdgeEmbedding, MoleculeEmbedding
from graph_creation import SFA, rho1, rho1_inv, adjacency_matrix
from faenet import FAENet

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = QM9(root="../data/qm9")

# %%

molecule = dataset[2003]
molecule = molecule.to(device)
print("Nom de la molécule :", molecule.name, "\n")
print("Device :", molecule.pos.device)

# %%

X = molecule.pos
t = X.mean(axis=0)
U_star1 = SFA(X,k=0)
print(U_star1.device)

U_star2 = SFA(X,k=5)

X_local = rho1_inv(X, t, U_star1)
X_global = rho1(X_local, t, U_star1)

# %%

assert torch.allclose(X, X_global, atol=1e-5, rtol=1e-5) # pour voir si rho1 * rho1_inv = Id
# %%

X_local2 = rho1_inv(X, t, U_star2)
A, D, edge_index = adjacency_matrix(X_local, cutoff=6)
_, D2, _ = adjacency_matrix(X_local2, cutoff=6)

# test pour vérifier que la distance est la même entre 2 repères différents
assert torch.allclose(D, D2, atol=1e-5, rtol=1e-5)

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
).to(device)

# %%

generator = torch.Generator().manual_seed(42)
idx_perm = torch.randperm(len(dataset), generator=generator)
dataset_shuffled = dataset[idx_perm]

# Plusieurs target de régression dans y. 
# 0 = moment dipolaire \mu
idx_regression_target = 0

size_train = int(0.85 * len(dataset))
size_val = int(0.05 * len(dataset))
size_test = size_train + size_val 

dataset_train = dataset_shuffled[:size_train]
dataset_val = dataset_shuffled[size_train:size_test]
dataset_test = dataset_shuffled[size_test:]

target_train = dataset_train.y[:,idx_regression_target]
target_val = dataset_val.y[:,idx_regression_target]
target_test = dataset_test.y[:,idx_regression_target]

mu_target_train = target_train.mean()
std_target_train = target_train.std()
target_train_scaled = (target_train - mu_target_train)/std_target_train

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

# A savoir : comme les molécules n'ont pas le même nombre d'atomes, on peut pas stocker les molécules
# sous la forme d'un batch de taille B x N x 3 ou N = nombre d'atomes et 3 pour x,y,z.
# A la place, on crée un graphe composé de B sous-graphes, où chaque graphe est déconnecté des 
# autres graphes 

# %%

it = iter(train_loader)
batch = next(it)
print(batch.batch) # permet de voir quel atome appartient à quelle molécule
print(batch.batch.shape) # nombre d'atomes dans le batch
print(batch.ptr) # donne les indices pour savoir quand commence et finit une molécule dans batch.batch
print(batch.pos) # positions x,y,z de chaque atome

# %%

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    faenet.parameters(),
    lr=1e-3
)
sfa_generator = torch.Generator().manual_seed(2003)
num_epochs = 1

faenet.train()
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):

        optimizer.zero_grad()
        batch = batch.to(device)
        y_true = batch.y[:, idx_regression_target:idx_regression_target+1]
        _, y_pred = faenet(batch, sfa_generator=sfa_generator)

        # check de sécurité pour éviter broadcasting dans la loss
        assert y_true.shape == y_pred.shape 

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: MSE = {loss.item():.6f}")
    
# %%