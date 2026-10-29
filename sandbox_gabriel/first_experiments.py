# %%
from pathlib import Path

import torch
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

X = molecule.pos

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

# %%
t = X.mean(axis=0)
U_star1 = SFA(X,0)
U_star2 = SFA(X,5)

X_local = rho1_inv(X, t, U_star1)
X_global = rho1(X_local, t, U_star1)
assert torch.isclose(X, X_global).all().item() == True # pour voir si rho1 * rho1_inv = Id

# %%

nb_atomes = X_local.shape[0]
D = torch.zeros(size=(nb_atomes, nb_atomes))
for i in range(nb_atomes):
    for j in range(nb_atomes):
        D[i,j] = torch.linalg.vector_norm(X_local[i] - X_local[j])

def build_D(X_local):
    nb_atomes = X_local.shape[0]
    D = torch.zeros(size=(nb_atomes, nb_atomes))
    for i in range(nb_atomes):
        for j in range(nb_atomes):
            D[i,j] = torch.linalg.vector_norm(X_local[i] - X_local[j])
    return D

X_local2 = rho1_inv(X, t, U_star2)

D1 = build_D(X_local)
D2 = build_D(X_local2)
# test pour vérifier si dans un repère différent les distances en changent pas
assert torch.isclose(D1, D2).all().item() == True
assert torch.isclose(D1, D1.t()).all().item() == True

# %%
D = torch.cdist(X_local, X_local) # plus efficace que build_D
# %%

