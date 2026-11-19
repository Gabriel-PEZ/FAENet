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
dataset.z
unique_z = torch.unique(dataset.z)
z_to_idx = {z:idx for idx,z in enumerate(unique_z.tolist())} # dict

d = 16
emb = nn.Embedding(len(z_to_idx), d)
# chaine -> z de la molecule into idxs, into embedding
# mais du coup on doit utiliser le dict avec un tenseur
# mieux vaut utiliser un tenseur
# -1.0 ne peut pas être un indice
# on veut que les valeurs de z soit les positions
z_to_idx_tensor_cst = torch.full((max(unique_z)+1,), fill_value = -1, dtype=torch.long)
idx_tensor = torch.arange(len(unique_z))
z_to_idx_tensor_cst[unique_z] = idx_tensor

# verifie si l'embedding marche
emb_z = emb(z_to_idx_tensor_cst[molecule.z])
len(molecule.z)
c = (molecule.z == 1)
# les atomes d'hydrogene ont bien le même embedding
torch.isclose(emb_z[c][0],emb_z[c][1])

element_z = element(unique_z.tolist())
group_z = [a.group_id for a in element_z]
period_z = [a.period for a in element_z]
print(period_z)

# %%
# meme process pour period et group
set_period_z = sorted(set(period_z))
period_to_idx = torch.full((max(set_period_z) + 1,), -1, dtype=torch.long)
period_to_idx[torch.tensor(set_period_z)] = torch.arange(len(set_period_z))

set_group_z = sorted(set(group_z))
group_to_idx = torch.full((max(set_group_z) + 1,), -1, dtype=torch.long)
group_to_idx[torch.tensor(set_group_z)] = torch.arange(len(set_group_z))  

class Embedding_h0(nn.Module):
    def __init__(self, z_to_idx_tensor_cst, period_to_idx, group_to_idx, group_z, period_z, emb_dim, out_dim):
        super().__init__()
        self.register_buffer("z_to_idx", z_to_idx_tensor_cst)                           # [max_z + 1]
        self.register_buffer("period_of", period_to_idx[torch.tensor(period_z)])    # [K]
        self.register_buffer("group_of",  group_to_idx[torch.tensor(group_z)])  

        emb_in_z = len(period_z) # une ligne par élement
        emb_in_period = len(set(period_z)) # une ligne par period
        emb_in_group = len(set(group_z)) # une ligne par groupe

        self.emb_z = nn.Embedding(emb_in_z, emb_dim)
        self.emb_period = nn.Embedding(emb_in_period, emb_dim)
        self.emb_group = nn.Embedding(emb_in_group, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(3*emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, z):
        # le z est une liste de numéro atomique
        # pour emb_z on transforme en indice et on passe
        # pour le reste on veut que la valeur unique en indice
        idx = self.z_to_idx[z]
        h_0 = torch.cat(
            [
                self.emb_z(idx),
                self.emb_period(self.period_of[idx]),
                self.emb_group(self.group_of[idx])
            ]
        , dim = 1)
        h_0 = self.mlp(h_0)
        return h_0

m = Embedding_h0(z_to_idx_tensor_cst, period_to_idx, group_to_idx,
                 group_z, period_z, 16, 400)

H0 = m(molecule.z)
H0.size()

# %%
"""
Construction du tenseur des propriétés scalés pour tous les atomes du dataset
"""
unique_z_list = unique_z.tolist()
template = torch.zeros((len(unique_z_list),10))
# tenseur propriétés à concat embedding
# on remplit
for i,atom in enumerate(element(unique_z_list)):
    pro_atom = torch.tensor([
            atom.atomic_radius,
            atom.atomic_volume,
            atom.density,
            atom.dipole_polarizability,
            atom.electron_affinity,
            atom.en_allen,
            atom.vdw_radius,
            atom.covalent_radius,
            atom.ionenergies[1],
            atom.ionenergies.get(2, torch.nan)]
    )
    template[i,:] = pro_atom

print(template)












properties_atomes = torch.zeros(len(unique_z_list), 10)
print(properties_atomes) 
for i, num in enumerate(unique_z_list):
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
def relative_distance(X_local, edge_index):
    i, j = edge_index
    return X_local[i] - X_local[j] 
rij = relative_distance(X_local, edge_index)