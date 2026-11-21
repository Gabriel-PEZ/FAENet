# %%
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import QM9
from faenet.atomic_features import build_atomic_features
from faenet.faenet import FAENet
from faenet.graph_creation import SFA, adjacency_matrix, rho1_inv

device = torch.device("cpu")

root = Path.cwd().resolve().parent 
qm9 = root / "data" / "qm9"
dataset = QM9(root=qm9)
checkpoint = root / "scripts" / "checkpoints" / "qm9_0_sfa" / "best.pt"

# %%

atomic_numbers_to_indices, properties_atomes_scaled = build_atomic_features(dataset)
split_generator = torch.Generator().manual_seed(2003)
# on reprend le train/val/test via le generator 2003
indices = torch.randperm(len(dataset), generator=split_generator)

#%%

train_size = 110000
val_size = 10000

test_start = train_size + val_size

# indices du train, val, test
train_indices = indices[:train_size]
val_indices = indices[train_size:test_start]
test_indices = indices[test_start:]
# slice le dataset
dataset_train = dataset[train_indices]
dataset_val = dataset[val_indices]
dataset_test = dataset[test_indices]
#%%
# build the model
# and load ckpt
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
    frame_averaging="full",
).to(device)

checkpoint_weights = torch.load(checkpoint, map_location=device, weights_only=True)

#%%
faenet.load_state_dict(checkpoint_weights["model_state_dict"])
faenet.eval()

#%%
# test sur une molécule
mol_test = dataset_test[0]

with torch.no_grad():
    y_atomes, y_graph = faenet(mol_test)

y_graph

#%%

# maintenant on veut isoler les deux effets : PCA et graphnorm
# mais en utilisant le mode full de frame averaging, le réseau recalcule la PCA selon l'entrée
# or si on fait ça, on ne peut pas étudier l'influence de A via B (tout en gardant uniquement A dans le réseau)
# donc il faut calculer la PCA de D = A U B à la main et inversement la PCA de A tout en gardant B dans le réseau pour l'effet graph norm

# on crée un helper qui permet de contrôler la PCA
@torch.no_grad()
def predict_pca_manual(mol, frames):
    # mol est le graphe qu'on passe à FAEnet
    # frames est le tenseur issu de SFA sur notre objet choisi -> 8x3x3 (les 8 U)
    '''
    fonction qui permet de faire des prédictions full FA avec un custom PCA
    '''
    #centroide
    center = mol.pos.mean(dim=0)
    # loop sur les frames pour calculer x_local
    pred_frames = []
    for frame in frames:
        X_local = (mol.pos - center)@frame
        # on crée la nouvelle molécule 
        mol_local = Data(z = mol.z, pos = X_local)

        y_atomes, y_graph = faenet(mol_local)
        pred_frames.append(y_atomes)
    return torch.stack(pred_frames)

#%%
frames = SFA(mol_test.pos, full = True)
predict_pca_manual(mol_test, frames).mean(dim = 0)

A = Data(z=mol_test.z, pos=mol_test.pos)
faenet.frame_averaging = "full"
with torch.no_grad():
    full_A, full_A_graph = faenet(A)

faenet.frame_averaging = "none"
frames_A = SFA(A.pos, full=True)
manual_A = predict_pca_manual(A, frames_A)

print(torch.isclose(full_A, manual_A, atol = 1e-4))

#%%
def add_two_molecules(A, B, direction, distance):
    pos_A = A.pos
    pos_B = B.pos
    # partir du même point d'origine
    pos_A_centered = pos_A - A.pos.mean(dim = 0)
    pos_B_centered = pos_B - B.pos.mean(dim = 0)
    # éloigner B de d
    # on normalise pour avoir une distance de d (vecteur unitaire)
    direction = direction / direction.norm()
    pos_B_shiftfed = pos_B_centered + distance*direction
    # assembler les molécules
    mol_AB = Data(pos = torch.concat([pos_A_centered, pos_B_shiftfed]), z = torch.concat([A.z,B.z]))

    # size N_A, N_B
    distance_atomes = torch.cdist(pos_A_centered,pos_B_shiftfed)
    distance_min = distance_atomes.min().item()
    return mol_AB, distance_min


#%%
