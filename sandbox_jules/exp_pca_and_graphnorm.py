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
#dataset = QM9(root=qm9)
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
    center = mol_test.pos.mean(dim=0)
    # loop sur les frames pour calculer x_local
    pred_frames = []
    for frame in frames:
        X_local = (mol_test.pos - center)@frame
        # on crée la nouvelle molécule 
        mol_local = Data(z = mol.z, pos = X_local)

        y_atomes, y_graph = faenet(mol_local)
        pred_frames.append(y_atomes)
    return pred_frames

#%%
frames = SFA(mol_test.pos, full = True)
predict_pca_manual(mol_test, frames)