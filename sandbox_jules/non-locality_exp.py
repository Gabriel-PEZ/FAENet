# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9

from faenet.atomic_features import build_atomic_features
from faenet.faenet import FAENet
from faenet.graph_creation import SFA, adjacency_matrix

device = torch.device("cpu")

root = Path.cwd().resolve().parent

# load le dataset et les weights
qm9 = root / "data" / "qm9"
dataset = QM9(root=qm9)
checkpoint = root / "scripts" / "checkpoints" / "fix_collapse" / "best_sfa_noweights.pt"

# dossier où on sauvegarde toutes les figures et animations
out_dir = root / "output" / "pca_and_graphnorm"
out_dir.mkdir(parents=True, exist_ok=True)

# palette commune à tous les plots

colors = {
    "Combined": "#224a83",
    "GraphNorm": "#007c66",
    "PCA": "#bd5a1d",
    "Random frame": "#7455a4",
    "SFA": "#6c7787",
}
color_A = "#172a41"
color_B = "#a3adb8"
color_red = "#b62e39"
# %%
atomic_numbers_to_indices, properties_atomes_scaled = build_atomic_features(dataset)
split_generator = torch.Generator().manual_seed(2003)
# on reprend le train/val/test via le generator 2003
indices = torch.randperm(len(dataset), generator=split_generator)
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

# build the model and load ckpt
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
    use_weights=False,
    xavier_init=True,
).to(device)

checkpoint_weights = torch.load(checkpoint, map_location=device, weights_only=True)
faenet.load_state_dict(checkpoint_weights["model_state_dict"])
faenet.eval()

# params de noramlisation de la cible pour repasser en Debye et calculer les écarts
mean_y = checkpoint_weights["target_mean"].item()
std_y = checkpoint_weights["target_std"].item()

# %%
# test sur une molécule
mol_test = dataset_test[0]

with torch.no_grad():
    y_atomes, y_graph = faenet(mol_test)

print(f"y_graph = {y_graph.item():.4f} | Debye = {mean_y + std_y * y_graph.item():.3f} D")

# %%
# maintenant on veut isoler les deux effets : PCA et GraphNorm
# mais en utilisant le mode full de frame averaging, le réseau recalcule la PCA selon l'entrée
# or si on fait ça, on ne peut pas étudier l'influence de B sur A uniquement via la PCA (en gardant seulement A dans le réseau)
# donc il faut calculer la PCA de A U B à la main, et inversement garder la PCA de A tout en mettant A U B dans le réseau pour l'effet GraphNorm

@torch.no_grad()
def predict_pca_manual(mol, frames):
    # mol est le graphe qu'on passe à FAENet
    # frames est le tenseur issu de SFA sur notre objet choisi -> (8, 3, 3) (les 8 repères U)
    """
    fonction qui permet de faire des prédictions full FA avec une PCA choisie à la main.
    """
    assert faenet.frame_averaging == "none"  # sinon faenet va override et remplacer notre PCA
    center = mol.pos.mean(dim=0)
    pred_frames = []
    # PCA manuelle
    for frame in frames:
        X_local = (mol.pos - center) @ frame # formule rho1_inv
        mol_local = Data(z=mol.z, pos=X_local) # crée un nouvel objet PyGeometric
        y_atomes, y_graph = faenet(mol_local) # prédiction
        pred_frames.append(y_atomes)
    # taille (n_frames, n_atomes, 1) 
    return torch.stack(pred_frames)


# %%
# test de vérification : notre predict_pca_manual doit donner exactement la même chose que frame_averaging="full"
A = Data(z=mol_test.z, pos=mol_test.pos)

faenet.frame_averaging = "full"
with torch.no_grad():
    full_A, full_A_graph = faenet(A)

faenet.frame_averaging = "none"
frames_A = SFA(A.pos, full=True)
manual_A = predict_pca_manual(A, frames_A).mean(dim=0)

test_ok = torch.isclose(full_A, manual_A, atol=1e-4)
print(test_ok.all())

# %%
# fonctions utilitaires pour assembler deux molécules et tirer des repères aléatoires


def add_two_molecules(A, B, direction, distance):
    pos_A = A.pos
    pos_B = B.pos
    # partir du même point d'origine
    pos_A_centered = pos_A - pos_A.mean(dim=0)
    pos_B_centered = pos_B - pos_B.mean(dim=0)
    # éloigner B de distance le long de direction (vecteur unitaire)
    direction = direction / direction.norm()
    pos_B_shifted = pos_B_centered + distance * direction
    # assembler les deux molécules dans un seul graphe
    mol_AB = Data(
        pos=torch.concat([pos_A_centered, pos_B_shifted]),
        z=torch.concat([A.z, B.z]),
    )
    # matrice de distances (n_A, n_B)
    distance_atomes = torch.cdist(pos_A_centered, pos_B_shifted)
    distance_min = distance_atomes.min().item()

    # on return le nouvel objet et la distance minimale
    # pour regarder si on est dans le graphe ou non (comparer au cutoff)
    return mol_AB, distance_min 


def random_eight_frames():
    # construit 8 repères orthogonaux aléatoires (non issus de la PCA)
    G = torch.randn(3, 3)
    # F est orthogonale donc constitue un repère
    F, R = torch.linalg.qr(G)
    signs = torch.cartesian_prod(
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
    )
    # (8, 3, 3) -> les 8 combinaisons de signes du repère 3x3
    F = F.unsqueeze(0)
    signs = signs.unsqueeze(1)
    return signs * F


# %%

# expérience 1:

# le but est d'étudier l'effet d'ajouter B à A en regardant la prédiction de A

# crée A et B
A = dataset_test[0]
B = dataset_test[1]
n_A = len(A.z)
d_far = 35.0

# centre + covariance -> vecteurs propres de la PCA de A
X_c = A.pos - A.pos.mean(dim=0)
sigma = X_c.T @ X_c
vals, vecs = torch.linalg.eigh(sigma)

# comme direction on prend la somme des axes de la PCA de A (normalisée)
# comme ça quand on éloigne B de A, on est sûrs de faire pivoter le repère PCA de A U B
direction = vecs.sum(dim=1)
direction = direction / direction.norm()

# REFERENCE = prédiction de A seul
faenet.frame_averaging = "none"
frames_Aref = SFA(A.pos, full=True)
preds_Aref = predict_pca_manual(A, frames_Aref)  # (8, n_A, 1)

pred_Aref = preds_Aref.mean(dim=0)
std_Aref = preds_Aref.std(dim=0)
ref_sum = pred_Aref.sum().item()

# bruit intrinsèque de SFA entre les 8 flips de signe
# pour voir si ce que l'on trouve en PCA est significatif
noise_D = std_y * preds_Aref.sum(dim=1).std().item()

# contrôle random frame sur A seul (sans B)
# pour voir si la modification de la PCA obtenu en introduisant B a un effet différent que de prendre une frame aléatoire pour A seul
torch.manual_seed(123)
random_frameA = []
for _ in range(20):
    frames_rand = random_eight_frames()
    preds_Arand = predict_pca_manual(A, frames_rand).mean(dim=0)
    random_frameA.append(std_y * abs(preds_Arand.sum().item() - ref_sum))

rand_mean_randomA = torch.tensor(random_frameA).mean().item()
rand_std_randomA = torch.tensor(random_frameA).std().item()

print(f"A de référence {mean_y + std_y * ref_sum:.3f} D")
print(f"Bruit SFA (8 flips) : {noise_D:.4f} D")
print(f"Random frame : {rand_mean_randomA:.3f} ± {rand_std_randomA:.3f} D")

# %%
# boucle de calcul

# espace plus sérré vers A car après il y a convergence
drange = np.geomspace(6.0, 40.0, 45)
d_min_values = [] # pour comparer au cutoff
full = [] # dérive totale de la prédiction de tous les atomes de A quand avec A U B dans le réseau et A U B en PCA
pca = [] # dérive totale de la prédiction de tous les atomes de A quand le repère utilisé pour la PCA est A U B
gn = [] # # dérive totale de la prédiction de tous les atomes de A quand on projette A normalement mais on laisse B dans le réseau

# infos géométriques stockées pour les plots 3D et le GIF
pos_B_list = [] # coordonnées de B à chaque step
cross_edges_list = [] # paires d'atomes à chaque step pour le gif
basis_AB_list = [] # repères PCA

basis_AB = frames_Aref[-1].clone()  # repère PCA de départ (celui de A)

for d in drange:

    # création molécule
    mol_AB, d_min = add_two_molecules(A, B, direction, d)
    d_min_values.append(d_min)

    # arêtes entre A et B (pour savoir si on est en-dessous du cutoff et pour les afficher en 3D)
    adj_AB, _, _ = adjacency_matrix(mol_AB.pos, faenet.cutoff)
    # arrêtes uniquement de A à B
    cross_edges = adj_AB[:n_A, n_A:].nonzero()
    # PCA de A U B
    frames_ABcomb = SFA(mol_AB.pos, full=True)

    # pour les vizualizations on a besoin d'axes continus, or le flip (180°) existe et est valide ici
    # donc quand on a flip vs l'étape précédente on reflip (via le torch.where)
    new_basis = frames_ABcomb[-1] # étape d'avant
    flips = torch.where((basis_AB * new_basis).sum(dim=0) >= 0, 1.0, -1.0)
    basis_AB = new_basis * flips # maj

    # effet combiné (PCA de A U B + GraphNorm sur A U B)
    preds_ABcomb = predict_pca_manual(mol_AB, frames_ABcomb)
    preds_Acomb = preds_ABcomb[:, :n_A, :].mean(dim=0)
    full.append(std_y * abs(preds_Acomb.sum().item() - ref_sum)) # on calcul la dérive à la ref

    # effet PCA seul (on passe seulement A au réseau, donc pas d'influence de B dans GraphNorm)
    preds_APCA = predict_pca_manual(A, frames_ABcomb).mean(dim=0)
    pca.append(std_y * abs(preds_APCA.sum().item() - ref_sum))

    # effet GraphNorm seul (on passe A U B mais avec la PCA de A)
    # on ne le quantifie que quand il n'y a plus aucune arête entre A et B sinon c'est du message passing
    if len(cross_edges) == 0:
        preds_ABgraph = predict_pca_manual(mol_AB, frames_Aref)
        preds_Agraph = preds_ABgraph[:, :n_A, :].mean(dim=0)
        gn.append(std_y * abs(preds_Agraph.sum().item() - ref_sum))
    else:
        gn.append(float("nan"))

    pos_B_list.append(mol_AB.pos[n_A:].clone())
    cross_edges_list.append(cross_edges)
    basis_AB_list.append(basis_AB.clone())



# %%
# dérive en fonction de la distance d_min (axe y cassé pour zoomer sur le bas)
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(8.2, 4.6), gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.08}
)

# panneau du haut : Combined et GraphNorm
ax_top.plot(d_min_values, full, color=colors["Combined"], lw=2, label="Combined")
ax_top.plot(d_min_values, gn, color=colors["GraphNorm"], lw=2, label="GraphNorm")
ax_top.axvline(faenet.cutoff, color="grey", ls=":", lw=1)
ax_top.set_ylim(2.15, 3.5)
ax_top.spines[["top", "right", "bottom"]].set_visible(False)
ax_top.tick_params(bottom=False)
ax_top.grid(axis="y", alpha=0.2, ls=":")
ax_top.set_title("Prediction drift vs. distance")

# panneau du bas (zoom) : PCA, Random frame, Test MAE, SFA
ax_bot.plot(d_min_values, pca, color=colors["PCA"], lw=2, label="PCA")
ax_bot.axhline(rand_mean_randomA, color=colors["Random frame"], ls=":", label="Random frame")
ax_bot.axhspan(
    max(0.0, rand_mean_randomA - rand_std_randomA),
    rand_mean_randomA + rand_std_randomA,
    color=colors["Random frame"],
    alpha=0.12,
    lw=0,
)
ax_bot.axhline(noise_D, color=colors["SFA"], ls="-.", label="SFA")
ax_bot.axhline(0.156, color=color_red, ls="--", label="Test MAE")
ax_bot.axvline(faenet.cutoff, color="grey", ls=":", lw=1)
ax_bot.set_ylim(-0.02, 0.48)
ax_bot.spines[["top", "right"]].set_visible(False)
ax_bot.grid(axis="y", alpha=0.2, ls=":")
ax_bot.set_xlabel(r"$d_{\min}$ (Å)")

# marques de cassure sur l'axe y
d_brk = 0.015
ax_top.plot((-d_brk, d_brk), (-d_brk, d_brk), transform=ax_top.transAxes, color="k", lw=1, clip_on=False)
ax_bot.plot((-d_brk, d_brk), (1 - d_brk, 1 + d_brk), transform=ax_bot.transAxes, color="k", lw=1, clip_on=False)

fig.supylabel(r"$|\Delta\hat{\mu}_A|$ (D)")
h1, l1 = ax_top.get_legend_handles_labels()
h2, l2 = ax_bot.get_legend_handles_labels()
ax_top.legend(
    [h1[0], h2[0], h1[1], h2[1], h2[2], h2[3]],
    [l1[0], l2[0], l1[1], l2[1], l2[2], l2[3]],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)
fig.savefig(out_dir / "fig1_distance.png", dpi=200, bbox_inches="tight")
plt.show()

# %%

# visuels 3D
# on construit une matrice de vue pour que la direction d'éloignement de B soit alignée avec l'axe x
e_ref = torch.eye(3)[direction.abs().argmin()] # choisit l'axe canonique le moins colinéaire à direction
sideways = torch.cross(e_ref, direction, dim=0) # vecteur orthogonal à direction (futur axe y de la vue)
sideways = sideways / sideways.norm() # vecteur unitaire
z_axis = torch.cross(direction, sideways, dim=0) # futur axe z
view_matrix = torch.stack([direction, sideways, z_axis], dim=1) # nouvelle base canonique

pos_A_centered = A.pos - A.pos.mean(dim=0) # centrage
pos_A_view = (pos_A_centered @ view_matrix).numpy() # nouvelle projection des atomes


def setup_3d_axes(ax, x_range, y_range, z_range, zoom=1.0):
    # bornes
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    # évite d'étirer les molécules en 1:1:1
    ax.set_box_aspect([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]], zoom=zoom)
    ax.set_proj_type("ortho")
    ax.view_init(elev=19, azim=-69)
    ax.set_axis_off() # cache les axes


def draw_molecule_3d(ax, pos, mol, color, size_scale=1.0):
    # liaisons (on ne trace chaque liaison qu'une seule fois avec i < j)
    bonds = mol.edge_index[:, mol.edge_index[0] < mol.edge_index[1]].T.numpy()
    segments = pos[bonds]
    ax.add_collection3d(Line3DCollection(segments, colors="#8998a6", linewidths=1.1 * (size_scale ** 0.5)))
    sizes = np.where(mol.z.numpy() == 1, 35, 72) * size_scale
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=sizes, depthshade=False, edgecolor="white", linewidth=0.6)


def draw_pca_frame_3d(ax, basis, length):
    # pour chaque axe on trache une fleche de l'origne dans la direction
    for col in range(3):
        v = basis[:, col]
        ax.quiver(0, 0, 0, v[0], v[1], v[2], length=length, arrow_length_ratio=0.12, color=colors["PCA"], linewidth=1.6)


# %%
# tableau
#
idx = 14 # dmin = 7
pos_B_near_view = (pos_B_list[idx] @ view_matrix).numpy()

# nom du fichier, afficher B ?, repère PCA
vignettes_config = [
    ("isolated", False, frames_Aref[-1]),
    ("pca", False, basis_AB_list[idx]),
    ("gn", True, frames_Aref[-1]),
    ("combined", True, basis_AB_list[idx]),
]

# boucle sur les vignettes
for name, show_B, basis in vignettes_config:
    fig = plt.figure(figsize=(6, 3))
    # 3D
    ax = fig.add_subplot(projection="3d")
    setup_3d_axes(ax, (-3.0, 14.0), (-3.0, 3.0), (-3.0, 3.0))
    # draw la flèche
    draw_pca_frame_3d(ax, (view_matrix.T @ basis).numpy(), length=4.6)
    # draw la molécule
    draw_molecule_3d(ax, pos_A_view, A, color_A)
    if show_B:
        draw_molecule_3d(ax, pos_B_near_view, B, color_B)
    fig.savefig(out_dir / f"design_{name}.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


# %%
# gif de link + dérive
fps = 12

# 2 panneaux (3D à gauche, axe y cassé à droite)
fig = plt.figure(figsize=(13.5, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.15], height_ratios=[1, 1.4], hspace=0.08, wspace=0.22)
fig.subplots_adjust(left=0.01, right=0.84, top=0.92, bottom=0.12)

ax3d = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)

# panneau du haut
ax_top.plot(d_min_values, full, color=colors["Combined"], lw=2, label="Combined")
ax_top.plot(d_min_values, gn, color=colors["GraphNorm"], lw=2, label="GraphNorm")
ax_top.axvline(faenet.cutoff, color="grey", ls=":", lw=1)
ax_top.set_ylim(2.15, 3.5)
ax_top.spines[["top", "right", "bottom"]].set_visible(False)
ax_top.tick_params(bottom=False, labelbottom=False)
ax_top.grid(axis="y", alpha=0.2, ls=":")
ax_top.set_title("Prediction drift vs. distance")

# panneau du bas (zoom)
ax_bot.plot(d_min_values, pca, color=colors["PCA"], lw=2, label="PCA")
ax_bot.axhline(rand_mean_randomA, color=colors["Random frame"], ls=":", label="Random frame")
ax_bot.axhspan(
    max(0.0, rand_mean_randomA - rand_std_randomA),
    rand_mean_randomA + rand_std_randomA,
    color=colors["Random frame"],
    alpha=0.12,
    lw=0,
)
ax_bot.axhline(noise_D, color=colors["SFA"], ls="-.", label="SFA")
ax_bot.axhline(0.156, color=color_red, ls="--", label="Test MAE")
ax_bot.axvline(faenet.cutoff, color="grey", ls=":", lw=1)
ax_bot.set_ylim(-0.02, 0.48)
ax_bot.spines[["top", "right"]].set_visible(False)
ax_bot.grid(axis="y", alpha=0.2, ls=":")
ax_bot.set_xlabel(r"$d_{\min}$ (Å)")
ax_bot.set_ylabel(r"$|\Delta\hat{\mu}_A|$ (D)", y=0.92)

d_brk = 0.015
ax_top.plot((-d_brk, d_brk), (-d_brk, d_brk), transform=ax_top.transAxes, color="k", lw=1, clip_on=False)
ax_bot.plot((-d_brk, d_brk), (1 - d_brk, 1 + d_brk), transform=ax_bot.transAxes, color="k", lw=1, clip_on=False)

h1, l1 = ax_top.get_legend_handles_labels()
h2, l2 = ax_bot.get_legend_handles_labels()
ax_top.legend(
    [h1[0], h2[0], h1[1], h2[1], h2[2], h2[3]],
    [l1[0], l2[0], l1[1], l2[1], l2[2], l2[3]],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)
cursor_top = ax_top.axvline(d_min_values[0], color=color_A, lw=1.2)
cursor_bot = ax_bot.axvline(d_min_values[0], color=color_A, lw=1.2)


def draw_dissociation_frame(i):
    ax3d.clear()
    setup_3d_axes(ax3d, (-3.6, 12.6), (-3.8, 3.8), (-3.8, 3.8), zoom=1.60)

    pos_B_view = (pos_B_list[i] @ view_matrix).numpy()
    # on arrête de dessiner B une fois qu'elle sort du cadre 3D
    if pos_B_view[:, 0].max() <= 12.6:
        draw_molecule_3d(ax3d, pos_B_view, B, color_B, size_scale=2.2)

    # arêtes en rouge pointillé tant que d_min < cutoff
    edges = cross_edges_list[i]
    if len(edges) > 0:
        ia, ib = edges.numpy().T
        cross_segments = np.stack([pos_A_view[ia], pos_B_view[ib]], axis=1)
        ax3d.add_collection3d(
            Line3DCollection(cross_segments, colors=color_red, linestyles="--", linewidths=0.8, alpha=0.55)
        )

    draw_pca_frame_3d(ax3d, (view_matrix.T @ basis_AB_list[i]).numpy(), length=4.0)
    draw_molecule_3d(ax3d, pos_A_view, A, color_A, size_scale=2.2)
    cursor_top.set_xdata([d_min_values[i], d_min_values[i]])
    cursor_bot.set_xdata([d_min_values[i], d_min_values[i]])


anim_dissoc = FuncAnimation(fig, draw_dissociation_frame, frames=len(d_min_values), interval=1000 / fps, cache_frame_data=False)
anim_dissoc.save(out_dir / "anim_dissociation.gif", writer=PillowWriter(fps=fps), dpi=95)
plt.close(fig)
print("Animation sauvegardée :", out_dir / "anim_dissociation.gif")

# %%
# prédiction single-frame sur les 8 flips de signe PCA vs des rotations aléatoires
torch.manual_seed(123)
n_rand_rot = 24

# on crée des rotations SO(3)
frames_rand_rot = torch.stack([torch.linalg.qr(torch.randn(3, 3))[0] for _ in range(n_rand_rot)])

# on réutilise preds_Aref pour les 8 flips 
n_flips = len(frames_Aref)
flip_values_D = mean_y + std_y * preds_Aref.sum(dim=1).squeeze(-1).numpy()
# prédictions pour les frames aléatoires
preds_Arand_rot = predict_pca_manual(A, frames_rand_rot)
rand_values_D = mean_y + std_y * preds_Arand_rot.sum(dim=1).squeeze(-1).numpy()

all_values_D = np.concatenate([flip_values_D, rand_values_D])
all_frames = torch.concat([frames_Aref, frames_rand_rot], dim=0)
all_positions = (pos_A_centered @ all_frames).numpy()
radius = float(np.linalg.norm(all_positions, axis=2).max()) + 0.35

fig = plt.figure(figsize=(11.5, 4.4), layout="constrained")
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax_rot = fig.add_subplot(1, 2, 2)

pad = 0.2 * float(all_values_D.max() - all_values_D.min()) + 1e-3



def draw_rotation_frame(i):
    ax3d.clear()
    # molécules + axes fixes (sans labels)
    setup_3d_axes(ax3d, (-radius, radius), (-radius, radius), (-radius, radius), zoom=1.45)
    axis_len = 0.85 * radius
    for vec in np.eye(3):
        ax3d.quiver(0, 0, 0, vec[0], vec[1], vec[2], length=axis_len, arrow_length_ratio=0.1, color="#8998a6", linewidth=1.3)
    draw_molecule_3d(ax3d, all_positions[i], A, color_A, size_scale=1.9)

    ax_rot.clear()
    # moyenne pour voir invariance
    ax_rot.axhline(flip_values_D.mean(), color="grey", ls="--", label="Mean of 8 sign flips")

    # sépare les flips des rotations
    ax_rot.axvline(n_flips - 0.5, color="grey", ls=":", lw=1)

    # on plot les flips jusqu'à 8
    ax_rot.plot(np.arange(min(i + 1, n_flips)), flip_values_D[: i + 1], "o-", ms=4, color=colors["Combined"], label="8 PCA sign flips")
    if i >= n_flips:
        # idem les PCA aléatoires
        ax_rot.plot(np.arange(n_flips, i + 1), rand_values_D[: i - n_flips + 1], "o-", ms=4, color=colors["Random frame"], label="Random rotations")
    else:
        ax_rot.plot([], [], "o-", ms=4, color=colors["Random frame"], label="Random rotations")
    ax_rot.set_xlim(-1, len(all_values_D))
    ax_rot.set_ylim(all_values_D.min() - pad, all_values_D.max() + pad)
    ax_rot.set_xlabel("Frame")
    ax_rot.set_ylabel(r"$\hat{\mu}$ (D)")
    ax_rot.set_title("PCA sign flips vs. random rotations")
    ax_rot.spines[["top", "right"]].set_visible(False)
    ax_rot.grid(axis="y", alpha=0.2, ls=":")
    ax_rot.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))


fps_rot = 4
anim_rot = FuncAnimation(fig, draw_rotation_frame, frames=len(all_values_D), interval=1000 / fps_rot, cache_frame_data=False)
anim_rot.save(out_dir / "anim_flips_vs_rotation.gif", writer=PillowWriter(fps=fps_rot), dpi=95)
plt.close(fig)

# %%
# multi pair

@torch.no_grad()
def first_graphnorm_shift(mol_AB, frames, n_A):
    """
    Calcule f_B * ||m_B - m_A||_2 qui est le décalage que B ramène à A au 1er graphnorm
    c'est moyenné sur les 8 repères
    """
    # dénominateur
    center = mol_AB.pos.mean(dim=0)
    H0 = faenet.molecule_embedding(mol_AB)
    blk0 = faenet.list_interaction_blocks[0]
    n_B = len(mol_AB.z) - n_A
    f_B = n_B / (n_A + n_B)

    shifts = []
    # récupère m_A et m_B en rebuildant le modèle
    for frame in frames:
        X_local = (mol_AB.pos - center) @ frame
        _, _, edge_index = adjacency_matrix(X_local, faenet.cutoff)
        E = faenet.edge_embedding(X_local, edge_index)
        filter_inputs = torch.concat([E, H0[edge_index[0]], H0[edge_index[1]]], dim=1)
        filters = blk0.filter_mlp(filter_inputs)
        M = torch.zeros_like(H0)
        M.index_add_(dim=0, index=edge_index[0], source=H0[edge_index[1]] * filters)

        m_A = M[:n_A].mean(dim=0)
        m_B = M[n_A:].mean(dim=0)
        shifts.append(f_B * (m_B - m_A).norm().item())

    return sum(shifts) / len(shifts)


# %%
torch.manual_seed(123)
faenet.frame_averaging = "none"

n_pairs = 50
perm = torch.randperm(len(dataset_test)).tolist()

ful_list, gn_list, pca_list, rand_list, noise_list, shift_list = [], [], [], [], [], []

# on refait le calcul sur plusieurs paires à distance fixe 
for i, j in zip(perm[:n_pairs], perm[n_pairs : 2 * n_pairs]):
    A_i = dataset_test[i]
    B_j = dataset_test[j]

    # direction aléatoire en 3D
    direction_ij = torch.randn(3)
    mol_AB, _ = add_two_molecules(A_i, B_j, direction_ij, d_far)

    n_A_i = len(A_i.z)

    # Référence (A_i isolé)
    frames_A = SFA(A_i.pos, full=True)
    preds_Aref_i = predict_pca_manual(A_i, frames_A)
    ref_sum_i = preds_Aref_i.mean(dim=0).sum().item()

    # 3 conditions + contrôle random frame
    frames_AB = SFA(mol_AB.pos, full=True)
    pred_Acomb = predict_pca_manual(mol_AB, frames_AB)[:, :n_A_i, :].mean(dim=0)
    pred_APCA = predict_pca_manual(A_i, frames_AB).mean(dim=0)
    pred_Agraph = predict_pca_manual(mol_AB, frames_A)[:, :n_A_i, :].mean(dim=0)
    pred_Arand = predict_pca_manual(A_i, random_eight_frames()).mean(dim=0)

    # Métriques 
    ful_list.append(std_y * abs(pred_Acomb.sum().item() - ref_sum_i))
    gn_list.append(std_y * abs(pred_Agraph.sum().item() - ref_sum_i))
    pca_list.append(std_y * abs(pred_APCA.sum().item() - ref_sum_i))
    rand_list.append(std_y * abs(pred_Arand.sum().item() - ref_sum_i))
    noise_list.append(std_y * preds_Aref_i.sum(dim=1).std().item())

    # shift moyen vu par le 1er GraphNorm
    shift_list.append(first_graphnorm_shift(mol_AB, frames_A, n_A_i))

# %%
# comparaison multi-paires 
results_pairs = {
    "Combined": ful_list,
    "GraphNorm": gn_list,
    "PCA": pca_list,
    "Random frame": rand_list,
    "SFA": noise_list,
}

fig, ax = plt.subplots(figsize=(8, 4.5))
rng = np.random.default_rng(123)

for idx, (name, values) in enumerate(results_pairs.items()):
    color = colors[name]
    style = {"color": color}
    ax.boxplot(
        values,
        positions=[idx],
        widths=0.5,
        showfliers=False,
        boxprops=style,
        medianprops=style,
        whiskerprops=style,
        capprops=style,
    )
    jitter = idx + rng.uniform(-0.15, 0.15, len(values))
    ax.scatter(jitter, values, s=14, color=color, alpha=0.5)

ax.axhline(0.156, color=color_red, ls="--", label=f"Test MAE ({0.156} D)")
ax.set_yscale("log")
ax.set_xticks(range(len(results_pairs)), results_pairs.keys())
ax.set_ylabel(r"$|\Delta\hat{\mu}_A|$ (D)")
ax.set_title("Drift across 50 random pairs (d = 35 Å)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2, ls=":")
ax.legend(frameon=False)

fig.savefig(out_dir / "fig2_pairs.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# Figure 3 : dérive GraphNorm vs shift moyen dans le 1er bloc GraphNorm
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(shift_list, gn_list, color=colors["GraphNorm"], s=22, alpha=0.7)
ax.set_xlabel(r"$f_B\,\|m_B - m_A\|_2$")
ax.set_ylabel("GraphNorm drift (D)")
ax.set_title("GraphNorm drift vs. first-layer mean shift")
ax.set_ylim(0, None)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2, ls=":")

fig.savefig(out_dir / "fig3_gn_shift.png", dpi=200, bbox_inches="tight")
plt.show()
