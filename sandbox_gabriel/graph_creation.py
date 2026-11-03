# %%

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool

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

def rho2(Y_local, U_star, output_type):
    """
    Fonction pour exprimer la sortie du MLP du repère local au repère global
    (Equivariant pour les forces et invariant pour les énergies) 
    [pas utiliser pour QM9]
    """
    if output_type == "invariant":
        return Y_local
    elif output_type == "equivariant":
        return Y_local @ U_star.t()
    else:
        raise ValueError("Unknown output_type")

def SFA(X, batch_indices=None, k=None, generator=None):

    """
    X : molecule.pos (position x,y,z de tous les atomes de la molecule) ou 
        batch.pos (position x,y,z de tous les atomes de toutes les molécules du batch) 
    """

    if batch_indices is None:
        t = X.mean(axis=0)
        X_c = X - t 
        Sigma = X_c.t() @ X_c
    else:
        t = global_mean_pool(x=X, batch=batch_indices)
        X_c = X - t[batch_indices]
        # permet de calculer l'ensemble des produits x_i x_i^T pour ensuite les sommer et obtenir Sigma
        # ie X^T X = sum_i x_i x_i^T 
        flat_block_product = (X_c.unsqueeze(2) @ X_c.unsqueeze(1)).reshape(-1,9) 
        Sigma = global_add_pool(x=flat_block_product, batch=batch_indices).reshape(-1,3,3)

    _, eigenvectors = torch.linalg.eigh(Sigma)
    signs = torch.cartesian_prod(
        torch.tensor([-1.0, 1.0], device=X.device),
        torch.tensor([-1.0, 1.0], device=X.device),
        torch.tensor([-1.0, 1.0], device=X.device)
    ).unsqueeze(1)
    U = signs.unsqueeze(1) * eigenvectors.unsqueeze(0)

    B = U.shape[1]
    if k is not None:
        U_star = U[k]
    else:
        k_batch = torch.randint(0, 8, size=(B,), generator=generator).to(U.device)
        U_star = U[k_batch, torch.arange(B, device=U.device)]

    return U_star.squeeze(0) if batch_indices is None else U_star

def adjacency_matrix(X_local, cutoff):

    D = torch.cdist(X_local, X_local)
    A = (D < cutoff).float()
    for i in range(X_local.shape[0]):
        A[i,i] = 0.0
    edge_index = torch.nonzero(A).t() # same shape que dans QM9 (2 x E)
    return A, D, edge_index