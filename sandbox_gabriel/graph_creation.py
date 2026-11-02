# %%

import torch

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
    """
    if output_type == "invariant":
        return Y_local
    elif output_type == "equivariant":
        return Y_local @ U_star.t()
    else:
        raise ValueError("Unknown output_type")

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
        U_star = U[torch.randint(0, 8, size=(1,)).item()]
    return U_star

def adjacency_matrix(X_local, cutoff):

    D = torch.cdist(X_local, X_local)
    A = (D < cutoff).float()
    for i in range(X_local.shape[0]):
        A[i,i] = 0.0
    edge_index = torch.nonzero(A).t() # same shape que dans QM9 (2 x E)
    return A, D, edge_index