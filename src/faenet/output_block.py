# %%

import torch
from torch import nn
from torch_geometric.nn import global_add_pool

# %%
class OutputBlock(nn.Module):
    """
    Valable pour QM9
    """

    def __init__(self, H_dim, hidden_dim, node_dim, use_weights = True, xavier_init = False):
        """
        H_dim dépend de jump connections : 
            - si pas de jump connections alors c'est simplement node_dim
            - sinon c'est L * node_dim si on prend chaque représentation des noeuds.

        """

        super().__init__()
        self.use_weights = use_weights
        self.mlp = nn.Sequential(
            nn.Linear(H_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.use_weights:
            self.weight_mlp = nn.Sequential(
            nn.Linear(node_dim, 1)
        )
        else:
            self.weight_mlp = None
        if xavier_init:
            for layer in [self.mlp[0], self.mlp[2]]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            if self.use_weights:
                nn.init.xavier_uniform_(self.weight_mlp[0].weight)
                nn.init.zeros_(self.weight_mlp[0].bias)

    def forward(self, H_jump, H_last, batch_indices=None):
        """
        Si on utilise jumping connections, H = concaténation des vecteurs produits à 
        l'issu des InteractionBlock. Sinon H_jump = H_last.
        H_last utilisé uniquement pour les poids.
        """

        Y = self.mlp(H_jump) # une prédiction d'énergie par atome 
        if self.use_weights:
            weights = self.weight_mlp(H_last) # poids qui dépendent uniquement de la dernière couche comme dans le papier 
            return Y, global_add_pool(x=Y * weights, batch=batch_indices)
        else:
            return Y, global_add_pool(x=Y, batch=batch_indices)