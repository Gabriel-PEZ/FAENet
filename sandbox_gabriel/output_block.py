# %%

import torch
from torch import nn


# %%
class OutputBlock(nn.Module):
    """
    Valable pour QM9
    """

    def __init__(self, H_dim, hidden_dim, node_dim):
        """
        H_dim dépend de jump connections : 
            - si pas de jump connections alors c'est simplement node_dim
            - sinon c'est L * node_dim si on prend chaque représentation des noeuds.

        """

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(H_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(node_dim, 1)
        )


    def forward(self, H_jump, H_last):
        """
        Si on utilise jumping connections, H = concaténation des vecteurs produits à 
        l'issu des InteractionBlock. 
        """

        Y = self.mlp(H_jump) # une prédiction d'énergie par atome 
        weights = self.weight_mlp(H_last) # poids qui dépendent uniquement de la dernière couche comme dans le papier 
        return Y, torch.sum(Y * weights)