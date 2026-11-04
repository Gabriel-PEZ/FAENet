import torch
from mendeleev import element


def build_atomic_features(dataset):

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

    return atomic_numbers_to_indices, properties_atomes_scaled