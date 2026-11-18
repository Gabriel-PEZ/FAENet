import torch  ##CHANGE
from torch import nn  ##CHANGE
from torch_geometric.datasets import QM9  ##CHANGE
from torch_geometric.loader import DataLoader  ##CHANGE

from faenet.atomic_features import build_atomic_features  ##CHANGE
from faenet_samuel import FAENet  ##CHANGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_dataset(root="../data/qm9"):
    dataset = QM9(root=root)
    atomic_numbers_to_indices, properties_atomes_scaled = build_atomic_features(dataset)
    atomic_features = [atomic_numbers_to_indices, properties_atomes_scaled]
    return dataset, atomic_features

def load_data(dataset, target_idx, seed, batch_size):
    generator = torch.Generator().manual_seed(seed)
    idx_perm = torch.randperm(len(dataset), generator=generator)
    dataset_shuffled = dataset[idx_perm]

    # Plusieurs target de régression dans y.
    # 0 = moment dipolaire \mu
    idx_regression_target = target_idx

    size_train = int(0.85 * len(dataset))
    size_val = int(0.05 * len(dataset))
    size_test = size_train + size_val

    dataset_train = dataset_shuffled[:size_train]
    dataset_val = dataset_shuffled[size_train:size_test]
    dataset_test = dataset_shuffled[size_test:]

    target_train = dataset_train.y[:,idx_regression_target].to(device)
    mu_target_train = target_train.mean()
    std_target_train = target_train.std()

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    loaders = [train_loader, val_loader, test_loader]

    return loaders, mu_target_train, std_target_train

def build_model(atomic_features,
                default_config = {
    "atom_embedding_dim": 128, "hidden_dim_node": 400,
    "hidden_dim_edge": 480, "hidden_dim_output_block": 200,
    "node_dim": 400, "edge_dim": 480,
    "nb_gaussians_rbf": 100, "cutoff": 6.0,
    "num_iteration_block": 5, "jump_connection": True},
    **overrides):

    model_config = {**default_config, **overrides}
    faenet = FAENet(
    **model_config,
    atomic_numbers_to_indices=atomic_features[0],
    properties_atomes_scaled=atomic_features[1]).to(device)
    return faenet

def model_train(faenet, train_loader, mu_target_train, std_target_train, target_idx,  
                num_epochs=1, lr=1e-3, generator_seed=2003, criterion=nn.MSELoss()):
    optimizer = torch.optim.AdamW(
        faenet.parameters(),
        lr=lr
    )
    train_generator = torch.Generator().manual_seed(generator_seed)

    faenet.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()
            batch = batch.to(device)
            y_true = (
                batch.y[:, target_idx:target_idx+1] - mu_target_train
            ) / std_target_train
            _, y_pred = faenet(batch, sfa_generator=train_generator)

            # check de sécurité pour éviter broadcasting dans la loss
            assert y_true.shape == y_pred.shape

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} - Batch {batch_idx}: MSE = {loss.item():.6f}")

def model_eval(faenet, loader, mu_target_train, std_target_train, target_idx, generator_seed=42):
    faenet.eval()

    mae = 0.0
    n_samples = 0

    list_results = []

    eval_generator = torch.Generator().manual_seed(generator_seed)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            y_true = batch.y[:, target_idx:target_idx+1]

            _, y_pred_scaled = faenet(batch, sfa_generator=eval_generator) 
            y_pred = mu_target_train + std_target_train * y_pred_scaled

            assert y_true.shape == y_pred.shape

            diff = torch.abs(y_pred - y_true)

            mae += diff.sum().item()
            list_results.append(diff.squeeze(1).cpu())
            n_samples += len(y_true)

    mae = mae / n_samples
    print(f"MAE : {mae:.6f}")
    return mae, list_results