# %%
from pathlib import Path

import torch
from torch import nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

print(Path.cwd())

from faenet.atomic_features import build_atomic_features
from faenet.faenet import FAENet

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = QM9(root="../data/qm9")

# %%

atomic_numbers_to_indices, properties_atomes_scaled = build_atomic_features(dataset)
print(atomic_numbers_to_indices)
print(properties_atomes_scaled)

# %%

torch.manual_seed(7)
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
    frame_averaging="sfa",
).to(device)

# %% Dataset split

idx_regression_target = 0  # Dipole moment mu

split_generator = torch.Generator().manual_seed(2003)
indices = torch.randperm(len(dataset), generator=split_generator)

n_train = 110_000
n_val = 10_000

dataset_train = dataset[indices[:n_train]]
dataset_val = dataset[indices[n_train:n_train + n_val]]
dataset_test = dataset[indices[n_train + n_val:]]

# norm target sur train 
target_train = dataset_train.y[:, idx_regression_target].to(device)
mu_target_train = target_train.mean()
std_target_train = target_train.std()

# DataLoaders
shuffle_generator = torch.Generator().manual_seed(2007)

train_loader = DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=True,
    generator=shuffle_generator
)
val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

# %% 

checkpoint_dir = Path("checkpoints") / (
    f"qm9_{idx_regression_target}_{faenet.frame_averaging}"
)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(faenet.parameters(), lr=1e-3)

train_generator = torch.Generator().manual_seed(1977)

num_epochs = 100  # First experiment
best_val_mae = float("inf")

# %% 

for epoch in range(num_epochs):

    # train
    faenet.train()
    train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        y_true = (
            batch.y[:, idx_regression_target:idx_regression_target + 1]
            - mu_target_train
        ) / std_target_train

        _, y_pred = faenet(batch, sfa_generator=train_generator)

        assert y_true.shape == y_pred.shape

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(dataset_train)

    # val
    faenet.eval()
    val_mae = 0.0
    val_generator = torch.Generator().manual_seed(1972)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            _, y_pred_scaled = faenet(
                batch, sfa_generator=val_generator
            )

            y_pred = mu_target_train + std_target_train * y_pred_scaled
            y_true = batch.y[:, idx_regression_target:idx_regression_target + 1]

            val_mae += (y_pred - y_true).abs().sum().item()

    val_mae /= len(dataset_val)

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MAE: {val_mae:.6f}"
    )

    # Checkpoints
    is_best = val_mae < best_val_mae
    best_val_mae = min(best_val_mae, val_mae)

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": faenet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": best_val_mae,
        "frame_averaging": faenet.frame_averaging,
        "target_idx": idx_regression_target,
        "target_mean": mu_target_train.detach().cpu(),
        "target_std": std_target_train.detach().cpu(),
        "sfa_generator_state": train_generator.get_state(),
        "shuffle_generator_state": shuffle_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }

    torch.save(checkpoint, checkpoint_dir / "last.pt")

    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best.pt")

# %%

checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
faenet.load_state_dict(checkpoint["model_state_dict"])
faenet.frame_averaging = checkpoint["frame_averaging"]
faenet.eval()

# norm du training
mu = checkpoint["target_mean"].to(device)
std = checkpoint["target_std"].to(device)
target_idx = checkpoint["target_idx"]

test_generator = torch.Generator().manual_seed(1972)

test_mae = 0.0
n_samples = 0

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)

        _, y_pred_scaled = faenet(
            batch, sfa_generator=test_generator
        )

        y_pred = mu + std * y_pred_scaled
        y_true = batch.y[:, target_idx:target_idx + 1]

        test_mae += (y_pred - y_true).abs().sum().item()
        n_samples += y_true.shape[0]

test_mae /= n_samples

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best Val MAE: {checkpoint['best_val_mae']:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# %%
