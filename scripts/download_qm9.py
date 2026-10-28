# %%

from torch_geometric.datasets import QM9
from pathlib import Path

# %%

root = Path(__file__).resolve().parents[1]
qm9_folder = root / "data" / "qm9"

dataset = QM9(root=str(qm9_folder))

# %%
