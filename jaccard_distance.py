import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_pos_edges_py(path: str, threshold: float = 1e-5):
    """
    Load JSON of nested lists, return set of 'module_name:idx_str' where value > threshold.
    Pure Python, no torch.
    """
    with open(path) as f:
        raw = json.load(f)
    edges = set()
    def recurse(name, arr):
        if isinstance(arr, (int, float)):
            if arr > threshold:
                edges.add(name)
        else:
            # arr is list
            for idx, val in enumerate(arr):
                recurse(f"{name}:{idx}", val)
    for mod, arr in raw.items():
        recurse(mod, arr)
    return edges

def jaccard_distance(a: set, b: set) -> float:
    u = a | b
    if not u:
        return 0.0
    return 1 - len(a & b) / len(u)

# Gather files
files = sorted(glob.glob("data/attr_scores/logic_*.json"))
names = [Path(f).stem for f in files]
edge_sets = [load_pos_edges_py(f) for f in files]

# Compute matrix
n = len(edge_sets)
mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        mat[i, j] = jaccard_distance(edge_sets[i], edge_sets[j])

# Plot
fig, ax = plt.subplots()
im = ax.imshow(mat)

mid = n // 2
ax.axhline(mid-0.5, color='k', lw=1, ls='--')
ax.axvline(mid-0.5, color='k', lw=1, ls='--')

ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(names, rotation=90); ax.set_yticklabels(names)
ax.set_title("Pairwise Jaccard Distance")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("data/attr_scores/jaccard_distance_matrix.png", dpi=300)