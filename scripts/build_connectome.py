"""Build the connection-based files the viewer loads, from the FlyWire 783 edge list.

Inputs (data/raw/):
  neuron_annotations.tsv            neuron order (row i = neuron i in every viewer file), flow, positions
  fafb_783_split_edgelist.feather   synapse counts per (pre, post, compartment)

Outputs (public/data/):
  connections.bin    CSR per direction over neuron pairs with >= 5 synapses (the FlyWire convention),
                     partners sorted by synapse count, strongest first:
                     out_offsets u32[N+1] | in_offsets u32[N+1] | out_partner u32[E] | in_partner u32[E]
                     | out_weight u16[E] | in_weight u16[E]
  layout_flow.bin    float32 xyz (N*3): sensory-to-motor axis (x = traversal step)
  layout_partners.bin float32 xyz (N*3): neurons with similar partners sit together (SVD + UMAP)
  flow_rank.bin      uint8 per neuron: traversal step (1 = afferent seed, 255 = never reached)
  connectome.json    counts and the step histogram

World space matches build_data.py: micrometres, +Y dorsal, +Z anterior, +X = the fly's left.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.sparse as sp

RAW = Path("data/raw")
OUT = Path("public/data")
MIN_SYN = 5
UNREACHED = 255

ann = pd.read_csv(RAW / "neuron_annotations.tsv", sep="\t", dtype={"root_id": str}, low_memory=False,
                  usecols=["root_id", "pos_x", "pos_y", "pos_z", "flow", "super_class", "cell_class"])
N = len(ann)
index = pd.Series(np.arange(N), index=ann.root_id.astype(np.int64))

# ---------------------------------------------------------------- edges
t = feather.read_table(RAW / "fafb_783_split_edgelist.feather", columns=["pre", "post", "count"])
pre = index.reindex(pc.cast(t["pre"], "int64").to_numpy()).to_numpy()
post = index.reindex(pc.cast(t["post"], "int64").to_numpy()).to_numpy()
count = t["count"].to_numpy()
keep = ~(np.isnan(pre) | np.isnan(post))
print(f"edge rows {len(count):,}; rows with both neurons annotated {keep.mean():.1%}")
# Sum the compartment-split rows into one count per neuron pair.
A = sp.coo_matrix((count[keep].astype(np.float64), (pre[keep].astype(np.int64), post[keep].astype(np.int64))),
                  shape=(N, N)).tocsr()
A.sum_duplicates()
print(f"pairs {A.nnz:,}; synapses {A.sum():,.0f}")
A.data[A.data < MIN_SYN] = 0
A.eliminate_zeros()
print(f"pairs with >= {MIN_SYN} synapses: {A.nnz:,}")


def csr_sorted(M: sp.csr_matrix):
    """Offsets, partners and weights with each row's partners sorted strongest first."""
    M = M.tocsr()
    M.sort_indices()
    offsets = M.indptr.astype(np.uint32)
    partner = M.indices.astype(np.uint32).copy()
    weight = M.data.copy()
    row = np.repeat(np.arange(N), np.diff(M.indptr))
    order = np.lexsort([-weight, row])
    return offsets, partner[order], np.minimum(weight[order], 65535).astype(np.uint16)


out_off, out_partner, out_w = csr_sorted(A)
in_off, in_partner, in_w = csr_sorted(A.T)
(OUT / "connections.bin").write_bytes(b"".join(x.tobytes() for x in [out_off, in_off, out_partner, in_partner, out_w, in_w]))

# ---------------------------------------------------------------- anatomy (same transform as build_data.py)
VOXEL_NM = np.array([4.0, 4.0, 40.0])


def load_obj_vertices(path: Path):
    with open(path) as f:
        return np.asarray([line.split()[1:4] for line in f if line.startswith("v ")], dtype=np.float64)


brain_v = load_obj_vertices(RAW / "fafb14_volume_raw.obj")
origin = (brain_v.min(0) + brain_v.max(0)) / 2
w = (ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM - origin) / 1000.0
anat = -w
rng = np.random.default_rng(783)

# ---------------------------------------------------------------- sensory-to-motor axis
# Deterministic version of the traversal in Schlegel et al. 2021 / Dorkenwald et al. 2024: start from
# afferent neurons; at each step a neuron joins once >= 30% of its input synapses come from neurons
# that have already joined. Step = when it joined.
THRESHOLD = 0.3
inputs = np.asarray(A.sum(0)).ravel()
rank = np.full(N, UNREACHED, dtype=np.int32)
joined = (ann.flow == "afferent").to_numpy().copy()
rank[joined] = 1
step = 1
while True:
    step += 1
    from_joined = A.T @ joined.astype(np.float64)
    new = ~joined & (inputs > 0) & (from_joined >= THRESHOLD * inputs)
    if not new.any() or step >= UNREACHED:
        break
    rank[new] = step
    joined |= new
hist = np.bincount(np.minimum(rank, UNREACHED), minlength=UNREACHED + 1)
print("steps:", {int(k): int(v) for k, v in enumerate(hist) if v})

for label, sel in [("Kenyon cells", ann.cell_class == "Kenyon_Cell"), ("ALPN", ann.cell_class == "ALPN"),
                   ("visual projection", ann.super_class == "visual_projection"),
                   ("descending", ann.super_class == "descending"), ("motor", ann.super_class == "motor")]:
    r = rank[sel.to_numpy()]
    print(f"  {label:18s} median step {np.median(r[r < UNREACHED]) if (r < UNREACHED).any() else '-'}"
          f"  unreached {np.mean(r == UNREACHED):.0%}")

# Within a step, order neurons by the mean step of their inputs so the axis reads as a gradient.
reached = rank < UNREACHED
W = A.T.tocsr()
mean_in = np.asarray(W @ np.where(reached, rank, 0).astype(float)).ravel() / np.maximum(np.asarray(W @ reached.astype(float)).ravel(), 1)
max_step = rank[reached].max()
SPACING = 900.0 / (max_step + 1)
frac = np.clip(mean_in - rank + 1, 0, 1) * 0.6 + rng.random(N) * 0.3
x = np.where(reached, rank - 1 + frac, max_step + 0.6 + rng.random(N) * 0.5) * SPACING
flow_xyz = np.stack([x - (max_step + 1) * SPACING / 2, anat[:, 1], anat[:, 0] * 0.5], 1)
(OUT / "layout_flow.bin").write_bytes(flow_xyz.astype(np.float32).tobytes())
(OUT / "flow_rank.bin").write_bytes(np.minimum(rank, UNREACHED).astype(np.uint8).tobytes())

# ---------------------------------------------------------------- partner map
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import umap

connected = (np.diff(A.indptr) + np.diff(A.tocsc().indptr)) > 0
F = sp.hstack([A, A.T.tocsr()]).tocsr()[connected]
F.data = np.log1p(F.data)
F = normalize(F)
emb = TruncatedSVD(64, random_state=0).fit_transform(F)
print(f"partner map: {connected.sum():,} connected neurons; running UMAP…", flush=True)
xyz = umap.UMAP(n_components=3, metric="cosine", n_neighbors=30, min_dist=0.2, random_state=0, low_memory=True).fit_transform(emb)
xyz -= np.median(xyz, 0)
xyz *= 700.0 / (np.percentile(xyz, 99, 0) - np.percentile(xyz, 1, 0)).max()
partners = np.zeros((N, 3))
partners[connected] = xyz
# Neurons with no strong connection sit in a flat disc below the map.
lone = np.flatnonzero(~connected)
k = np.arange(len(lone)) + 0.5
r = 1.0 * np.sqrt(k)
theta = k * np.deg2rad(137.508)
partners[lone] = np.stack([r * np.cos(theta), np.full(len(lone), xyz[:, 1].min() - 120), r * np.sin(theta)], 1)
(OUT / "layout_partners.bin").write_bytes(partners.astype(np.float32).tobytes())

(OUT / "connectome.json").write_text(json.dumps({
    "minSynapses": MIN_SYN, "edges": int(A.nnz), "synapses": int(A.sum()),
    "flowSteps": int(max_step), "flowHistogram": hist[: max_step + 1].tolist() + [int(hist[UNREACHED])],
    "partnerMapConnected": int(connected.sum()),
}))
for f in ["connections.bin", "layout_flow.bin", "layout_partners.bin", "flow_rank.bin"]:
    print(f"{f:20s} {(OUT / f).stat().st_size / 1e6:6.1f} MB")
