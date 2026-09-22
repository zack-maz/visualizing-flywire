"""Groupings found from the wiring alone: modules, flow modules, connectivity types, partner clusters, hub level.

Run after build_connectome.py:
  uv run --with numpy --with pandas --with scipy --with scikit-learn --with igraph --with leidenalg --with infomap \
    python scripts/build_clusters.py

Reads connections.bin, layout_partners.bin, groupings.json/bin. Appends groupings with row "wiring" to
groupings.json/bin (replacing any earlier wiring entries, so reruns are idempotent) and writes coreness.bin
(uint16 k-core coreness per neuron). Methods and expected numbers: PLAN.md §2, BACKLOG "Grouping algorithms benchmarked".
"""
import json
import time
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse as sp
from infomap import Infomap
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

OUT = Path("public/data")
N = 139248
MIN_GROUP = 20
NONE = "No strong connection or tiny module"
WIRING_IDS = ["leiden_coarse", "leiden_fine", "infomap", "conn_type", "conn_kmeans", "hub_band"]

# ---------------------------------------------------------------- inputs
buf = (OUT / "connections.bin").read_bytes()
oo = np.frombuffer(buf, np.uint32, N + 1, 0)
E = int(oo[-1])
op = np.frombuffer(buf, np.uint32, E, 8 * (N + 1))
ow = np.frombuffer(buf, np.uint16, E, 8 * (N + 1) + 8 * E)
A = sp.csr_matrix((ow.astype(float), (np.repeat(np.arange(N), np.diff(oo)), op)), shape=(N, N))
U = (A + A.T).tocsr()   # undirected, summed weights
connected = np.diff(U.indptr) > 0
print(f"{E:,} edges, {connected.sum():,} connected neurons")

g_meta = [g for g in json.loads((OUT / "groupings.json").read_text()) if g["id"] not in WIRING_IDS]
gb = (OUT / "groupings.bin").read_bytes()
lab = {g["id"]: np.frombuffer(gb, np.uint8 if g["bytes"] == 1 else np.uint16, N, g["offset"]).astype(int) for g in g_meta}
names = {g["id"]: g["values"] for g in g_meta}
nones = {g["id"]: g["none"] for g in g_meta}
g_chunks, offset = [], 0
for g in g_meta:   # re-pack the kept groupings so their offsets stay contiguous
    blob = gb[g["offset"]: g["offset"] + N * g["bytes"]]
    blob += b"\0" * (-len(blob) % 4)
    g["offset"] = offset
    g_chunks.append(blob)
    offset += len(blob)


def tidy(c):
    """-1 for unconnected neurons and members of groups under MIN_GROUP; otherwise unchanged."""
    c = np.where(connected, np.asarray(c), -1)
    sizes = np.bincount(c[c >= 0], minlength=1)
    return np.where((c >= 0) & (sizes[np.maximum(c, 0)] >= MIN_GROUP), c, -1)


def dominant(c, ref):
    """Per cluster: 'most common labelled value of grouping `ref` and its share', e.g. 'OL_R 81%'."""
    df = pd.DataFrame({"c": c, "v": lab[ref]})[c >= 0]
    size = df.groupby("c").size()
    top = df[df.v > 0].groupby("c").v.agg(lambda s: s.value_counts().idxmax())
    share = (df[df.v > 0].groupby("c").v.agg(lambda s: s.value_counts().iloc[0]) / size).fillna(0)
    return {k: f"{names[ref][top[k]]} {round(100 * share[k])}%" if k in top.index else nones[ref] for k in size.index}


def named(c, prefix, ref):
    """Order clusters by size and name them '<prefix><n> · <dominant ref value> <share>%'."""
    sizes = pd.Series(c[c >= 0]).value_counts()   # largest first
    dom = dominant(c, ref)
    values = [f"{prefix}{n} · {dom[k]}" for n, k in enumerate(sizes.index, start=1)]
    code = np.zeros(N, np.int64)
    code[c >= 0] = pd.Series(c[c >= 0]).map({k: n for n, k in enumerate(sizes.index, start=1)}).to_numpy()
    return code, values


def report(gid, code):
    sizes = np.bincount(code)[1:]
    row = {"id": gid, "values": len(sizes), ">=50": int((sizes >= 50).sum()),
           "largest%": round(100 * sizes.max() / N, 1), "none%": round(100 * (code == 0).mean(), 1)}
    for r in ["region", "cell_type", "ito_lee_hemilineage"]:
        m = (lab[r] > 0) & (code > 0)
        row[r] = round(nmi(lab[r][m], code[m]), 2)
    return row


results = {}   # id -> (label, code, values)

# ---------------------------------------------------------------- Leiden modules (undirected)
Ut = sp.triu(U, 1).tocoo()
G = ig.Graph(n=N, edges=np.stack([Ut.row, Ut.col], 1).tolist(), directed=False)
G.es["weight"] = Ut.data.tolist()
for gid, label, res, prefix in [("leiden_coarse", "Brain modules", 1.0, "M"), ("leiden_fine", "Brain modules (fine)", 5.0, "m")]:
    t = time.time()
    p = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=res, seed=0)
    results[gid] = (label, *named(tidy(p.membership), prefix, "region"))
    print(f"{gid}: {time.time() - t:.0f}s")

# ---------------------------------------------------------------- Infomap flow modules (directed)
t = time.time()
im = Infomap("--directed --two-level --silent --seed 1 --num-trials 1")
Ac = A.tocoo()
im.add_links(zip(Ac.row.tolist(), Ac.col.tolist(), Ac.data.tolist()))
im.run()
c = np.full(N, -1)
for node in im.nodes:
    c[node.node_id] = node.module_id
results["infomap"] = ("Flow modules", *named(tidy(c), "F", "region"))
print(f"infomap: {time.time() - t:.0f}s")

# ---------------------------------------------------------------- connectivity profiles
t = time.time()
F = sp.hstack([A, A.T.tocsr()]).tocsr()
F.data = np.log1p(F.data)
F = normalize(F)
Z = normalize(TruncatedSVD(64, random_state=0).fit_transform(F))

P = np.fromfile(OUT / "layout_partners.bin", np.float32).reshape(N, 3)
c = np.full(N, -1)
c[connected] = HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(P[connected])
noise = connected & (c < 0)
print(f"HDBSCAN noise {noise.sum() / connected.sum():.0%} of connected")
pre = report("conn_type (before noise assignment)", named(tidy(c), "C", "cell_type")[0])
knn = KNeighborsClassifier(15).fit(Z[c >= 0], c[c >= 0])
c[noise] = knn.predict(Z[noise])
results["conn_type"] = ("Connectivity types", *named(tidy(c), "C", "cell_type"))
print(f"conn_type: {time.time() - t:.0f}s")

t = time.time()
c = np.full(N, -1)
c[connected] = MiniBatchKMeans(1000, random_state=0, n_init=3, batch_size=8192).fit_predict(Z[connected])
results["conn_kmeans"] = ("Partner clusters", *named(tidy(c), "K", "cell_type"))
print(f"conn_kmeans: {time.time() - t:.0f}s")

# ---------------------------------------------------------------- hub level (k-core coreness)
core = np.array(G.coreness(), dtype=np.int64)
cuts = np.percentile(core, [50, 80, 95, 99]).astype(int)
band = np.digitize(core, cuts)   # 0 = periphery .. 4 = core
band_names = [
    f"Periphery, bottom 50% (k < {cuts[0]})",
    f"Middle, 50–80% (k {cuts[0]}–{cuts[1] - 1})",
    f"Inner, 80–95% (k {cuts[1]}–{cuts[2] - 1})",
    f"Near core, 95–99% (k {cuts[2]}–{cuts[3] - 1})",
    f"Core, top 1% (k ≥ {cuts[3]})",
]
code = np.where(connected, 5 - band, 0)   # value 1 = core first, value 5 = periphery
results["hub_band"] = ("Hub level", code, band_names[::-1])

# ---------------------------------------------------------------- write
rows = [pre]
for gid in WIRING_IDS:
    label, code, values = results[gid]
    dtype = np.uint8 if len(values) < 256 else np.uint16
    blob = code.astype(dtype).tobytes()
    blob += b"\0" * (-len(blob) % 4)
    g_meta.append({
        "id": gid, "label": label, "row": "wiring", "none": "No strong connection" if gid == "hub_band" else NONE, "sided": False,
        "values": [""] + values,
        "counts": np.bincount(code, minlength=len(values) + 1).tolist(),
        "offset": offset, "bytes": np.dtype(dtype).itemsize,
    })
    g_chunks.append(blob)
    offset += len(blob)
    rows.append(report(gid, code))

(OUT / "groupings.bin").write_bytes(b"".join(g_chunks))
(OUT / "groupings.json").write_text(json.dumps(g_meta, separators=(",", ":")))
(OUT / "coreness.bin").write_bytes(core.astype(np.uint16).tobytes())
print(pd.DataFrame(rows).to_string(index=False))
print("hub bands:", dict(zip(band_names[::-1], np.bincount(results["hub_band"][1], minlength=6)[1:].tolist())))
