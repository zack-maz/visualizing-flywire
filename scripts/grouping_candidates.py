"""Try candidate grouping algorithms on FlyWire 783 and score them against the annotations.

Run: uv run --with numpy --with pandas --with scipy --with scikit-learn --with igraph --with leidenalg --with infomap python scripts/grouping_candidates.py
Writes analysis/grouping_candidates.csv. Needs public/data/ built first.
"""
import time, json
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score as nmi

N = 139248
buf = open("public/data/connections.bin", "rb").read()
oo = np.frombuffer(buf, np.uint32, N + 1, 0)
E = int(oo[-1])
op = np.frombuffer(buf, np.uint32, E, 8 * (N + 1))
ow = np.frombuffer(buf, np.uint16, E, 8 * (N + 1) + 8 * E)
row = np.repeat(np.arange(N), np.diff(oo))
A = sp.csr_matrix((ow.astype(float), (row, op)), shape=(N, N))
U = (A + A.T).tocsr()   # undirected, summed weights

g = json.load(open("public/data/groupings.json"))
gb = open("public/data/groupings.bin", "rb").read()
lab = {x["id"]: np.frombuffer(gb, np.uint8 if x["bytes"] == 1 else np.uint16, N, x["offset"]).astype(int) for x in g}
side = lab["side"]
REFS = ["super_class", "cell_class", "cell_type", "neuropil", "region", "ito_lee_hemilineage"]

def score(name, c, secs, note=""):
    c = np.asarray(c)
    sizes = np.bincount(c[c >= 0])
    out = {"method": name, "secs": round(secs, 1), "k": int((sizes > 0).sum()),
           "k>=50": int((sizes >= 50).sum()), "largest%": round(100 * sizes.max() / N, 1)}
    for r in REFS:
        m = (lab[r] > 0) & (c >= 0)
        out[r] = round(nmi(lab[r][m], c[m]), 2)
    # bilateral: share of neurons in clusters that hold both hemispheres (>=20% minority side)
    df = pd.DataFrame({"c": c, "s": side})
    lr = df[df.s.isin([1, 2])].groupby("c").s.agg(lambda s: min((s == 1).mean(), (s == 2).mean()))
    sz = df.groupby("c").size()
    out["bilateral%"] = round(100 * sz[lr[lr >= 0.2].index].sum() / N, 1)
    # home for unlabelled central neurons: NMI not defined; report their share in clusters with >=50 classed neurons
    out["note"] = note
    print(json.dumps(out), flush=True)
    return out

results = []
import igraph as ig, leidenalg as la
src, dst = U.nonzero(); keep = src < dst
G = ig.Graph(n=N, edges=np.stack([src[keep], dst[keep]], 1).tolist(), directed=False)
G.es["weight"] = np.asarray(U[src[keep], dst[keep]]).ravel()
for res in [1.0, 5.0]:
    t = time.time()
    p = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=res, seed=0)
    results.append(score(f"Leiden modularity (res {res})", p.membership, time.time() - t))

# Directed flow communities
try:
    from infomap import Infomap
    t = time.time()
    im = Infomap("--directed --two-level --silent --seed 1 --num-trials 1")
    ai, bi = A.nonzero()
    for a, b, w in zip(ai, bi, A.data):
        im.add_link(int(a), int(b), float(w))
    im.run()
    c = np.full(N, -1)
    for node in im.nodes:
        c[node.node_id] = node.module_id
    results.append(score("Infomap (directed flow)", c, time.time() - t))
except Exception as e:
    print("infomap failed", e)

# Connectivity-profile clustering: SVD of [inputs|outputs], then k-means / HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
F = sp.hstack([A, A.T.tocsr()]).tocsr(); F.data = np.log1p(F.data); F = normalize(F)
t = time.time(); Z = normalize(TruncatedSVD(64, random_state=0).fit_transform(F)); tsvd = time.time() - t
for k in [100, 1000]:
    t = time.time()
    c = MiniBatchKMeans(k, random_state=0, n_init=3, batch_size=8192).fit_predict(Z)
    results.append(score(f"k-means on partner SVD (k={k})", c, tsvd + time.time() - t))
P = np.fromfile("public/data/layout_partners.bin", np.float32).reshape(N, 3)
from sklearn.cluster import HDBSCAN
t = time.time()
connected = (np.diff(A.indptr) + np.diff(A.tocsc().indptr)) > 0
c = np.full(N, -1); c[connected] = HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(P[connected])
results.append(score("HDBSCAN on partner UMAP (3D)", c, time.time() - t, f"noise {np.mean(c==-1):.0%}"))

# Spectral: normalised-Laplacian eigenvectors of the undirected graph, then k-means
from scipy.sparse.linalg import eigsh
t = time.time()
d = np.asarray(U.sum(1)).ravel(); dinv = 1 / np.sqrt(np.maximum(d, 1e-9))
Anorm = sp.diags(dinv) @ U @ sp.diags(dinv)
vals, vecs = eigsh(Anorm, k=64, which="LA")
c = MiniBatchKMeans(200, random_state=0, n_init=3).fit_predict(normalize(vecs))
results.append(score("Spectral (64 eigvecs, k=200)", c, time.time() - t))

# Core-periphery: k-core shell on the undirected graph
t = time.time()
core = np.array(G.coreness())
bins = np.digitize(core, np.percentile(core, [50, 80, 95, 99]))
results.append(score("k-core shells (5 bands)", bins, time.time() - t, f"max core {core.max()}"))

pd.DataFrame(results).to_csv("analysis/grouping_candidates.csv", index=False)
