"""Cell class organisation vs neuropil organisation in FlyWire 783.

Each neuron gets a synapse profile over neuropils (inputs + outputs, from neuron_neuropil_counts.parquet).
We ask how far the annotated cell classes line up with those profiles, where they don't, and
what the profiles say about the 23% of neurons with no cell class.

Run: uv run --with pandas --with pyarrow --with scikit-learn --with scipy python scripts/class_vs_neuropil.py
Writes analysis/*.csv and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.cluster import KMeans

RAW = Path("data/raw")
OUT = Path("analysis")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# Ito et al. 2014 neuropil super-categories (FlyWire names).
NP_GROUP = {
    **dict.fromkeys(["ME", "LO", "LOP", "AME", "LA"], "OL optic lobe"),
    **dict.fromkeys(["MB_CA", "MB_PED", "MB_VL", "MB_ML"], "MB mushroom body"),
    "LH": "LH lateral horn", "AL": "AL antennal lobe",
    **dict.fromkeys(["SLP", "SIP", "SMP"], "SNP superior"),
    **dict.fromkeys(["CRE", "SCL", "ICL", "IB", "ATL"], "INP inferior"),
    **dict.fromkeys(["LAL", "BU", "GA"], "LX lateral complex"),
    **dict.fromkeys(["FB", "EB", "PB", "NO"], "CX central complex"),
    **dict.fromkeys(["VES", "EPA", "GOR", "SPS", "IPS"], "VMNP ventromedial"),
    **dict.fromkeys(["AOTU", "AVLP", "PVLP", "PLP", "WED"], "VLNP ventrolateral"),
    **dict.fromkeys(["AMMC", "FLA", "CAN", "PRW", "SAD", "GNG"], "PENP periesophageal"),
}


def entropy(p, axis=-1):
    p = np.asarray(p, float)
    p = p / p.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0), axis=axis)


def cosine_rows(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    return X @ X.T


# ------------------------------------------------------------------ load
ann = pd.read_csv(RAW / "neuron_annotations.tsv", sep="\t", dtype={"root_id": str}, low_memory=False)
ann["super_class"] = ann.super_class.replace({"sensory_ascending": "ascending"})
ann["has_class"] = ann.cell_class.notna()
ann["cell_class_f"] = ann.cell_class.fillna("unclassified:" + ann.super_class)

cnt = pd.read_parquet(RAW / "neuron_neuropil_counts.parquet")
cnt = cnt[cnt.neuropil.notna() & ~cnt.neuropil.isin(["outside_neuropil", "meshes"])].copy()
cnt["root_id"] = cnt.root_id.astype(str)
# Side-collapsed neuropil (ME, not ME_L): the question is "which region", not "which hemisphere".
M = cnt.pivot_table(index="root_id", columns="neuropil", values="n", aggfunc="sum", fill_value=0)
NPS = list(M.columns)
ann = ann[ann.root_id.isin(M.index)].copy()   # neurons with >=1 synapse in a mapped neuropil
M = M.loc[ann.root_id]
X = M.to_numpy(float)
tot = X.sum(1)
F = X / tot[:, None]                           # per-neuron neuropil fractions
ann["primary_np"] = np.array(NPS)[X.argmax(1)]
ann["primary_group"] = ann.primary_np.map(NP_GROUP)
ann["np_entropy"] = entropy(X)                 # bits; 0 = all synapses in one neuropil
ann["n_syn"] = tot
print(f"{len(ann):,} neurons with mapped synapses, {len(NPS)} neuropils (side-collapsed)\n")

# ------------------------------------------------------------------ 1. how much does each label say about neuropil?
print("== 1. Label vs primary neuropil (neurons with the label only) ==")
rows = []
for label in ["super_class", "cell_class", "cell_sub_class", "ito_lee_hemilineage", "top_nt", "cell_type"]:
    d = ann[ann[label].notna()]
    rows.append({
        "label": label, "n_neurons": len(d), "n_values": d[label].nunique(),
        "NMI(label, primary_np)": normalized_mutual_info_score(d[label], d.primary_np),
        "NMI(label, np_group)": normalized_mutual_info_score(d[label], d.primary_group),
        "AMI(label, primary_np)": adjusted_mutual_info_score(d[label], d.primary_np),   # chance-corrected
    })
assoc = pd.DataFrame(rows)
print(assoc.round(3).to_string(index=False), "\n")
assoc.to_csv(OUT / "label_vs_neuropil.csv", index=False)

# Circularity check: optic cell classes are named after neuropils (ME>LO). Repeat for central brain only.
cen = ann[(ann.super_class == "central") & ann.has_class]
print(f"central brain only, classed ({len(cen):,}): NMI(cell_class, primary_np) = "
      f"{normalized_mutual_info_score(cen.cell_class, cen.primary_np):.3f}")
for label in ["ito_lee_hemilineage", "cell_type"]:
    d = ann[(ann.super_class == "central") & ann[label].notna()]
    print(f"central brain, {label} ({len(d):,}): NMI = {normalized_mutual_info_score(d[label], d.primary_np):.3f}")
print()

# ------------------------------------------------------------------ 2. class x neuropil matrix, specificity
print("== 2. Per-class neuropil spread (local vs broadcast classes) ==")
C = pd.DataFrame(X, columns=NPS).groupby(ann.cell_class_f.to_numpy()).sum()
Cn = C.div(C.sum(1), axis=0)
cls_stats = pd.DataFrame({
    "n_neurons": ann.groupby("cell_class_f").size(),
    "syn_per_neuron": ann.groupby("cell_class_f").n_syn.median(),
    "class_np_entropy_bits": entropy(C.to_numpy()),                    # spread of the class as a whole
    "median_neuron_np_entropy": ann.groupby("cell_class_f").np_entropy.median(),  # spread of its members
    "top_neuropil": Cn.idxmax(1),
    "top_share": Cn.max(1),
    "n_np_for_80pct": (np.sort(Cn.to_numpy(), 1)[:, ::-1].cumsum(1) < 0.8).sum(1) + 1,
})
# If members each use few neuropils but the class covers many, the class tiles space (a population
# of local neurons); if members are individually broad, the class is made of integrators.
cls_stats["tiling_ratio"] = cls_stats.class_np_entropy_bits - cls_stats.median_neuron_np_entropy
cls_stats = cls_stats[cls_stats.n_neurons >= 20].sort_values("class_np_entropy_bits")
print(cls_stats.round(2).to_string(), "\n")
cls_stats.to_csv(OUT / "class_spread.csv")
Cn.to_csv(OUT / "class_x_neuropil_fraction.csv")

print("== 2b. Per-neuropil class dominance ==")
N = C.T                                         # neuropil x class synapses
Nn = N.div(N.sum(1), axis=0)
np_stats = pd.DataFrame({
    "synapses": N.sum(1),
    "effective_n_classes": 2 ** entropy(N.to_numpy()),   # perplexity
    "dominant_class": Nn.idxmax(1),
    "dominant_share": Nn.max(1),
    "unclassified_share": Nn[[c for c in Nn.columns if c.startswith("unclassified")]].sum(1),
    "group": [NP_GROUP[n] for n in N.index],
}).sort_values("effective_n_classes")
print(np_stats.round(2).to_string(), "\n")
np_stats.to_csv(OUT / "neuropil_dominance.csv")
r, p = spearmanr(np_stats.synapses, np_stats.effective_n_classes)
print(f"neuropil size vs class diversity: Spearman r={r:.2f} p={p:.3f}\n")

# ------------------------------------------------------------------ 3. co-clustering: do classes group like neuropils?
print("== 3. Clustering classes by neuropil profile vs Ito neuropil groups ==")
big = cls_stats.index
P = Cn.loc[big].to_numpy()
Z = linkage(np.sqrt(P), "average", metric="euclidean")   # Hellinger distance
k = 11
lab = fcluster(Z, k, "maxclust")
dom_group = pd.Series([NP_GROUP[n] for n in cls_stats.top_neuropil], index=big)
print(f"ARI(class clusters k={k}, class's dominant Ito group) = {adjusted_rand_score(lab, dom_group):.3f}")
for c in sorted(set(lab)):
    members = list(big[lab == c])
    prof = Cn.loc[members].mean().sort_values(ascending=False)
    print(f"  cluster {c}: {', '.join(prof.index[:4])} <- {', '.join(members[:10])}{' ...' if len(members) > 10 else ''}")
print()

# ------------------------------------------------------------------ 4. per-class coherence
print("== 4. Class coherence: are members' neuropil profiles alike? ==")
S = np.sqrt(F)                                  # Hellinger embedding: euclid dist = Hellinger*sqrt2
coh = []
for c, idx in ann.groupby("cell_class_f").indices.items():
    if len(idx) < 20:
        continue
    samp = idx if len(idx) <= 800 else rng.choice(idx, 800, replace=False)
    within = cosine_rows(S[samp])[np.triu_indices(len(samp), 1)].mean()
    other = rng.choice(np.setdiff1d(np.arange(len(S)), idx), 800, replace=False)
    between = (S[samp] / np.linalg.norm(S[samp], axis=1, keepdims=True)) @ \
              (S[other] / np.linalg.norm(S[other], axis=1, keepdims=True)).T
    coh.append({"cell_class": c, "n": len(idx), "within_sim": within, "between_sim": between.mean(),
                "coherence": within - between.mean()})
coh = pd.DataFrame(coh).sort_values("coherence")
print(coh.round(3).to_string(index=False), "\n")
coh.to_csv(OUT / "class_coherence.csv", index=False)

# ------------------------------------------------------------------ 5. predict class from neuropil profile
print("== 5. Can the neuropil profile alone predict the label? (5-fold CV, logistic regression) ==")
feat = np.hstack([S, np.log1p(tot)[:, None] / 10])
def cv_acc(mask, y, min_n=20):
    y = pd.Series(y)[mask]
    keep = y.map(y.value_counts()) >= min_n
    idx = np.flatnonzero(mask)[keep.to_numpy()]
    y = y[keep].to_numpy()
    if len(idx) > 40000:
        sel = rng.choice(len(idx), 40000, replace=False)
        idx, y = idx[sel], y[sel]
    clf = LogisticRegression(max_iter=2000, C=10)
    pred = cross_val_predict(clf, feat[idx], y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    base = pd.Series(y).value_counts(normalize=True).iloc[0]
    return (pred == y).mean(), base, len(set(y)), pd.DataFrame({"true": y, "pred": pred})
res = {}
for name, mask, y in [
    ("super_class (all)", np.ones(len(ann), bool), ann.super_class),
    ("cell_class (all classed)", ann.has_class.to_numpy(), ann.cell_class),
    ("cell_class (central only)", (ann.has_class & (ann.super_class == "central")).to_numpy(), ann.cell_class),
    ("hemilineage (central)", (ann.ito_lee_hemilineage.notna() & (ann.super_class == "central")
                               & (ann.ito_lee_hemilineage != "putative_primary")).to_numpy(), ann.ito_lee_hemilineage),
    ("top_nt (central)", (ann.top_nt.notna() & (ann.super_class == "central")).to_numpy(), ann.top_nt),
]:
    acc, base, k_, df = cv_acc(mask, y.to_numpy())
    res[name] = df
    print(f"  {name:28s} acc {acc:.3f}  (majority baseline {base:.3f}, {k_} classes)")
cm = pd.crosstab(res["cell_class (all classed)"].true, res["cell_class (all classed)"].pred)
per_class = (np.diag(cm.reindex(columns=cm.index, fill_value=0)) / cm.sum(1)).sort_values()
print("\n  worst-recalled classes:", per_class.head(8).round(2).to_dict())
conf = cm.where(~np.eye(len(cm), dtype=bool) if cm.shape[0] == cm.shape[1] else cm.notna(), 0)
pairs = conf.stack().sort_values(ascending=False).head(10)
print("  top confusions (true -> pred):", {f"{a}->{b}": int(v) for (a, b), v in pairs.items() if a != b})
print()

# ------------------------------------------------------------------ 6. unclassified neurons: what are they?
print("== 6. Unclassified neurons: nearest classed profile + unsupervised groups ==")
un = ~ann.has_class.to_numpy()
cls_mask = ann.has_class.to_numpy()
clf = LogisticRegression(max_iter=2000, C=10).fit(feat[cls_mask], ann.cell_class[cls_mask])
proba = clf.predict_proba(feat[un])
guess = pd.DataFrame({"super_class": ann.super_class[un].to_numpy(),
                      "nearest_class": clf.classes_[proba.argmax(1)], "conf": proba.max(1)})
print("  share of unclassified neurons confidently (>0.8) matching an existing class:",
      f"{(guess.conf > 0.8).mean():.2f}")
print(guess.groupby("super_class").apply(
    lambda g: g[g.conf > 0.8].nearest_class.value_counts().head(3).to_dict(), include_groups=False).to_string())
uc = ann[un & (ann.super_class == "central").to_numpy()]
Su = S[un & (ann.super_class == "central").to_numpy()]
km = KMeans(20, n_init=4, random_state=0).fit(Su)
uc = uc.assign(cluster=km.labels_)
cent = pd.DataFrame(km.cluster_centers_ ** 2, columns=NPS)
print(f"\n  {len(uc):,} unclassified central neurons -> 20 k-means groups on neuropil profile:")
summary = []
for c in range(20):
    g = uc[uc.cluster == c]
    top = cent.loc[c].sort_values(ascending=False)
    hl = g.ito_lee_hemilineage.dropna()
    summary.append({"cluster": c, "n": len(g), "top_neuropils": ", ".join(f"{k}:{v:.2f}" for k, v in top.head(3).items()),
                    "top_nt": g.top_nt.value_counts(normalize=True).head(1).round(2).to_dict(),
                    "hemilineage_purity": hl.value_counts(normalize=True).iloc[0].round(2) if len(hl) > 10 else None,
                    "n_cell_types": g.cell_type.nunique()})
summary = pd.DataFrame(summary).sort_values("n", ascending=False)
print(summary.to_string(index=False))
summary.to_csv(OUT / "unclassified_central_clusters.csv", index=False)
print(f"  NMI(kmeans cluster, hemilineage) = "
      f"{normalized_mutual_info_score(uc.cluster[uc.ito_lee_hemilineage.notna()], uc.ito_lee_hemilineage.dropna()):.3f}")
print()

# ------------------------------------------------------------------ 7. left/right symmetry of class profiles
print("== 7. Left/right symmetry per class (sided neuropil profiles, mirrored) ==")
sided = cnt.copy()
sided["np_side"] = sided.neuropil + sided.side.map({"left": "_R", "right": "_L"}).fillna("")   # image -> fly side
sided = sided.merge(ann[["root_id", "cell_class_f", "side"]], on="root_id")
sided = sided[sided.side_y.isin(["left", "right"])]
def mirror(n):
    return n[:-2] + {"_L": "_R", "_R": "_L"}[n[-2:]] if n[-2:] in ("_L", "_R") else n
sym = []
for c, g in sided.groupby("cell_class_f"):
    L = g[g.side_y == "left"].groupby("np_side").n.sum()
    R = g[g.side_y == "right"].groupby("np_side").n.sum()
    if L.sum() < 1000 or R.sum() < 1000:
        continue
    R = R.rename(index=mirror)
    v = pd.concat([L, R], axis=1).fillna(0).to_numpy()
    v = v / v.sum(0)
    sym.append({"cell_class": c, "hellinger_LR": np.sqrt(0.5 * ((np.sqrt(v[:, 0]) - np.sqrt(v[:, 1])) ** 2).sum()),
                "n_left": (g[g.side_y == "left"].root_id.nunique()), "n_right": g[g.side_y == "right"].root_id.nunique()})
sym = pd.DataFrame(sym).sort_values("hellinger_LR", ascending=False)
print(sym.head(10).round(3).to_string(index=False))
print(f"  median L/R Hellinger across classes: {sym.hellinger_LR.median():.3f}\n")
sym.to_csv(OUT / "class_lr_symmetry.csv", index=False)

# ------------------------------------------------------------------ 8. neurotransmitter by neuropil
print("== 8. Neurotransmitter identity of the neurons using each neuropil (synapse-weighted) ==")
# Counts mix inputs and outputs, so this is "which NT neurons meet here", not "what is released here".
# Prefer literature known_nt over the image-based prediction: the predictor calls every Kenyon cell
# dopaminergic, though they are cholinergic.
KNOWN = {"acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine"}
known = ann.known_nt.fillna("").str.split(r"[;,]").str[0].str.strip()
nt_label = known.where(known.isin(KNOWN), ann.top_nt).fillna("unknown")
print(f"  known_nt overrides the prediction for {(known.isin(KNOWN) & (known != ann.top_nt)).sum():,} neurons")
nt = pd.DataFrame(X, columns=NPS).groupby(nt_label.to_numpy()).sum().T
ntn = nt.div(nt.sum(1), axis=0)
print(ntn.round(2).sort_values("acetylcholine").to_string(), "\n")
ntn.to_csv(OUT / "neuropil_nt_mix.csv")
