"""Build the compact files the viewer loads from the raw FlyWire 783 downloads.

Inputs (data/raw/):
  neuron_annotations.tsv          flywire_annotations Supplemental_file1 (positions, classes, lineages)
  neuron_neuropil_counts.parquet  from scripts/neuropil_counts.py (root_id, role in/out, neuropil, side, n)
  neuropils/*.obj, fafb14_volume_raw.obj   FAFB14 meshes in nm

Outputs (public/data/):
  neurons.bin      float32 xyz (N*3): anatomical position (the neuron's anchor point)
  neurons.json     count, release, root ids, neuropils (name, region)
  groupings.bin    per grouping, one code per neuron (uint8, or uint16 when > 255 values); offsets in groupings.json
  groupings.json   [{id, label, row, none, sided, values, counts, offset, bytes}] — value 0 is "none / unlabelled"
  layout_soma.bin  float32 xyz (N*3): cell body position (anchor position where the soma is outside the brain)
  layout_flat.bin  float32 xyz (N*3): UMAP 2D of each neuron's input and output neuropil profile
  meshes.bin       per mesh: float32 xyz vertices then uint32 triangle indices
  meshes.json      offsets into meshes.bin

World space: micrometres, centred on the brain volume, +Y dorsal, +Z anterior, +X = the fly's left.

Left/right: the FAFB images were mirrored left/right during acquisition. FlyWire corrected the
annotations (`side` is the fly's real side) but not the image-space data: coordinates, meshes and
the mesh/synapse-table neuropil names (ME_L, side="left") still follow the mirrored image.
Here we mirror x back and swap the L/R neuropil names so everything is biologically correct.
See https://codex.flywire.ai/faq.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("public/data")
OUT.mkdir(parents=True, exist_ok=True)

VOXEL_NM = np.array([4.0, 4.0, 40.0])
# Ordered roughly by how the brain is organised: vision, then central, then in/out of the brain.
SUPER_CLASSES = [
    "optic", "visual_projection", "visual_centrifugal", "central",
    "sensory", "ascending", "descending", "motor", "endocrine",
]
UNASSIGNED = "Unassigned"


def load_obj(path: Path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def to_world(nm: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """nm (FAFB axes) -> micrometres, centred, +Y dorsal, +Z anterior, x mirrored back to biology."""
    w = (nm - origin) / 1000.0
    return np.stack([-w[:, 0], -w[:, 1], -w[:, 2]], axis=1)


def bio_name(image_name: str) -> str:
    """FAFB image-space neuropil name -> the fly's real side (ME_L <-> ME_R)."""
    if image_name.endswith("_L"):
        return image_name[:-2] + "_R"
    if image_name.endswith("_R"):
        return image_name[:-2] + "_L"
    return image_name


# ---------------------------------------------------------------- meshes
brain_v, brain_f = load_obj(RAW / "fafb14_volume_raw.obj")
origin = (brain_v.min(0) + brain_v.max(0)) / 2

meshes = {"brain": (to_world(brain_v, origin), brain_f)}
for p in sorted((RAW / "neuropils").glob("*.obj")):
    v, f = load_obj(p)
    meshes[bio_name(p.stem)] = (to_world(v, origin), f)

mesh_index, chunks, offset = [], [], 0
for name, (v, f) in meshes.items():
    vb = v.astype(np.float32).tobytes()
    fb = f.astype(np.uint32).tobytes()
    mesh_index.append({
        "name": name,
        "vertexOffset": offset, "vertexCount": len(v),
        "indexOffset": offset + len(vb), "indexCount": f.size,
    })
    chunks += [vb, fb]
    offset += len(vb) + len(fb)
(OUT / "meshes.bin").write_bytes(b"".join(chunks))
(OUT / "meshes.json").write_text(json.dumps(mesh_index))
mesh_centre = {n: v.mean(0) for n, (v, _) in meshes.items() if n != "brain"}

# ---------------------------------------------------------------- neurons
ann = pd.read_csv(RAW / "neuron_annotations.tsv", sep="\t", dtype={"root_id": str}, low_memory=False)
# Annotations v3 split out 612 "sensory_ascending" neurons; they enter from the nerve cord, so keep them with ascending.
ann["super_class"] = ann.super_class.replace({"sensory_ascending": "ascending"})
unknown = set(ann.super_class.dropna()) - set(SUPER_CLASSES)
assert not unknown, f"unexpected super classes: {unknown}"
N = len(ann)

pos_nm = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
anat = to_world(pos_nm, origin)

# Synapse counts per neuron and mesh, by role. Image side -> fly's side.
counts = pd.read_parquet(RAW / "neuron_neuropil_counts.parquet")
counts = counts[counts.neuropil.notna() & ~counts.neuropil.isin(["outside_neuropil", "meshes"])]
counts["mesh"] = counts.neuropil + counts.side.map({"left": "_R", "right": "_L"}).fillna("")
assert set(counts.mesh) <= set(meshes), set(counts.mesh) - set(meshes)


def main_neuropil(c: pd.DataFrame) -> pd.Series:
    """Neuropil with the most synapses, per neuron; Unassigned where there are none."""
    per = c.groupby(["root_id", "mesh"], as_index=False).n.sum()
    top = per.sort_values("n", ascending=False).drop_duplicates("root_id").set_index("root_id").mesh
    return ann.root_id.map(top).fillna(UNASSIGNED)


ann["neuropil"] = main_neuropil(counts).to_numpy()
ann["in_neuropil"] = main_neuropil(counts[counts.role == "in"]).to_numpy()
ann["out_neuropil"] = main_neuropil(counts[counts.role == "out"]).to_numpy()

# Sanity check: neurons whose main neuropil is X should sit near mesh X (catches L/R or axis flips).
for name in ["ME_L", "ME_R", "AL_L", "AL_R", "EB"]:
    sel = ann.neuropil.to_numpy() == name
    d = np.linalg.norm(np.median(anat[sel], 0) - mesh_centre[name])
    print(f"{name}: {sel.sum():6d} neurons, median-to-mesh-centre {d:6.1f} um")
# ...and a neuron annotated on the fly's left should mostly use left neuropils (catches a missed L/R swap).
for side, suf in [("left", "_L"), ("right", "_R")]:
    sided = ann[(ann.side == side) & ann.neuropil.str.match(r".*_[LR]$")]
    agree = sided.neuropil.str.endswith(suf).mean()
    print(f"annotated {side}: {agree:.0%} have a main neuropil ending {suf}")
    assert agree > 0.8
# Inputs and outputs mostly land in the same neuropil for local neurons, less for projection neurons.
print(f"main input neuropil == main output neuropil: {(ann.in_neuropil == ann.out_neuropil).mean():.0%}")

# ---------------------------------------------------------------- regions (Ito et al. 2014 neuropil super groups)
NP_SUPER_GROUP = {
    **dict.fromkeys(["ME", "LO", "LOP", "AME", "LA"], "OL"),
    **dict.fromkeys(["MB_CA", "MB_PED", "MB_VL", "MB_ML"], "MB"),
    "LH": "LH", "AL": "AL",
    **dict.fromkeys(["SLP", "SIP", "SMP"], "SNP"),
    **dict.fromkeys(["CRE", "SCL", "ICL", "IB", "ATL"], "INP"),
    **dict.fromkeys(["LAL", "BU", "GA"], "LX"),
    **dict.fromkeys(["FB", "EB", "PB", "NO"], "CX"),
    **dict.fromkeys(["VES", "EPA", "GOR", "SPS", "IPS"], "VMNP"),
    **dict.fromkeys(["AOTU", "AVLP", "PVLP", "PLP", "WED"], "VLNP"),
    **dict.fromkeys(["AMMC", "FLA", "CAN", "PRW", "SAD", "GNG"], "PENP"),
}


def np_base(name: str) -> tuple[str, str]:
    return (name[:-2], name[-2:]) if name[-2:] in ("_L", "_R") else (name, "")


neuropil_names = sorted(mesh_centre)
# A region is split by side like its neuropils, unless it holds unpaired midline neuropils (CX, PENP).
midline = {NP_SUPER_GROUP[np_base(n)[0]] for n in neuropil_names if not np_base(n)[1]}


def region_of(name: str) -> str:
    if name == UNASSIGNED:
        return UNASSIGNED
    base, side = np_base(name)
    group = NP_SUPER_GROUP[base]
    return group if group in midline else group + side


ann["region"] = ann.neuropil.map(region_of)

# ---------------------------------------------------------------- soma positions and clusters
has_soma = ann.soma_x.notna().to_numpy()
soma_nm = ann[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * VOXEL_NM
soma = np.where(has_soma[:, None], to_world(np.nan_to_num(soma_nm), origin), anat)
print(f"soma position: {has_soma.mean():.1%} of neurons; the rest use their anchor position")

from sklearn.cluster import KMeans

SOMA_CLUSTERS = 40
km = KMeans(SOMA_CLUSTERS, random_state=0, n_init=4).fit(soma[has_soma])
# Name each cluster after the nearest neuropil, e.g. "Rind near AVLP_L"; order by size.
names = []
for c in km.cluster_centers_:
    near = min(neuropil_names, key=lambda n: np.linalg.norm(mesh_centre[n] - c))
    name, k = f"near {near}", 2
    while name in names:
        name, k = f"near {near} ({k})", k + 1
    names.append(name)
soma_cluster = np.full(N, "", dtype=object)
soma_cluster[has_soma] = np.asarray(names, dtype=object)[km.labels_]
ann["soma_cluster"] = soma_cluster

# ---------------------------------------------------------------- groupings
# Value 0 is always "none": unlabelled, unassigned, no soma in the brain.
LINEAGE_NONE = "No hemilineage"
GROUPINGS = [
    # id, label, row, column (or series), none label, sided (already one value per hemisphere), fixed order
    ("side", "Hemisphere", "where", ann.side.replace({"na": np.nan}), "Side unknown", True, ["left", "right", "center"]),
    ("region", "Region", "where", ann.region, "No main neuropil", True, None),
    ("neuropil", "Neuropil", "where", ann.neuropil, "No main neuropil", True, None),
    ("soma_cluster", "Soma position", "where", ann.soma_cluster, "No soma in the brain", False, None),
    ("in_neuropil", "Input neuropil", "where", ann.in_neuropil, "No input synapses", True, None),
    ("out_neuropil", "Output neuropil", "where", ann.out_neuropil, "No output synapses", True, None),
    ("flow", "Flow", "what", ann.flow, "No flow", False, ["afferent", "intrinsic", "efferent"]),
    ("super_class", "Super class", "what", ann.super_class, "No super class", False, SUPER_CLASSES),
    ("cell_class", "Cell class", "what", ann.cell_class, "No cell class", False, None),
    ("cell_sub_class", "Cell sub class", "what", ann.cell_sub_class, "No sub class", False, None),
    ("cell_type", "Cell type", "what", ann.cell_type, "Untyped", False, None),
    ("hemibrain_type", "Hemibrain type", "what", ann.hemibrain_type, "No hemibrain match", False, None),
    ("ito_lee_hemilineage", "Hemilineage (Ito/Lee)", "lineage", ann.ito_lee_hemilineage, LINEAGE_NONE, False, None),
    ("hartenstein_hemilineage", "Hemilineage (Hartenstein)", "lineage", ann.hartenstein_hemilineage, LINEAGE_NONE, False, None),
]
g_meta, g_chunks, g_offset = [], [], 0
for gid, label, row, col, none, sided, order in GROUPINGS:
    s = pd.Series(np.asarray(col, dtype=object)).replace({UNASSIGNED: np.nan, "": np.nan})
    vc = s.value_counts()
    values = [v for v in order if v in vc.index] if order else list(vc.index)   # largest first
    assert set(values) == set(vc.index), (gid, set(vc.index) - set(values))
    code = s.map({v: i + 1 for i, v in enumerate(values)}).fillna(0).to_numpy().astype(np.int64)
    dtype = np.uint8 if len(values) < 256 else np.uint16
    blob = code.astype(dtype).tobytes()
    blob += b"\0" * (-len(blob) % 4)
    g_meta.append({
        "id": gid, "label": label, "row": row, "none": none, "sided": sided,
        "values": [""] + [str(v) for v in values],
        "counts": np.bincount(code, minlength=len(values) + 1).tolist(),
        "offset": g_offset, "bytes": np.dtype(dtype).itemsize,
    })
    g_chunks.append(blob)
    g_offset += len(blob)
    print(f"{gid:24s} {len(values):5d} values  {1 - (code == 0).mean():6.1%} labelled")

# ---------------------------------------------------------------- write
(OUT / "neurons.bin").write_bytes(anat.astype(np.float32).tobytes())
(OUT / "layout_soma.bin").write_bytes(soma.astype(np.float32).tobytes())
(OUT / "groupings.bin").write_bytes(b"".join(g_chunks))
(OUT / "groupings.json").write_text(json.dumps(g_meta, separators=(",", ":")))
(OUT / "neurons.json").write_text(json.dumps({
    "count": N,
    "release": 783,
    "rootIds": ann.root_id.tolist(),
    "neuropils": [{"name": n, "region": region_of(n)} for n in neuropil_names],
}, separators=(",", ":")))

# ---------------------------------------------------------------- flattened map
# Each neuron's input fractions and output fractions over neuropils, square-rooted (the Hellinger
# embedding: euclidean distance between these vectors compares the profiles), then UMAP to 2D.
import umap

prof = counts.pivot_table(index="root_id", columns=["role", "mesh"], values="n", aggfunc="sum", fill_value=0)
prof = prof.reindex(ann.root_id).fillna(0)
X = np.zeros((N, prof.shape[1]))
for role in ["in", "out"]:
    cols = [i for i, c in enumerate(prof.columns) if c[0] == role]
    block = prof.iloc[:, cols].to_numpy(float)
    X[:, cols] = np.sqrt(block / np.maximum(block.sum(1, keepdims=True), 1))
mapped = X.any(1)
print(f"flattened map: {mapped.sum():,} neurons with mapped synapses; running UMAP…", flush=True)
from sklearn.decomposition import PCA
# Many neurons have identical profiles (all synapses in one neuropil); exact duplicates make UMAP's
# neighbour search crawl, so add a tiny deterministic jitter, then reduce to 40 principal components.
jitter = np.random.default_rng(0).normal(0, 1e-3, (int(mapped.sum()), X.shape[1]))
Xp = PCA(40, random_state=0).fit_transform(X[mapped] + jitter)
xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0, low_memory=True).fit_transform(Xp)
xy -= np.median(xy, 0)
xy *= 800.0 / (np.percentile(xy, 99, 0) - np.percentile(xy, 1, 0)).max()
rng = np.random.default_rng(783)
flat = np.zeros((N, 3))
flat[mapped, :2] = xy
flat[:, 2] = rng.normal(0, 1.5, N)
# Neurons with no mapped synapses sit in a strip below the map.
lone = np.flatnonzero(~mapped)
flat[lone, 0] = (np.arange(len(lone)) % 200 - 100) * 1.2
flat[lone, 1] = xy[:, 1].min() - 40 - (np.arange(len(lone)) // 200) * 1.2

(OUT / "layout_flat.bin").write_bytes(flat.astype(np.float32).tobytes())

for f in sorted(OUT.iterdir()):
    print(f"{f.name:20s} {f.stat().st_size / 1e6:6.1f} MB")
