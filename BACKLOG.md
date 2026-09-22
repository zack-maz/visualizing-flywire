# Backlog: ideas not being built now

From the 2026-09-22 catalog of ways to organise the data. What *is* being built is in [PLAN.md](./PLAN.md).
Data key: ✅ already in `data/raw/` · 🔸 small computation · 🔻 needs connection data (now downloaded) or more.

## Groupings
- Neurotransmitter: predicted `top_nt` with a confidence slider (`top_nt_conf`) ✅; literature `known_nt` and neuropeptides (e.g. sNPF) ✅. Note: `top_nt` calls all Kenyon cells dopaminergic; they are cholinergic.
- Sex differences: `dimorphism`, `fru_dsx` ✅
- Nerve of entry/exit (`nerve`) ✅
- Unclassified central neurons' 20 neuropil-profile groups (from `analysis/unclassified_central_clusters.csv`) 🔸
- Clusters of similar neuropil profiles 🔸
- *Planned (PLAN §2): communities (Leiden, Infomap), connectivity types, hub level.* Not planned: spectral clustering (benchmarked weaker).
- Extras in `fafb_783_meta.feather`: cell_function, body_part_sensory / body_part_effector, dense-core-vesicle densities, volume, primary-dendrite width ✅

### Grouping algorithms benchmarked (2026-09-22, graph = pairs with 5+ synapses; `scripts/grouping_candidates.py`)
NMI against the annotations (higher NMI with more clusters is partly just more clusters; HDBSCAN scores exclude its noise):
| Method | Time | Groups (>= 50 neurons) | NMI region | NMI cell type | NMI hemilineage |
|---|---|---|---|---|---|
| Leiden modularity, res 1 | 31 s | 12 | **0.60** | 0.31 | 0.36 |
| Leiden, res 5 | 31 s | 61 | 0.51 | 0.38 | 0.43 |
| Infomap, directed flow | 11 s | 150 | 0.47 | 0.40 | 0.43 |
| k-means on partner SVD, k=100 / 1000 | 12–15 s | 100 / 976 | 0.49 / 0.40 | 0.51 / 0.61 | 0.44 / 0.48 |
| HDBSCAN on partner UMAP | 55 s | 470 (38% noise) | 0.46 | **0.69** | **0.56** |
| Spectral, 64 eigenvectors, k=200 | 24 s | 200 | 0.46 | 0.33 | 0.41 |
| k-core shells | 0.1 s | 5 bands | 0.11 | 0.19 | 0.14 |
Leiden and all others leave ~8k neurons with no strong connection as singletons. Most communities are one-hemisphere (wiring is mostly ipsilateral).

## Layouts
- Grid of small copies (small multiples): one mini brain per super class or transmitter ✅
- Timeline: an animated sequence position → region → neuropil → class ✅
- Spectral layout: Laplacian eigenmaps of the connectome (the alternative math layout) 🔻

## Colour by (continuous)
- Neuropil spread (entropy over neuropils) ✅ · synapse count ✅ · transmitter confidence ✅
- Soma-to-main-neuropil distance ✅
- Label agreement: does the neuropil profile predict the annotated class? (blue/red; shows odd neurons and likely annotation errors) 🔸
- Sensory-to-motor rank as a colour (the layout is being built; the colour ramp is not) 🔸

## Views beyond the point cloud
- Class × neuropil heatmap, clickable and linked to the 3D view ✅ (`analysis/class_x_neuropil_fraction.csv`)
- Sankey: super class → cell class → region → neuropil ✅
- Treemap / sunburst of the hierarchy ✅
- Label-agreement matrix across all label pairs ✅ (analysis section 1)
- Neuropil × transmitter bars ✅
- Local vs broad scatter (class spread vs member spread; the local / tiling / integrating kinds) ✅
- **Region-to-region connection matrix / chord diagram** 🔻 — the likely next step once connections are in
- Path finder: shortest route between two neurons or classes 🔻
- Left/right comparison panel ✅

## Found while building (2026-09-22)
- Soma clusters are named after the nearest neuropil *centre*, which can mislead (DA1 PN somata read "near Gall"). Name by nearest mesh surface or by the dominant hemilineage instead 🔸
- Partner list shows the same type several times (one row per partner neuron); group by type with summed synapses 🔸

## Removed at the user's request (2026-09-22)
- Balls layout (group by / split by, nested balls) and the fractal taxonomy tree. Don't re-add without asking.

## Analysis follow-ups
- With the input/output split (being built), redo the class analysis per direction: where classes receive vs send, and transmitter *released* per neuropil.
- Compare classes defined by region with classes defined by connections.
