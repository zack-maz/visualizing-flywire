# Plan: Fly Brain Explorer

Handoff for a fresh session. §1 is what exists, §2 is the algorithmic-groupings build (approved 2026-09-22,
built 2026-09-22; 2.3 still optional), §3–§5 are how to work on it. Ideas not being built: [BACKLOG.md](./BACKLOG.md).
Repo: https://github.com/zack-maz/visualizing-flywire (private).

## 1. Current state (verified 2026-09-22)

Vite + TypeScript + three.js viewer of all 139,248 FlyWire FAFB-783 neurons, one point each,
with a connection matrix between the groups of the current colour-by (§2b).

**Layouts** (keys 1–6): Where = Anatomical, Soma, Mirrored · Maps = Flattened (UMAP 2D of in/out neuropil
profile), Partners (SVD + UMAP 3D of connectivity), Sensory → motor (traversal step on x).
The Balls and Fractal layouts were built, then **removed at the user's request**; don't re-add without asking.

**Groupings** (20, generic): Where = hemisphere, region, neuropil, soma position (40 k-means clusters), input
neuropil, output neuropil · What = flow, super class, cell class, cell sub class, cell type, hemibrain type ·
Lineage = hemilineage Ito/Lee, Hartenstein · Wiring = brain modules (Leiden, coarse + fine), flow modules (Infomap),
connectivity types (HDBSCAN), partner clusters (k-means 1000), hub level (k-core bands). Each colours the cloud (top 10 values in colour, rest neutral,
legend hover-to-isolate / click-hide / double-click-solo) and is searchable in Focus (dims the rest, frames the group).

**Connections**: clicking a neuron lazy-loads `connections.bin` (36 MB), lists top 8 inputs/outputs in the card,
highlights partners in 3D (inputs = rings, outputs = discs, rest ghosted).

**Style**: brand tokens/fonts copied from `~/Documents/PROJECTS/BRAND` (`src/brand.css`, `public/fonts/`).
Dark only; blue (`--accent`) marks only the selected/focused thing; square corners, hairlines, mono uppercase labels.
Categorical palette order in `src/labels.ts` passes the dataviz validator for neighbouring rows.

### File map
| File | Role |
|---|---|
| `src/main.ts` | Layout registry (`LAYOUTS`), CPU-owned positions + transitions, colour-by legend, focus search, picking, card, labels, `window.flywire` (read by the probe) |
| `src/scene.ts` | Point shader: `position` (CPU), `aColor`, `aState` (HIDDEN/SHOWN/DIM/INPUT/OUTPUT); shell material |
| `src/data.ts` | Loaders: neurons, groupings, meshes; lazy `loadLayout(id)`, `loadFlowRank`, `loadConnections` |
| `src/matrix.ts` | Connection matrix panel (HTML table, computed client-side from `connections.bin`) |
| `src/labels.ts` | Palette, display names (`valueLabel`, `shortLabel`) |
| `scripts/neuropil_counts.py` | DuckDB pass over the remote synapse table → per (neuron, role in/out, neuropil) counts |
| `scripts/build_data.py` | meshes, anatomical positions, `groupings.json/bin`, `layout_soma.bin`, `layout_flat.bin` |
| `scripts/build_clusters.py` | Wiring groupings appended to `groupings.json/bin` (idempotent), `coreness.bin` (uint16 per neuron) |
| `scripts/build_connectome.py` | `connections.bin` (CSR both directions, ≥5 synapses), flow rank + `layout_flow.bin`, `layout_partners.bin` |
| `scripts/class_vs_neuropil.py` | Research: labels vs neuropil profiles → `analysis/*.csv` |
| `scripts/grouping_candidates.py` | Benchmark of clustering algorithms → `analysis/grouping_candidates.csv` (results in BACKLOG) |
| `scripts/ui_probe.py` | Headless Chromium check of every layout, search, focus, partners |

### Data format facts
- Neuron index i = row i of `data/raw/neuron_annotations.tsv`, everywhere.
- `groupings.json`: `[{id, label, row: where|what|lineage, none, sided, values, counts, offset, bytes}]`; value 0 = none;
  codes in `groupings.bin` (uint8, or uint16 above 255 values; offsets 4-byte aligned). The client builds the
  colour-by `<optgroup>`s from `row`; a new row value needs a heading in `fillSelect` in `main.ts`.
- World space: µm, +Y dorsal, +Z anterior, **+X = fly's left** (FAFB x is mirrored back; L/R neuropil names swapped).
- `data/raw/` is gitignored (~600 MB); `public/data/` is committed so the viewer runs from a clone.
- `npm run data` ≈ 3 min for the builds (+ downloads). Flat-map UMAP needs the jitter + PCA(40) in build_data.py:
  exact-duplicate profiles made plain UMAP run >25 min.

### Results worth remembering
- Sensory → motor: 22 steps; Kenyon/ALPN median step 3, visual projection 7, descending 4, motor 4; 4,576 unreached.
  (Descending is not late; reported to the user, threshold not tuned.)
- Role-split synapse counts reproduce every `analysis/*.csv` byte for byte.
- Wiring groupings (build_clusters.py, ~2.5 min): Leiden 12/61 groups ≥ 50 (NMI region 0.62/0.52), Infomap 150 ≥ 50
  (NMI cell type 0.39), connectivity types 529 ≥ 50 (NMI cell type 0.69 before noise assignment, 0.62 after; HDBSCAN
  noise 34%), partner clusters 910 ≥ 50 (0.63), hub bands k < 19 / 19–29 / 30–48 / 49–54 / ≥ 55. 7,979 neurons unconnected.
- Analysis headlines: NMI(cell class, main neuropil) 0.52; neuropil profile predicts central-brain class 99%,
  hemilineage 41%; 68% of central neurons have no cell class.

## 2. Algorithmic groupings (built 2026-09-22; kept as the spec)

The user approved all four recommendations from the benchmark (BACKLOG "Grouping algorithms benchmarked").
Spectral clustering was recommended against (worse and slower than Leiden/Infomap); ask before adding it.

### 2.1 New script `scripts/build_clusters.py` (runs after build_connectome.py)
Reads `connections.bin`, `layout_partners.bin`, `groupings.json/bin`; **appends** groupings to groupings.json/bin
with `row: "wiring"` (drop any existing entries with the same ids first, so reruns are idempotent). Add it to
`npm run data` (`--with igraph --with leidenalg --with infomap --with scikit-learn --with scipy`). Fixed seeds.
Start from the code in `scripts/grouping_candidates.py`.

| id | Label | Method | Expected (benchmark) |
|---|---|---|---|
| `leiden_coarse` | Brain modules | Leiden RBConfiguration, undirected summed weights, resolution 1, seed 0 | 12 groups ≥ 50; NMI region 0.60 |
| `leiden_fine` | Brain modules (fine) | same, resolution 5 | 61 groups ≥ 50 |
| `infomap` | Flow modules | Infomap `--directed --two-level --seed 1` | ~150 groups ≥ 50; NMI cell type 0.40 |
| `conn_type` | Connectivity types | HDBSCAN(min_cluster_size 40, min_samples 10) on the Partners UMAP, then assign noise points to the nearest cluster (k-NN k=15, majority vote, in the 64-d partner SVD space) | ~470 groups; NMI cell type 0.69 before noise assignment, expect a little lower after |
| `conn_kmeans` | Partner clusters | MiniBatchKMeans k=1000 on L2-normalised 64-d partner SVD | NMI cell type 0.61 |
| `hub_band` | Hub level | k-core coreness of the undirected graph, bands at the 50/80/95/99th percentiles | 5 bands; orthogonal to labels |

Rules:
- Value 0 ("none"): neurons with no ≥5-synapse partner (~8k) and members of communities with < 20 neurons
  (label "No strong connection or tiny module").
- Name each value by its dominant annotation so the legend reads: modules by dominant **region** + share
  (`"M3 · OL_R 81%"`), connectivity types / partner clusters by dominant **cell type** (`"C117 · T4a 64%"`),
  hub bands by range (`"Core, top 1% (k ≥ 40)"`). Order values by size (hub bands by level).
- Print a sanity table: groups ≥ 50, largest %, NMI vs region / cell type / hemilineage; should match the benchmark.
- Also write a per-neuron float for **coreness** to support 2.3.

### 2.2 Client
- `fillSelect` in `main.ts`: add the `wiring` optgroup ("Wiring").
- Card (selected neuron): add rows Module, Flow module, Connectivity type.
- Focus search already indexes every grouping; check that 1,000 k-means values don't flood results (ranked by count).
- Hub level is ordinal: keep the fixed band order in the legend (like super class / flow in `applyColourBy`).
- README: add a "Wiring" column to the groupings table.

### 2.3 Optional (ask first): continuous colour-by
Hub level (coreness), flow step and synapse count want a sequential single-hue ramp, not categories.
Would need `measures.json/bin` (uint8 quantised), a "Measures" optgroup, a ramp legend, and a validated
sequential ramp (dataviz skill). Listed in BACKLOG "Colour by (continuous)".

## 2b. Connection matrix (approved and built 2026-09-22)

The user picked "Region connection matrix" from §5. Built generically, client-only (no new data file):
- A collapsible **Connections** floating tab, top right (the user's request; was a link in the side panel). The info
  card (bottom right) caps its height with `--matrix-h`, published by a ResizeObserver, so the two never overlap. Rows = the presynaptic group, columns = the postsynaptic group, for the **current colour-by grouping**
 . It updates when colour-by changes.
- Groups: every value when there are ≤ 24, otherwise the 16 largest + "Other". The none value is a last row/column when it has neurons.
- Cell = summed synapses over pairs with 5+ synapses, from `connections.bin` (lazy, 36 MB, same as the card).
- Two scales: **Share of row output** (default, linear 0–100%) and **Synapses** (log). One-hue sequential ramp
  (orange, Tokyo Night hue; blue is reserved for the selection), validated with the dataviz validator `--ordinal --mode dark`.
- It is an HTML `<table>`, so it is its own table view; each cell has a title and aria-label with the numbers.
- Hover a cell: tooltip (A → B, synapses, % of A's output, % of B's input) and the 3D view dims everything but the two groups.
  Click a row or column header: Focus on that group.
- Probe: open the matrix for region, read the diagonal share, screenshot.

## 3. How to verify
- `npx tsc --noEmit && npm run build`.
- `npx vite --port 5199 --strictPort &` then `uv run --with playwright python scripts/ui_probe.py`
  (SwiftShader is slow; the probe waits on `flywire.state.shown === layout && blend >= 1`, never on time).
  Extend it to colour by each new grouping and read the legend rows. Expect no console errors.
- Look at screenshots (`PROBE_OUT=<dir>`), don't just trust state: framing and label clutter only show there.
- Commit and push to `origin main` when a step is verified (end commit messages with the attribution lines).

## 4. Brand rules (from BRAND/brand.css — import values, don't invent)
Tokens `--void #0A0A0A`, `--panel #0D0F12`, `--hairline #1E2227`, `--line-strong #2A2F35`, `--muted #7C848D`,
`--text #C9CDD2`, `--bright #E8EBED`, `--accent #7AA2F7`. Hanken Grotesk body, JetBrains Mono only for short
uppercase labels. Blue marks one thing per view. Sentence case (except the product name, "Fly Brain Explorer", at the user's request), no exclamation marks. Tap targets ≥ 44 px on
mobile. Honour prefers-reduced-motion. `src/brand.css` is a copy; re-copy to update, don't edit.

## 5. Open questions (never asked; defaults in force)
- Region-to-region connection matrix chart? **Answered: yes (§2b).**
- Spectral clustering as a grouping? (Default: no.)
- Continuous colour-by (2.3)? (Default: not yet.)
- Repo visibility: created **private**. Flip with `gh repo edit zack-maz/visualizing-flywire --visibility public
  --accept-visibility-change-consequences`. FlyWire data is CC BY-NC 4.0; keep the attribution in README if it goes public.
