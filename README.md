# Fly brain explorer

A 3D point cloud of all 139,248 neurons in the adult fruit fly brain, from the
[FlyWire](https://flywire.ai) connectome (FAFB release 783). Each neuron is one point.

## Layouts

**Where**
- **Anatomical**: each neuron at its anchor point, inside the neuropil meshes.
- **Soma**: each neuron at its cell body. 15% have no soma in the brain (mostly sensory neurons) and stay at their anchor point.
- **Mirrored**: the right hemisphere folded onto the left, so matching neurons overlap. Colour by hemisphere to compare.

**Maps**
- **Flattened**: a 2D UMAP of each neuron's input and output neuropil profile (Hellinger embedding).
- **Partners**: a 3D UMAP of who each neuron connects to (SVD of inputs and outputs, pairs with 5+ synapses).
- **Sensory → motor**: x = how many steps from the sensory neurons (a neuron joins once 30% of its input
  comes from neurons already reached, after Schlegel et al. 2021); y = dorsal/ventral.

## Groupings

Every grouping can colour the cloud and be searched in Focus.

| Where | What | Lineage | Wiring |
|---|---|---|---|
| Hemisphere, region (Ito et al. 2014 super groups), main neuropil, soma position (40 k-means clusters), input neuropil, output neuropil | Flow, super class, cell class, cell sub class, cell type, hemibrain type | Hemilineage (Ito/Lee), hemilineage (Hartenstein) | Brain modules, brain modules (fine), flow modules, connectivity types, partner clusters, hub level |

The Wiring groupings come from the connections alone (pairs with 5+ synapses), never from the annotations:

- **Brain modules**: Leiden communities of the undirected graph (resolution 1; fine: resolution 5).
- **Flow modules**: Infomap on the directed graph: groups that information tends to circulate within.
- **Connectivity types**: HDBSCAN on the Partners map; the neurons it leaves unassigned join their nearest cluster.
- **Partner clusters**: k-means (k = 1,000) on each neuron's input and output partners.
- **Hub level**: k-core coreness, in bands at the 50th, 80th, 95th and 99th percentiles.

Each value is named after what dominates it: modules by region (`M3 · OL_R 94%`), connectivity types
and partner clusters by cell type (`C5 · T4b 50%`). Neurons with no 5+ synapse partner, and modules
under 20 neurons, have no value.

Colour by shows the ten largest values in colour and the rest in a neutral; hover a legend row to isolate it.

## Connections

Click a neuron to see its strongest inputs and outputs (pairs with 5+ synapses) and highlight them:
inputs as rings, outputs as solid discs. Click a partner in the card to jump to it. The connection
table (~36 MB) loads on the first click.

## Run

```bash
npm install
npm run data   # downloads the raw files, rebuilds public/data/ (two UMAPs: ~10–20 minutes)
npm run dev
```

`npm run data` needs [uv](https://docs.astral.sh/uv/). The files in `public/data/` are enough to run the viewer.
UI check: `npx vite --port 5199 --strictPort` then `uv run --with playwright python scripts/ui_probe.py`.

## Controls

Drag to rotate, right-drag to pan, scroll to zoom toward the cursor. `W A S D` / `Q E` fly
(hold Shift for faster), click selects a neuron, double-click flies to it, `Space` or `1`–`6`
switches layout, `R` resets the view, `Esc` clears selection and focus.

## Data notes

- Positions are each neuron's annotated anchor point (`pos_x/y/z`, 4×4×40 nm voxels), converted to µm.
- Main neuropil = the neuropil with the most synapses (inputs + outputs); input/output neuropil count only one role.
  The synapse table pass (`scripts/neuropil_counts.py`) keeps the role. Neurons with no synapses in a mapped
  neuropil (e.g. photoreceptors in the lamina, which has no mesh) have no main neuropil.
- Sensory → motor: 22 steps; Kenyon cells and antennal lobe projection neurons at step 3 (median), visual
  projection neurons at 7, descending and motor neurons at 4. 4,576 neurons are never reached.
- The 612 neurons annotated `sensory_ascending` are shown as Ascending.
- Left/right: the FAFB images were mirrored during acquisition. FlyWire corrected the `side`
  annotations but not the coordinates, meshes or neuropil names, so the build mirrors x and
  swaps L/R neuropil names to show the fly's real sides ([Codex FAQ](https://codex.flywire.ai/faq)).
- Colours: Tokyo Night hues from the brand palette, ordered so neighbouring legend rows stay distinct
  for colour-blind viewers. Ten hues cannot all be; hover-to-isolate is the backup.
- Style: tokens and fonts from `~/Documents/PROJECTS/BRAND` (copied into `src/brand.css` and `public/fonts/`).

Data licence: FlyWire data is CC BY-NC 4.0; the processed files in `public/data/` inherit it.

Sources: Dorkenwald et al. 2024 and Schlegel et al. 2024 (*Nature*); annotations from
[flywire_annotations](https://github.com/flyconnectome/flywire_annotations); meshes, synapses and the
connection table from the Lee lab's public FAFB 783 bucket.
