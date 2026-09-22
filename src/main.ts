import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { loadAll, loadConnections, loadFlowRank, loadLayout, type Connections, type Grouping } from './data';
import { DIM, HIDDEN, INPUT, OUTPUT, SHOWN, createNeuronPoints, createShellMaterial, meshGeometry } from './scene';
import { NEUTRAL, PALETTE, SUPER_CLASS_STYLE, shortLabel, valueLabel } from './labels';
import { createMatrix } from './matrix';

// The custom shaders write colours straight to the screen, so keep hex values as-is (no linear conversion).
THREE.ColorManagement.enabled = false;

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const loadingEl = $('#loading');
const loadingText = $('#loading-text');

const { meta, anatomical, groupings, meshes } = await loadAll((mb) => {
  loadingText.textContent = `Loading connectome… ${mb.toFixed(1)} MB`;
}).catch((err) => {
  loadingText.textContent = `Could not load data: ${err.message}. Run "npm run data" first.`;
  throw err;
});

const N = meta.count;
const byId = new Map(groupings.map((g) => [g.id, g]));
const G = (id: string) => byId.get(id)!;
const superClass = G('super_class'), cellType = G('cell_type'), side = G('side');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const V3 = (a: ArrayLike<number>, i = 0) => new THREE.Vector3(a[i], a[i + 1], a[i + 2]);

/** "MB_CA_L · Neuropil" style name of a value, as the focus search and labels show it. */
const nameOf = (g: Grouping, code: number) => (code === 0 ? g.none : valueLabel(g.id, g.values[code]));

// ------------------------------------------------------------------ renderer, camera, controls
const canvas = $<HTMLCanvasElement>('#viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor('#0A0A0A');
const labelRenderer = new CSS2DRenderer({ element: $('#labels') });

const scene = new THREE.Scene();
scene.fog = new THREE.Fog('#0A0A0A', 2500, 6000);
const camera = new THREE.PerspectiveCamera(45, 1, 1, 20000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.zoomToCursor = true;
controls.screenSpacePanning = true;
controls.minDistance = 5;
controls.maxDistance = 8000;

// ------------------------------------------------------------------ scene contents
const current = anatomical.slice();     // what is on screen; the only position attribute
const points = createNeuronPoints(current);
scene.add(points.group);
const stateArr = points.state.array as Float32Array;
const colorArr = points.color.array as Float32Array;

const brainShell = new THREE.Mesh(meshGeometry(meshes.find((m) => m.name === 'brain')!), createShellMaterial('#7C848D', 0.25));
brainShell.renderOrder = 2;
scene.add(brainShell);

const neuropilShells = new Map<string, THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>>();
const meshCentre = new Map<string, THREE.Vector3>();
const shellGroup = new THREE.Group();
for (const m of meshes) {
  if (m.name === 'brain') continue;
  const shell = new THREE.Mesh(meshGeometry(m), createShellMaterial('#C9CDD2', 0.12));
  shell.renderOrder = 2;
  neuropilShells.set(m.name, shell);
  shellGroup.add(shell);
  const c = new THREE.Vector3();
  for (let k = 0; k < m.positions.length; k += 3) c.add(V3(m.positions, k));
  meshCentre.set(m.name, c.divideScalar(m.positions.length / 3));
}
scene.add(shellGroup);

// Ring that marks the selected neuron: the one accent-blue thing in the view.
const markerGeom = new THREE.BufferGeometry();
markerGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
const marker = new THREE.Points(markerGeom, new THREE.ShaderMaterial({
  transparent: true, depthTest: false,
  vertexShader: 'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); gl_PointSize = 26.0; }',
  fragmentShader: 'void main(){ float r = length(gl_PointCoord - 0.5); if (r > 0.5 || r < 0.38) discard; gl_FragColor = vec4(0.478, 0.635, 0.969, 1.0); }',
}));
marker.visible = false;
marker.renderOrder = 3;
scene.add(marker);

// ------------------------------------------------------------------ labels
interface LabelSpec {
  text: string;
  title: string;
  pos: THREE.Vector3;
  priority: number;                        // bigger wins when labels overlap
  focus?: { grouping: string; code: number };
  major?: boolean;
}
let labels: { spec: LabelSpec; obj: CSS2DObject }[] = [];
function setLabels(specs: LabelSpec[]) {
  for (const l of labels) scene.remove(l.obj);
  labels = specs.sort((a, b) => b.priority - a.priority).map((spec) => {
    const el = document.createElement('button');
    el.className = spec.major ? 'np-label major' : 'np-label';
    el.textContent = spec.text;
    el.title = spec.title;
    if (spec.focus) el.addEventListener('click', () => setFocus(spec.focus!.grouping, spec.focus!.code));
    const obj = new CSS2DObject(el);
    obj.position.copy(spec.pos);
    obj.visible = false;
    scene.add(obj);
    return { spec, obj };
  });
  markActiveLabels();
}
function markActiveLabels() {
  for (const { spec, obj } of labels) {
    const on = !!focus && spec.focus?.grouping === focus.grouping && spec.focus.code === focus.code;
    (obj.element as HTMLElement).classList.toggle('active', on);
  }
}

/** Labels for the largest values of a grouping, placed at their members' median position in `pos`. */
function valueLabels(g: Grouping, pos: Float32Array, limit: number, lift = 0): LabelSpec[] {
  const top = g.counts.map((c, code) => ({ c, code })).filter((v) => v.code > 0 && v.c > 0)
    .sort((a, b) => b.c - a.c).slice(0, limit);
  const want = new Map(top.map((v, k) => [v.code, k]));
  const xs = top.map(() => [[], [], []] as number[][]);
  for (let i = 0; i < N; i++) {
    const k = want.get(g.codes[i]);
    if (k === undefined || (i % Math.ceil(top[k].c / 400))) continue;    // a sample is plenty for a median
    for (let a = 0; a < 3; a++) xs[k][a].push(pos[i * 3 + a]);
  }
  const median = (v: number[]) => v.sort((a, b) => a - b)[v.length >> 1] ?? 0;
  return top.map((v, k) => ({
    text: shortLabel(g.id, g.values[v.code]),
    title: `${nameOf(g, v.code)} · ${v.c.toLocaleString()} neurons`,
    pos: new THREE.Vector3(median(xs[k][0]), median(xs[k][1]) + lift, median(xs[k][2])),
    priority: v.c,
    focus: { grouping: g.id, code: v.code },
  }));
}

// ------------------------------------------------------------------ layouts
// A layout produces a target: where every neuron goes.
interface Target {
  whole: Float32Array;
  labels: () => LabelSpec[];
  shells: number;                 // opacity factor for the anatomy meshes (0 = hidden)
}
interface LayoutDef { id: string; label: string; row: 'Where' | 'Maps'; hint: string; build: () => Promise<Target> }

const cache = new Map<string, Promise<Target>>();
const once = (key: string, make: () => Promise<Target>) => {
  if (!cache.has(key)) cache.set(key, make().catch((e) => { cache.delete(key); throw e; }));
  return cache.get(key)!;
};

const anatomyLabels = (keep: (name: string) => boolean = () => true) => () => {
  const np = G('neuropil');
  return np.values.flatMap((v, code) => {
    const c = meshCentre.get(v);
    if (!code || !c || !keep(v)) return [];
    return [{ text: v.replace(/_/g, ' '), title: `${nameOf(np, code)} · ${np.counts[code].toLocaleString()} neurons`, pos: c, priority: np.counts[code], focus: { grouping: 'neuropil', code } }];
  });
};

let flowRank: Uint8Array | null = null;
const flowRankReady = loadFlowRank().then((r) => (flowRank = r));

const LAYOUTS: LayoutDef[] = [
  {
    id: 'anatomical', label: 'Anatomical', row: 'Where',
    hint: 'Each neuron at its anchor point in the brain, the spot FlyWire uses to identify it.',
    build: () => once('anatomical', async () => ({ whole: anatomical, labels: anatomyLabels(), shells: 1 })),
  },
  {
    id: 'soma', label: 'Soma', row: 'Where',
    hint: 'Each neuron at its cell body. Cell bodies sit in a rind around the neuropils; 15% (mostly sensory neurons) have theirs outside the brain and stay at their anchor point.',
    build: () => once('soma', async () => {
      const whole = await loadLayout('soma');
      return { whole, labels: () => valueLabels(G(colourBy), whole, 14), shells: 1 };
    }),
  },
  {
    id: 'mirrored', label: 'Mirrored', row: 'Where',
    hint: 'The right hemisphere folded onto the left, so matching neurons of both sides overlap. Colour by hemisphere to compare them.',
    build: () => once('mirrored', async () => {
      const whole = anatomical.slice();
      const right = side.values.indexOf('right');
      // +X is the fly's left, so folding the right hemisphere over means x -> -x.
      for (let i = 0; i < N; i++) if (side.codes[i] === right) whole[i * 3] = -whole[i * 3];
      return { whole, labels: anatomyLabels((v) => !v.endsWith('_R')), shells: 1 };
    }),
  },
  {
    id: 'flat', label: 'Flattened', row: 'Maps',
    hint: 'A 2D map where neurons that receive and send in the same neuropils sit together (UMAP of input and output neuropil profiles). Neurons with no synapses in a mapped neuropil sit in the strip below.',
    build: () => once('flat', async () => {
      const whole = await loadLayout('flat');
      return { whole, labels: () => valueLabels(G(colourBy), whole, 16), shells: 0 };
    }),
  },
  {
    id: 'partners', label: 'Partners', row: 'Maps',
    hint: 'Neurons with similar connection partners sit together (SVD + UMAP of who each neuron connects to, pairs with 5+ synapses). Neurons without a strong connection sit in the disc below.',
    build: () => once('partners', async () => {
      const whole = await loadLayout('partners');
      return { whole, labels: () => valueLabels(G(colourBy), whole, 16), shells: 0 };
    }),
  },
  {
    id: 'flow', label: 'Sensory → motor', row: 'Maps',
    hint: 'Left to right: how many steps from the sensory neurons each neuron sits. A neuron joins a step once 30% of its inputs come from neurons already reached (after Schlegel et al. 2021). Vertical = dorsal/ventral.',
    build: () => once('flow', async () => {
      const [whole] = await Promise.all([loadLayout('flow'), flowRankReady]);
      return { whole, labels: () => flowLabels(whole), shells: 0 };
    }),
  },
];

function flowLabels(pos: Float32Array): LabelSpec[] {
  if (!flowRank) return [];
  const steps = new Map<number, { x: number; n: number }>();
  let minY = Infinity;
  for (let i = 0; i < N; i++) {
    const s = steps.get(flowRank[i]) ?? { x: 0, n: 0 };
    s.x += pos[i * 3]; s.n++;
    steps.set(flowRank[i], s);
    minY = Math.min(minY, pos[i * 3 + 1]);
  }
  return [...steps].filter(([, s]) => s.n > 50).map(([step, s]) => ({
    text: step === 255 ? 'Not reached' : step === 1 ? 'Step 1 · sensory' : `${step}`,
    title: `${s.n.toLocaleString()} neurons ${step === 255 ? 'never reached from the sensory neurons' : `at step ${step}`}`,
    pos: new THREE.Vector3(s.x / s.n, minY - 25, 0),
    priority: s.n,
    major: step === 1 || step === 255,
  }));
}

// ------------------------------------------------------------------ view state
let layoutId = 'anatomical';
let shownId = 'anatomical';      // the layout whose target is loaded (layoutId can still be loading)
let target: Target = { whole: anatomical, labels: anatomyLabels(), shells: 1 };
let targetPos: Float32Array = anatomical;    // where neurons are heading
const from = anatomical.slice();             // where the current transition started
let blendT = 1;                              // 0 → 1 across a layout transition
let shellFade = 1;
let focus: { grouping: string; code: number } | null = null;
let selected = -1;
let colourBy = 'super_class';
let layoutRequest = 0;

const TRANSITION = reducedMotion ? 0.001 : 1.4;
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Where neuron i will be once everything settles. */
function goalPosition(i: number, out = new THREE.Vector3()) {
  return out.set(targetPos[i * 3], targetPos[i * 3 + 1], targetPos[i * 3 + 2]);
}

const statusEl = $('#layout-status');
async function setLayout(id: string, opts: { keepCamera?: boolean } = {}) {
  const def = LAYOUTS.find((l) => l.id === id)!;
  layoutId = id;
  const req = ++layoutRequest;
  document.querySelectorAll<HTMLButtonElement>('[data-layout]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.layout === id)));
  $('#layout-hint').textContent = def.hint;
  const slow = setTimeout(() => { if (req === layoutRequest) statusEl.textContent = 'Loading layout…'; }, 150);
  let next: Target;
  try {
    next = await def.build();
  } catch (err) {
    clearTimeout(slow);
    statusEl.textContent = `Could not load this layout: ${(err as Error).message}`;
    return;
  }
  clearTimeout(slow);
  if (req !== layoutRequest) return;       // a newer request won
  statusEl.textContent = '';
  from.set(current);
  target = next;
  shownId = id;
  targetPos = target.whole;
  blendT = 0;
  setLabels(target.labels());
  if (!opts.keepCamera) { if (focus) frameFocus(); else frameAll(); }
}

// ------------------------------------------------------------------ camera
const tween = { active: false, t: 0, fromPos: new THREE.Vector3(), fromTarget: new THREE.Vector3(), toPos: new THREE.Vector3(), toTarget: new THREE.Vector3() };
function flyTo(to: THREE.Vector3, distance: number) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  tween.fromPos.copy(camera.position);
  tween.fromTarget.copy(controls.target);
  tween.toTarget.copy(to);
  tween.toPos.copy(to).addScaledVector(dir, distance);
  tween.t = reducedMotion ? 1 : 0;
  tween.active = true;
}
/** Distance at which a sphere of this radius fills the narrower field of view. */
function fitDistance(radius: number) {
  const halfV = (camera.fov * Math.PI) / 360;
  // Only the part of the canvas right of the panel counts as width.
  const w = canvas.clientWidth, visible = w > 0 ? (w - panelPx) / w : 1;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect * visible);
  return radius / Math.tan(Math.min(halfV, halfH));
}
/** Frame a set of neurons as they will sit once the transition ends (90th-percentile spread). */
function frameNeurons(members: number[] | null, pad: number) {
  const pts: THREE.Vector3[] = [];
  const step = Math.max(1, Math.floor((members?.length ?? N) / 20000));
  if (members) for (let k = 0; k < members.length; k += step) pts.push(goalPosition(members[k]));
  else for (let i = 0; i < N; i += step) pts.push(goalPosition(i));
  if (!pts.length) return;
  const centre = pts.reduce((acc, p) => acc.add(p), new THREE.Vector3()).divideScalar(pts.length);
  const d = pts.map((p) => p.distanceTo(centre)).sort((x, y) => x - y);
  const radius = Math.max(25, d[Math.floor(d.length * (members ? 0.9 : 0.98))]);
  flyTo(centre, fitDistance(radius) * pad);
}
const frameAll = () => frameNeurons(null, 1.05);
function frameFocus() {
  if (!focus) return;
  const g = G(focus.grouping), members: number[] = [];
  for (let i = 0; i < N; i++) if (g.codes[i] === focus.code) members.push(i);
  frameNeurons(members, 2);
}

// ------------------------------------------------------------------ colour by + legend
const LEGEND_TOP = 10, OTHER = LEGEND_TOP, NONE = LEGEND_TOP + 1;
const slotOf = new Uint8Array(N);            // legend row per neuron
let slotCodes: number[] = [];                // value code per coloured row
const slotHidden = new Array(NONE + 1).fill(false);
let previewSlot = -1, soloSlot = -1;

function applyColourBy() {
  const g = G(colourBy);
  // Top values by count get the palette (super class keeps its canonical order); the rest share the neutral.
  const ranked = g.counts.map((c, code) => ({ c, code })).filter((v) => v.code > 0 && v.c > 0);
  if (!['super_class', 'flow', 'side', 'hub_band'].includes(g.id)) ranked.sort((a, b) => b.c - a.c);
  slotCodes = ranked.slice(0, LEGEND_TOP).map((v) => v.code);
  const slotByCode = new Map(slotCodes.map((c, s) => [c, s]));
  const colours = [...PALETTE.slice(0, LEGEND_TOP), NEUTRAL, NEUTRAL].map((h) => new THREE.Color(h));
  for (let i = 0; i < N; i++) {
    const code = g.codes[i];
    const s = code === 0 ? NONE : slotByCode.get(code) ?? OTHER;
    slotOf[i] = s;
    const c = colours[s];
    colorArr[i * 3] = c.r; colorArr[i * 3 + 1] = c.g; colorArr[i * 3 + 2] = c.b;
  }
  points.color.needsUpdate = true;
  slotHidden.fill(false);
  previewSlot = soloSlot = -1;

  const legend = $('#legend');
  legend.replaceChildren();
  const others = ranked.slice(LEGEND_TOP);
  const rows: { slot: number; name: string; count: number; colour: string; title: string }[] = slotCodes.map((code, s) => ({
    slot: s, name: nameOf(g, code), count: g.counts[code], colour: PALETTE[s],
    title: g.id === 'super_class' ? SUPER_CLASS_STYLE[g.values[code]]?.blurb ?? '' : `${g.values[code]}`,
  }));
  if (others.length) rows.push({ slot: OTHER, name: `${others.length.toLocaleString()} other values`, count: others.reduce((a, v) => a + v.c, 0), colour: NEUTRAL, title: 'Every value outside the largest ten' });
  if (g.counts[0]) rows.push({ slot: NONE, name: g.none, count: g.counts[0], colour: NEUTRAL, title: 'No value in this grouping' });
  for (const r of rows) {
    const row = document.createElement('button');
    row.className = 'legend-row';
    row.dataset.slot = String(r.slot);
    row.title = r.title;
    row.innerHTML = `<span class="swatch" style="background:${r.colour}${r.slot === NONE ? ';opacity:.5' : ''}"></span><span class="legend-name"></span><span class="legend-count">${r.count.toLocaleString()}</span>`;
    row.querySelector('.legend-name')!.textContent = r.name;
    row.addEventListener('mouseenter', () => { previewSlot = r.slot; updateState(); });
    row.addEventListener('mouseleave', () => { previewSlot = -1; updateState(); });
    row.addEventListener('click', (e) => {
      if (e.detail === 2) return;
      slotHidden[r.slot] = !slotHidden[r.slot];
      updateState();
    });
    row.addEventListener('dblclick', () => {
      // Double-click shows only this row; double-clicking it again shows everything.
      if (soloSlot === r.slot) { slotHidden.fill(false); soloSlot = -1; }
      else { slotHidden.fill(true); slotHidden[r.slot] = false; soloSlot = r.slot; }
      updateState();
    });
    legend.appendChild(row);
  }
  updateState();
  if (labels.length && ['soma', 'flat', 'partners'].includes(layoutId)) setLabels(target.labels());
}

// ------------------------------------------------------------------ per-neuron state
let conn: Connections | null = null;
let showPartners = true;
const partnerState = new Map<number, number>();    // neuron -> INPUT | OUTPUT while a neuron is selected
let matrixTest: ((i: number) => boolean) | null = null;   // neurons of the hovered matrix cell or header

function updateState() {
  const fg = focus ? G(focus.grouping) : null;
  const partnersOn = selected >= 0 && showPartners && partnerState.size > 0;
  for (let i = 0; i < N; i++) {
    const s = slotOf[i];
    let st = SHOWN;
    if (slotHidden[s]) st = HIDDEN;
    else if (partnersOn) st = i === selected ? SHOWN : partnerState.get(i) ?? DIM;
    else if (matrixTest) st = matrixTest(i) ? SHOWN : DIM;
    else if ((previewSlot >= 0 && s !== previewSlot) || (fg && fg.codes[i] !== focus!.code)) st = DIM;
    stateArr[i] = st;
  }
  points.state.needsUpdate = true;
  document.querySelectorAll<HTMLElement>('.legend-row').forEach((row) => {
    const off = slotHidden[Number(row.dataset.slot)];
    row.classList.toggle('off', off);
    row.setAttribute('aria-pressed', String(!off));
  });
}

// ------------------------------------------------------------------ focus search
interface Hit { grouping: Grouping; code: number; name: string; hay: string }
const index: Hit[] = groupings.flatMap((g) => g.values.map((v, code) => ({
  grouping: g, code, name: code ? v : g.none,
  hay: `${v} ${code ? valueLabel(g.id, v) : g.none} ${g.label}`.toLowerCase(),
})).filter((h) => h.code > 0 && h.grouping.counts[h.code] > 0));

const searchInput = $<HTMLInputElement>('#focus-search');
const results = $<HTMLUListElement>('#focus-results');
const searchBox = searchInput.parentElement!;
let hits: Hit[] = [], active = -1;

function search(q: string): Hit[] {
  const t = q.trim().toLowerCase();
  if (!t) return [];
  const words = t.split(/\s+/);
  const scored: { h: Hit; score: number }[] = [];
  for (const h of index) {
    if (!words.every((w) => h.hay.includes(w))) continue;
    const exact = h.name.toLowerCase() === t ? 2 : h.name.toLowerCase().startsWith(t) ? 1 : 0;
    scored.push({ h, score: exact * 1e7 + h.grouping.counts[h.code] });
  }
  return scored.sort((a, b) => b.score - a.score).slice(0, 40).map((s) => s.h);
}
function renderResults() {
  results.replaceChildren(...hits.map((h, k) => {
    const li = document.createElement('li');
    li.role = 'option';
    li.id = `hit-${k}`;
    li.setAttribute('aria-selected', String(k === active));
    const full = valueLabel(h.grouping.id, h.name);
    li.innerHTML = '<span class="r-name"></span><span class="r-kind"></span><span class="r-count"></span>';
    li.querySelector('.r-name')!.textContent = h.name;
    li.querySelector('.r-kind')!.textContent = full !== h.name ? `${h.grouping.label} · ${full}` : h.grouping.label;
    li.querySelector('.r-count')!.textContent = h.grouping.counts[h.code].toLocaleString();
    li.addEventListener('mousedown', (e) => { e.preventDefault(); choose(k); });
    return li;
  }));
  results.hidden = !hits.length;
  searchBox.setAttribute('aria-expanded', String(!results.hidden));
  if (active >= 0) { searchInput.setAttribute('aria-activedescendant', `hit-${active}`); results.children[active]?.scrollIntoView({ block: 'nearest' }); }
  else searchInput.removeAttribute('aria-activedescendant');
}
function choose(k: number) {
  const h = hits[k];
  if (!h) return;
  hits = []; active = -1; renderResults();
  setFocus(h.grouping.id, h.code, true);
  searchInput.blur();
}
searchInput.addEventListener('input', () => { hits = search(searchInput.value); active = hits.length ? 0 : -1; renderResults(); });
searchInput.addEventListener('focus', () => { if (focus) searchInput.select(); });
searchInput.addEventListener('blur', () => { hits = []; renderResults(); showFocusText(); });
searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') { active = Math.min(hits.length - 1, active + 1); renderResults(); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { active = Math.max(0, active - 1); renderResults(); e.preventDefault(); }
  else if (e.key === 'Enter') choose(active);
  else if (e.key === 'Escape') { searchInput.blur(); }
});
$('#focus-clear').addEventListener('click', () => setFocus(null));

function showFocusText() {
  if (focus) {
    const g = G(focus.grouping);
    searchInput.value = `${focus.code ? g.values[focus.code] : g.none} · ${g.label}`;
  } else searchInput.value = '';
  searchInput.classList.toggle('has-focus', !!focus);
  $('#focus-clear').hidden = !focus;
}

function setFocus(grouping: string | null, code = 0, force = false) {
  const same = !force && grouping && focus?.grouping === grouping && focus.code === code;
  focus = grouping && !same ? { grouping, code } : null;
  showFocusText();
  markActiveLabels();
  updateState();
  if (focus) frameFocus();
}

// ------------------------------------------------------------------ picking
// CPU screen-space nearest point over what is on screen.
const proj = new THREE.Matrix4();
function pick(clientX: number, clientY: number): number {
  const rect = canvas.getBoundingClientRect();
  const mx = ((clientX - rect.left) / rect.width) * 2 - 1;
  const my = -((clientY - rect.top) / rect.height) * 2 + 1;
  proj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = proj.elements;
  const tolX = (8 / rect.width) * 2, tolY = (8 / rect.height) * 2;
  let best = -1, bestZ = Infinity;
  for (let i = 0; i < N; i++) {
    const st = stateArr[i];
    if (st === HIDDEN || st === DIM) continue;
    const k = i * 3, x = current[k], y = current[k + 1], z = current[k + 2];
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (w <= 0) continue;
    const sx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
    if (Math.abs(sx - mx) > tolX) continue;
    const sy = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
    if (Math.abs(sy - my) > tolY) continue;
    if (w < bestZ) { bestZ = w; best = i; }
  }
  return best;
}

// ------------------------------------------------------------------ info card
const card = $('#info');
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!));
const typeName = (i: number) => cellType.values[cellType.codes[i]] || 'Untyped';
const val = (id: string, i: number) => { const g = G(id); const c = g.codes[i]; return c ? valueLabel(id, g.values[c]) : ''; };

function partnerList(i: number, dir: 'in' | 'out') {
  if (!conn) return [];
  const off = dir === 'in' ? conn.inOffsets : conn.outOffsets;
  const partner = dir === 'in' ? conn.inPartner : conn.outPartner;
  const weight = dir === 'in' ? conn.inWeight : conn.outWeight;
  const out: { j: number; w: number }[] = [];
  for (let k = off[i]; k < off[i + 1]; k++) out.push({ j: partner[k], w: weight[k] });
  return out;
}

function describe(i: number) {
  const isSel = i === selected;
  const sc = superClass.values[superClass.codes[i]];
  const swatch = slotOf[i] < LEGEND_TOP ? PALETTE[slotOf[i]] : NEUTRAL;
  const cb = G(colourBy);
  const rows: [string, string][] = [
    ...(colourBy === 'super_class' ? [] : [['Super class', SUPER_CLASS_STYLE[sc]?.label ?? sc] as [string, string]]),
    ['Class', val('cell_class', i) || 'No cell class'],
    ['Sub class', val('cell_sub_class', i)],
    ['Hemibrain', val('hemibrain_type', i)],
    ['Side', val('side', i)],
    ['Neuropil', G('neuropil').codes[i] ? `${G('neuropil').values[G('neuropil').codes[i]]} · ${val('neuropil', i)}` : 'No main neuropil'],
  ];
  if (isSel) {
    const inNp = G('in_neuropil').values[G('in_neuropil').codes[i]], outNp = G('out_neuropil').values[G('out_neuropil').codes[i]];
    rows.push(['In → out', `${inNp || '–'} → ${outNp || '–'}`], ['Region', val('region', i)], ['Soma', val('soma_cluster', i) || 'Outside the brain'],
      ['Lineage', [val('ito_lee_hemilineage', i), val('hartenstein_hemilineage', i)].filter(Boolean).join(' · ')],
      ['Flow', val('flow', i)],
      ['Module', val('leiden_coarse', i)], ['Flow module', val('infomap', i)], ['Connectivity type', val('conn_type', i)],
      ['Hub level', val('hub_band', i)]);
    if (flowRank) rows.push(['Step', flowRank[i] === 255 ? 'Not reached from sensory' : `${flowRank[i]} from sensory`]);
  }
  const id = meta.rootIds[i];
  let partners = '';
  if (isSel) {
    if (!conn) partners = '<div class="partners"><span class="label">Loading partners…</span></div>';
    else {
      const ins = partnerList(i, 'in'), outs = partnerList(i, 'out');
      const col = (list: typeof ins) => list.slice(0, 8).map(({ j, w }) =>
        `<button class="partner" data-j="${j}" title="${esc(`${typeName(j)} · ${val('cell_class', j) || superClass.values[superClass.codes[j]]}`)}"><span>${esc(typeName(j))}</span><span>${w}</span></button>`).join('') || '<span class="info-hint">None with 5+ synapses</span>';
      partners = `<div class="partners">
        <div class="partners-head"><span class="label">Partners · synapses</span>
          <label class="check"><input type="checkbox" id="opt-partners" ${showPartners ? 'checked' : ''}/> Highlight</label></div>
        <div class="partners-cols">
          <div><h3><span class="mark"></span>Inputs ${ins.length}</h3>${col(ins)}</div>
          <div><h3><span class="mark solid"></span>Outputs ${outs.length}</h3>${col(outs)}</div>
        </div></div>`;
    }
  }
  card.classList.toggle('selected', isSel);
  card.innerHTML = `
    <div class="info-type">${esc(typeName(i))}</div>
    <div class="info-sub" title="${esc(cb.label)}"><span class="swatch" style="background:${swatch}"></span>${esc(nameOf(cb, cb.codes[i]))}</div>
    <dl class="info-grid">${rows.filter(([, v]) => v).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('')}</dl>
    ${partners}
    <div class="info-id">${id}</div>
    ${isSel ? `<a class="info-link" href="https://codex.flywire.ai/app/cell_details?root_id=${id}&data_version=${meta.release}" target="_blank" rel="noopener">Open in Codex ↗</a>` : '<div class="info-hint">Click to select and show partners</div>'}
  `;
  card.hidden = false;
  card.querySelectorAll<HTMLButtonElement>('.partner').forEach((b) => b.addEventListener('click', () => select(Number(b.dataset.j), true)));
  card.querySelector<HTMLInputElement>('#opt-partners')?.addEventListener('change', (e) => {
    showPartners = (e.target as HTMLInputElement).checked;
    updateState();
  });
}
function refreshCard(hover: number) {
  if (hover >= 0) describe(hover);
  else if (selected >= 0) describe(selected);
  else card.hidden = true;
}

function select(i: number, fly = false) {
  selected = i;
  marker.visible = i >= 0;
  partnerState.clear();
  if (i >= 0) {
    const fill = () => {
      if (selected !== i) return;
      partnerState.clear();
      // Inputs as rings, outputs as discs; a neuron that is both shows its stronger direction.
      const ins = new Map(partnerList(i, 'in').map((p) => [p.j, p.w]));
      for (const j of ins.keys()) partnerState.set(j, INPUT);
      for (const p of partnerList(i, 'out')) if (p.w >= (ins.get(p.j) ?? 0)) partnerState.set(p.j, OUTPUT);
      updateState();
      refreshCard(-1);
    };
    if (conn) fill();
    else loadConnections(N).then((c) => { conn = c; fill(); });
    if (fly) flyTo(goalPosition(i), Math.max(80, camera.position.distanceTo(controls.target) * 0.5));
  }
  updateState();
  refreshCard(-1);
}

let hovered = -1;
let pointer: { x: number; y: number } | null = null;
let downAt: { x: number; y: number } | null = null;
canvas.addEventListener('pointermove', (e) => { pointer = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('pointerleave', () => { pointer = null; hovered = -1; refreshCard(-1); canvas.style.cursor = ''; });
canvas.addEventListener('pointerdown', (e) => { downAt = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return;
  select(pick(e.clientX, e.clientY));
});
canvas.addEventListener('dblclick', (e) => {
  const i = pick(e.clientX, e.clientY);
  if (i >= 0) flyTo(V3(current, i * 3), Math.max(60, camera.position.distanceTo(controls.target) * 0.35));
});

// ------------------------------------------------------------------ keyboard flight
const keys = new Set<string>();
window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === 'INPUT' || tag === 'SELECT') return;
  keys.add(e.key.toLowerCase());
  if (e.key === 'r' || e.key === 'R') frameAll();
  if ((e.key === 'f' || e.key === 'F') && selected >= 0) flyTo(goalPosition(selected), 80);
  if (e.key === ' ') {
    e.preventDefault();
    const k = LAYOUTS.findIndex((l) => l.id === layoutId);
    setLayout(LAYOUTS[(k + 1) % LAYOUTS.length].id);
  }
  const n = Number(e.key);
  if (n >= 1 && n <= LAYOUTS.length) setLayout(LAYOUTS[n - 1].id);
  if (e.key === 'Escape') { select(-1); setFocus(null); }
});
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener('blur', () => keys.clear());

function fly(dt: number) {
  if (!keys.size) return;
  const speed = camera.position.distanceTo(controls.target) * (keys.has('shift') ? 2.5 : 0.9) * dt;
  const fwd = new THREE.Vector3();
  camera.getWorldDirection(fwd);
  const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
  const move = new THREE.Vector3();
  if (keys.has('w') || keys.has('arrowup')) move.add(fwd);
  if (keys.has('s') || keys.has('arrowdown')) move.sub(fwd);
  if (keys.has('d') || keys.has('arrowright')) move.add(right);
  if (keys.has('a') || keys.has('arrowleft')) move.sub(right);
  if (keys.has('e')) move.add(camera.up);
  if (keys.has('q')) move.sub(camera.up);
  if (!move.lengthSq()) return;
  move.normalize().multiplyScalar(speed);
  tween.active = false;
  camera.position.add(move);
  controls.target.add(move);
}

// ------------------------------------------------------------------ panel UI
$('#stat-neurons').textContent = N.toLocaleString();
$('#stat-neuropils').textContent = String(G('neuropil').values.length - 1);
$('#stat-types').textContent = (cellType.values.length - 1).toLocaleString();

// Layout buttons, one row per kind, numbered in panel order.
const layoutRows = $('#layouts');
for (const row of ['Where', 'Maps'] as const) {
  const lab = document.createElement('span');
  lab.className = 'label';
  lab.textContent = row;
  const seg = document.createElement('div');
  seg.className = 'segmented';
  seg.role = 'group';
  seg.ariaLabel = `${row} layouts`;
  for (const l of LAYOUTS.filter((d) => d.row === row)) {
    const b = document.createElement('button');
    b.dataset.layout = l.id;
    b.textContent = l.label;
    b.title = `${l.label} (${LAYOUTS.indexOf(l) + 1})`;
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => setLayout(l.id));
    seg.appendChild(b);
  }
  layoutRows.append(lab, seg);
}

// Grouping pickers: every grouping under Where / What / Lineage / Wiring headings.
function fillSelect(sel: HTMLSelectElement, value: string) {
  sel.replaceChildren();
  for (const [row, title] of [['where', 'Where'], ['what', 'What'], ['lineage', 'Lineage'], ['wiring', 'Wiring']] as const) {
    const og = document.createElement('optgroup');
    og.label = title;
    for (const g of groupings.filter((x) => x.row === row)) og.append(new Option(`${g.label} (${(g.values.length - 1).toLocaleString()})`, g.id));
    sel.append(og);
  }
  sel.value = value;
}
const colourSel = $<HTMLSelectElement>('#colour-by');
fillSelect(colourSel, colourBy);
colourSel.addEventListener('change', () => { colourBy = colourSel.value; applyColourBy(); renderMatrix(); });

// ------------------------------------------------------------------ connection matrix
const matrixEl = $('#matrix'), matrixToggle = $('#matrix-toggle'), matrixContent = $('#matrix-content');
let matrixOpen = false;
const matrix = createMatrix(matrixEl, {
  onHover: (test) => { matrixTest = test; updateState(); },
  onPick: (grouping, code) => setFocus(grouping, code),
});
function renderMatrix() {
  $('#matrix-title').textContent = `Connections · ${G(colourBy).label}`;
  if (!matrixOpen) return;
  if (conn) { matrix.render(G(colourBy), conn); return; }
  matrixEl.querySelector('.matrix-readout')!.textContent = 'Loading connections…';
  loadConnections(N).then((c) => { conn = c; renderMatrix(); });
}
function showMatrix(on: boolean) {
  matrixOpen = on;
  matrixContent.hidden = !on;
  matrixEl.classList.toggle('open', on);
  matrixToggle.setAttribute('aria-expanded', String(on));
  if (!on) { matrixTest = null; updateState(); }
  renderMatrix();
}
matrixToggle.addEventListener('click', () => showMatrix(!matrixOpen));
renderMatrix();   // sets the tab title
// The info card (bottom right) gives way to the tab (top right): publish the tab's height.
new ResizeObserver(() => document.body.style.setProperty('--matrix-h', `${matrixEl.offsetHeight}px`)).observe(matrixEl);
$('#legend-all').addEventListener('click', () => { slotHidden.fill(false); soloSlot = -1; updateState(); });

const toggle = (id: string, fn: (on: boolean) => void) => {
  const el = $<HTMLInputElement>(id);
  el.addEventListener('change', () => fn(el.checked));
  fn(el.checked);
};
let showShells = true, showBrain = true;
toggle('#opt-shells', (on) => { showShells = on; });
toggle('#opt-brain', (on) => { showBrain = on; });
toggle('#opt-labels', (on) => { $('#labels').hidden = !on; });
const sizeInput = $<HTMLInputElement>('#opt-size');
sizeInput.addEventListener('input', () => { points.uniforms.uSize.value = Number(sizeInput.value); });
points.uniforms.uSize.value = Number(sizeInput.value);

$('#help-toggle').addEventListener('click', () => {
  const help = $('#help');
  help.toggleAttribute('hidden');
  $('#help-toggle').setAttribute('aria-expanded', String(!help.hidden));
});
$('#panel-toggle').addEventListener('click', () => { document.body.classList.toggle('panel-collapsed'); resize(); });

// ------------------------------------------------------------------ labels per frame
function updateLabels() {
  const settled = blendT >= 0.9;
  for (const { obj } of labels) obj.visible = settled;
}
const labelPos = new THREE.Vector3();
function declutterLabels() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const placed: [number, number, number, number][] = [];
  const isFocused = (s: LabelSpec) => !!focus && s.focus?.grouping === focus.grouping && s.focus.code === focus.code;
  const order = [...labels.filter((l) => isFocused(l.spec)), ...labels.filter((l) => !isFocused(l.spec))];
  for (const { obj, spec } of order) {
    if (!obj.visible) continue;
    const el = obj.element as HTMLElement;
    labelPos.copy(obj.position).project(camera);
    if (labelPos.z > 1) { el.classList.add('overlap'); continue; }
    const x = ((labelPos.x + 1) * w) / 2, y = ((1 - labelPos.y) * h) / 2;
    const hw = spec.text.length * (spec.major ? 3.6 : 3.3) + 6, hh = 8;
    const hit = placed.some(([px, py, pw, ph]) => Math.abs(px - x) < pw + hw && Math.abs(py - y) < ph + hh);
    el.classList.toggle('overlap', hit);
    if (!hit) placed.push([x, y, hw, hh]);
  }
}

if (window.innerWidth < 720) document.body.classList.add('panel-collapsed');

// ------------------------------------------------------------------ resize + loop
let panelPx = 0;
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  // Centre the scene in the space right of the panel rather than behind it.
  const panel = document.body.classList.contains('panel-collapsed') || w < 720 ? 0 : $('#panel').getBoundingClientRect().right;
  panelPx = panel;
  if (panel) camera.setViewOffset(w, h, -panel / 2, 0, w, h);
  else camera.clearViewOffset();
  camera.updateProjectionMatrix();
  // Converts a size in micrometres at distance d into pixels: px = size * scale / d.
  points.uniforms.uScale.value = (h * renderer.getPixelRatio()) / (2 * Math.tan((camera.fov * Math.PI) / 360));
}
new ResizeObserver(resize).observe(canvas);
resize();

applyColourBy();
setLabels(target.labels());
$('#layout-hint').textContent = LAYOUTS[0].hint;
document.querySelector('[data-layout="anatomical"]')!.setAttribute('aria-pressed', 'true');
camera.position.set(0, 150, 2200);
frameAll();
camera.position.copy(tween.toPos);
controls.target.copy(tween.toTarget);
tween.active = false;

const regionOf = new Map(meta.neuropils.map((p) => [p.name, p.region]));
const timer = new THREE.Timer();
const tmp = new THREE.Vector3();
renderer.setAnimationLoop(() => {
  timer.update();
  const dt = Math.min(timer.getDelta(), 0.05);

  // Layout transition: blend from wherever we were to the target.
  if (blendT < 1) {
    blendT = Math.min(1, blendT + dt / TRANSITION);
    const e = ease(blendT);
    for (let k = 0; k < current.length; k++) current[k] = from[k] + (targetPos[k] - from[k]) * e;
    points.position.needsUpdate = true;
  }

  // Anatomy meshes only make sense where neurons sit anatomically.
  const fadeTo = (v: number, goal: number) => (reducedMotion ? goal : v + (goal - v) * Math.min(1, dt * 4));
  shellFade = fadeTo(shellFade, target.shells);
  shellGroup.visible = showShells && shellFade > 0.02;
  brainShell.visible = showBrain && shellFade > 0.02 && layoutId !== 'mirrored';   // the outline has both hemispheres
  (brainShell.material as THREE.ShaderMaterial).uniforms.uOpacity.value = (focus ? 0.08 : 0.25) * shellFade;
  const focusNp = focus && ['neuropil', 'in_neuropil', 'out_neuropil'].includes(focus.grouping) ? G(focus.grouping).values[focus.code] : null;
  const focusRegion = focus?.grouping === 'region' ? G('region').values[focus.code] : null;
  neuropilShells.forEach((s, name) => {
    const lit = name === focusNp || regionOf.get(name) === focusRegion;
    const base = !focus ? 0.12 : lit ? 0.5 : 0.025;
    s.visible = !(layoutId === 'mirrored' && name.endsWith('_R'));
    s.material.uniforms.uOpacity.value = base * shellFade;
  });
  updateLabels();
  if (selected >= 0) {
    tmp.copy(V3(current, selected * 3));
    markerGeom.attributes.position.setXYZ(0, tmp.x, tmp.y, tmp.z);
    markerGeom.attributes.position.needsUpdate = true;
  }

  if (tween.active) {
    tween.t = Math.min(1, tween.t + dt / 1.1);
    const k = ease(tween.t);
    camera.position.lerpVectors(tween.fromPos, tween.toPos, k);
    controls.target.lerpVectors(tween.fromTarget, tween.toTarget, k);
    if (tween.t >= 1) tween.active = false;
  }
  fly(dt);
  controls.update();

  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  declutterLabels();

  // Hover picking, at most once per frame and only when the pointer moved.
  if (pointer) {
    const i = pick(pointer.x, pointer.y);
    pointer = null;
    if (i !== hovered) { hovered = i; refreshCard(i); }
    canvas.style.cursor = i >= 0 ? 'pointer' : '';
  }
});

loadingEl.classList.add('done');
setTimeout(() => loadingEl.remove(), 600);

// Read by scripts/ui_probe.py and handy from the devtools console.
Object.assign(window, {
  flywire: {
    groupings: byId, camera, controls, current,
    get state() {
      return {
        layout: layoutId, shown: shownId, blend: +blendT.toFixed(2), focus, colourBy, selected,
        partners: partnerState.size, labels: labels.filter((l) => l.obj.visible && !(l.obj.element as HTMLElement).classList.contains('overlap')).map((l) => l.spec.text).slice(0, 10),
      };
    },
    setLayout, setFocus, select, showMatrix,
  },
});
