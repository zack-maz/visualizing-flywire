// Connection matrix: synapses between the values of one grouping, rows = presynaptic, columns = postsynaptic.
import type { Connections, Grouping } from './data';
import { shortLabel, valueLabel } from './labels';

// One-hue sequential ramp (orange, OKLCH hue 50–62°), dark end first. Checked with the dataviz validator
// (--ordinal --mode dark --surface #0D0F12): monotone lightness, adjacent ΔL ≥ 0.06, dark end 2.2:1.
const RAMP = ['#6c4128', '#994a04', '#bf5900', '#d67620', '#dd9a65', '#e7ba97', '#ffd4af'];
const FIXED_ORDER = new Set(['super_class', 'flow', 'side', 'hub_band']);   // ordered groupings keep their order
const ALL_UP_TO = 24, TOP = 16;

export type Scale = 'share' | 'synapses';
export interface MatrixHooks {
  /** A neuron test while a cell or header is hovered (null when the pointer leaves). */
  onHover: (test: ((i: number) => boolean) | null) => void;
  /** A header was clicked: focus that value. */
  onPick: (grouping: string, code: number) => void;
}

const hexRgb = (h: string) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
const RAMP_RGB = RAMP.map(hexRgb);
/** t in (0, 1] -> ramp colour, interpolated between steps. */
function rampColour(t: number) {
  const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1), k = Math.min(RAMP.length - 2, Math.floor(x)), f = x - k;
  const [a, b] = [RAMP_RGB[k], RAMP_RGB[k + 1]];
  return `rgb(${a.map((v, c) => Math.round(v + (b[c] - v) * f)).join(',')})`;
}

interface Group { name: string; short: string; code: number }   // code -1 = "Other"

export function createMatrix(root: HTMLElement, hooks: MatrixHooks) {
  let scale: Scale = 'share';
  let last: { g: Grouping; conn: Connections } | null = null;

  const body = root.querySelector<HTMLElement>('.matrix-body')!;
  const readout = root.querySelector<HTMLElement>('.matrix-readout')!;
  const key = root.querySelector<HTMLElement>('.matrix-key')!;
  root.querySelectorAll<HTMLButtonElement>('[data-scale]').forEach((b) => b.addEventListener('click', () => {
    scale = b.dataset.scale as Scale;
    root.querySelectorAll('[data-scale]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    if (last) render(last.g, last.conn);
  }));

  function groupsOf(g: Grouping) {
    const ranked = g.counts.map((c, code) => ({ c, code })).filter((v) => v.code > 0 && v.c > 0);
    if (!FIXED_ORDER.has(g.id)) ranked.sort((a, b) => b.c - a.c);
    const keep = ranked.length <= ALL_UP_TO ? ranked : ranked.slice(0, TOP);
    const groups: Group[] = keep.map((v) => ({ name: valueLabel(g.id, g.values[v.code]), short: shortLabel(g.id, g.values[v.code]), code: v.code }));
    const groupOfCode = new Int16Array(g.values.length).fill(-1);
    keep.forEach((v, k) => (groupOfCode[v.code] = k));
    if (keep.length < ranked.length) {
      groups.push({ name: `${(ranked.length - keep.length).toLocaleString()} other values`, short: 'Other', code: -1 });
      for (const v of ranked.slice(TOP)) groupOfCode[v.code] = groups.length - 1;
    }
    if (g.counts[0]) { groups.push({ name: g.none, short: 'None', code: 0 }); groupOfCode[0] = groups.length - 1; }
    return { groups, groupOfCode };
  }

  function render(g: Grouping, conn: Connections) {
    last = { g, conn };
    const { groups, groupOfCode } = groupsOf(g);
    const k = groups.length, n = g.codes.length;
    const M = new Float64Array(k * k);
    for (let i = 0; i < n; i++) {
      const a = groupOfCode[g.codes[i]];
      for (let e = conn.outOffsets[i]; e < conn.outOffsets[i + 1]; e++) M[a * k + groupOfCode[g.codes[conn.outPartner[e]]]] += conn.outWeight[e];
    }
    const rowSum = new Float64Array(k), colSum = new Float64Array(k);
    let max = 0;
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const v = M[a * k + b]; rowSum[a] += v; colSum[b] += v; max = Math.max(max, v); }
    const logMax = Math.log10(max);
    const t = (a: number, b: number) => {
      const v = M[a * k + b];
      if (!v) return 0;
      return scale === 'share' ? v / rowSum[a] : Math.max(0.02, Math.log10(v) / logMax);
    };
    const members = (a: number) => (i: number) => groupOfCode[g.codes[i]] === a;

    const table = document.createElement('table');
    table.className = 'matrix';
    table.innerHTML = `<caption class="sr-only">Synapses from each ${g.label.toLowerCase()} (rows) to each ${g.label.toLowerCase()} (columns)</caption>`;
    const head = table.createTHead().insertRow();
    head.append(Object.assign(document.createElement('td'), { className: 'corner', innerHTML: '<span>from ↓ · to →</span>' }));
    const header = (grp: Group, a: number, scope: 'col' | 'row') => {
      const th = document.createElement('th');
      th.scope = scope;
      th.title = `${grp.name} · ${Math.round(scope === 'row' ? rowSum[a] : colSum[a]).toLocaleString()} synapses ${scope === 'row' ? 'out' : 'in'}`;
      th.innerHTML = '<button type="button"><span></span></button>';
      th.querySelector('span')!.textContent = grp.short;
      const btn = th.querySelector('button')!;
      btn.disabled = grp.code < 0;
      btn.addEventListener('click', () => hooks.onPick(g.id, grp.code));
      th.addEventListener('mouseenter', () => { hooks.onHover(members(a)); show(th.title); });
      return th;
    };
    groups.forEach((grp, b) => head.append(header(grp, b, 'col')));
    const tbody = table.createTBody();
    groups.forEach((from, a) => {
      const row = tbody.insertRow();
      row.append(header(from, a, 'row'));
      groups.forEach((to, b) => {
        const cell = row.insertCell();
        const v = M[a * k + b], tv = t(a, b);
        if (v) cell.style.background = rampColour(tv);
        const text = `${from.name} → ${to.name} · ${Math.round(v).toLocaleString()} synapses · `
          + `${pct(v / rowSum[a])} of ${from.short} output · ${pct(v / colSum[b])} of ${to.short} input`;
        cell.setAttribute('aria-label', text);
        cell.addEventListener('mouseenter', () => {
          const inA = members(a), inB = members(b);
          hooks.onHover((i) => inA(i) || inB(i));
          show(text);
        });
      });
    });
    table.addEventListener('mouseleave', () => { hooks.onHover(null); show(''); });
    body.replaceChildren(table);

    const lo = scale === 'share' ? '0%' : '1', hi = scale === 'share' ? '100%' : Math.round(max).toLocaleString();
    key.innerHTML = `<span>${lo}</span><span class="matrix-ramp" style="background:linear-gradient(90deg,${RAMP.join(',')})"></span><span>${hi}</span>`
      + `<span class="matrix-key-note">${scale === 'share' ? 'share of the row’s output' : 'synapses, log scale'}</span>`;
    show('');
  }

  const pct = (x: number) => (x ? (x < 0.001 ? '<0.1%' : `${(100 * x).toFixed(x < 0.1 ? 1 : 0)}%`) : '0%');
  function show(text: string) {
    readout.textContent = text || 'Hover a cell for its numbers · click a row or column name to focus it';
  }

  return { render };
}
