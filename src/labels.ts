// Display names for grouping values, and the categorical palette.

// Categorical colours: Tokyo Night hues from the brand palette, minus the accent blue (reserved
// for the one selected / focused thing). Order checked with the dataviz validator (dark, #0A0A0A):
// neighbouring slots stay apart under protan/deutan (worst ΔE 11.9) and normal vision (21.7).
// Some non-neighbouring pairs collide under colour-vision deficiency, so the legend's
// hover-to-isolate is the secondary encoding. The low-chroma lavender sits in the last slot.
export const PALETTE = ['#73DACA', '#FF9E64', '#7DCFFF', '#F7768E', '#2AC3DE', '#E0AF68', '#BB9AF7', '#9ECE6A', '#DB4B4B', '#C0CAF5'];
export const NEUTRAL = '#565F89';   // "other" and "none"

export const SUPER_CLASS_STYLE: Record<string, { label: string; blurb: string }> = {
  optic:              { label: 'Optic',              blurb: 'Local circuits inside the optic lobes' },
  visual_projection:  { label: 'Visual projection',  blurb: 'Carry vision from optic lobes to central brain' },
  visual_centrifugal: { label: 'Visual centrifugal', blurb: 'Feedback from central brain to optic lobes' },
  central:            { label: 'Central',            blurb: 'Local circuits inside the central brain' },
  sensory:            { label: 'Sensory',            blurb: 'Enter from eyes, antennae, taste and touch organs' },
  ascending:          { label: 'Ascending',          blurb: 'Come up from the nerve cord (incl. sensory ascending)' },
  descending:         { label: 'Descending',         blurb: 'Send commands down to the nerve cord' },
  motor:              { label: 'Motor',              blurb: 'Drive muscles of head, mouth and neck' },
  endocrine:          { label: 'Endocrine',          blurb: 'Release hormones' },
};

const FLOW_NAMES: Record<string, string> = {
  afferent: 'Afferent (into the brain)', intrinsic: 'Intrinsic (within the brain)', efferent: 'Efferent (out of the brain)',
};
const SIDE_NAMES: Record<string, string> = { left: 'Left', right: 'Right', center: 'Midline' };

const NEUROPIL_NAMES: Record<string, string> = {
  AL: 'Antennal lobe', AME: 'Accessory medulla', AMMC: 'Antennal mechanosensory & motor centre',
  AOTU: 'Anterior optic tubercle', ATL: 'Antler', AVLP: 'Anterior ventrolateral protocerebrum',
  BU: 'Bulb', CAN: 'Cantle', CRE: 'Crepine', EB: 'Ellipsoid body', EPA: 'Epaulette',
  FB: 'Fan-shaped body', FLA: 'Flange', GA: 'Gall', GNG: 'Gnathal ganglia', GOR: 'Gorget',
  IB: 'Inferior bridge', ICL: 'Inferior clamp', IPS: 'Inferior posterior slope', LAL: 'Lateral accessory lobe',
  LH: 'Lateral horn', LO: 'Lobula', LOP: 'Lobula plate', MB_CA: 'Mushroom body calyx',
  MB_ML: 'Mushroom body medial lobe', MB_PED: 'Mushroom body peduncle', MB_VL: 'Mushroom body vertical lobe',
  ME: 'Medulla', NO: 'Noduli', PB: 'Protocerebral bridge', PLP: 'Posterior lateral protocerebrum',
  PRW: 'Prow', PVLP: 'Posterior ventrolateral protocerebrum', SAD: 'Saddle', SCL: 'Superior clamp',
  SIP: 'Superior intermediate protocerebrum', SLP: 'Superior lateral protocerebrum',
  SMP: 'Superior medial protocerebrum', SPS: 'Superior posterior slope', VES: 'Vest', WED: 'Wedge',
  Unassigned: 'No synapses inside a mapped neuropil',
};

export function neuropilLabel(code: string): string {
  const m = code.match(/^(.*?)(?:_([LR]))?$/)!;
  const base = NEUROPIL_NAMES[m[1]] ?? m[1];
  return m[2] ? `${base} (${m[2] === 'L' ? 'left' : 'right'})` : base;
}

// Cell class codes from the FlyWire annotations. Optic classes are written as region codes:
// "ME>LO" = from medulla to lobula; "ME.LO" = within medulla and lobula; a single code = local to that region.
const CELL_CLASS_NAMES: Record<string, string> = {
  Kenyon_Cell: 'Kenyon cells (mushroom body)', CX: 'Central complex neurons', AN: 'Ascending neurons',
  ALPN: 'Antennal lobe projection neurons', ALLN: 'Antennal lobe local neurons',
  ALIN: 'Antennal lobe input neurons', ALON: 'Antennal lobe output neurons',
  LHLN: 'Lateral horn local neurons', LHCENT: 'Lateral horn centrifugal neurons',
  DAN: 'Dopaminergic neurons', MBON: 'Mushroom body output neurons', MBIN: 'Mushroom body input neurons',
  TuBu: 'Tubercle–bulb neurons', bilateral: 'Bilateral optic lobe neurons', optic_lobes: 'Other optic lobe neurons',
  pars_intercerebralis: 'Pars intercerebralis (neurosecretory)', pars_lateralis: 'Pars lateralis (neurosecretory)',
  brain_motor_neuron: 'Brain motor neurons', neck_motor_neuron: 'Neck motor neurons',
  visual: 'Visual sensory (photoreceptors)', olfactory: 'Olfactory receptor neurons',
  gustatory: 'Gustatory (taste) receptor neurons', mechanosensory: 'Mechanosensory neurons',
  hygrosensory: 'Hygrosensory (humidity) neurons', thermosensory: 'Thermosensory neurons',
  ocellar: 'Ocellar neurons', unknown_sensory: 'Unknown sensory neurons',
};
const OPTIC_REGIONS: Record<string, string> = { LA: 'lamina', ME: 'medulla', LO: 'lobula', LOP: 'lobula plate', AME: 'accessory medulla' };

/** "ME>LO_L" -> "Medulla → lobula (left)"; "unclassified:central_R" -> "Unclassified central (right)". */
export function cellClassLabel(group: string): string {
  const m = group.match(/^(.*?)(?:_([LR]))?$/)!;
  const side = m[2] ? ` (${m[2] === 'L' ? 'left' : 'right'})` : '';
  const code = m[1];
  if (code.startsWith('unclassified:')) {
    const sc = code.slice('unclassified:'.length);
    return `Unclassified ${SUPER_CLASS_STYLE[sc]?.label.toLowerCase() ?? sc}${side}`;
  }
  if (CELL_CLASS_NAMES[code]) return CELL_CLASS_NAMES[code] + side;
  const regions = (s: string) => s.split('.').map((r) => OPTIC_REGIONS[r] ?? r).join(' + ');
  if (/^[A-Z.>]+$/.test(code) && code.split(/[.>]/).every((r) => OPTIC_REGIONS[r])) {
    const [from, to] = code.split('>');
    const text = to ? `${regions(from)} → ${regions(to)}` : `${regions(from)} local`;
    return text[0].toUpperCase() + text.slice(1) + side;
  }
  return code + side;
}

// Neuropil super groups from Ito et al. 2014, split by side unless they hold midline neuropils.
const SUPER_GROUP_NAMES: Record<string, string> = {
  OL: 'Optic lobe', MB: 'Mushroom body', LH: 'Lateral horn', AL: 'Antennal lobe',
  SNP: 'Superior neuropils', INP: 'Inferior neuropils', LX: 'Lateral complex', CX: 'Central complex',
  VMNP: 'Ventromedial neuropils', VLNP: 'Ventrolateral neuropils', PENP: 'Periesophageal neuropils',
  Unassigned: 'No main neuropil',
};

/** "MB_L" -> "Mushroom body (left)" */
export function superGroupLabel(code: string): string {
  const m = code.match(/^(.*?)(?:_([LR]))?$/)!;
  const base = SUPER_GROUP_NAMES[m[1]] ?? m[1];
  return m[2] ? `${base} (${m[2] === 'L' ? 'left' : 'right'})` : base;
}

/** "MB_L" -> "Mushroom body L", short enough for a floating label. */
export function superGroupShortLabel(code: string): string {
  const m = code.match(/^(.*?)(?:_([LR]))?$/)!;
  return (m[1] === 'Unassigned' ? 'Unassigned' : SUPER_GROUP_NAMES[m[1]] ?? m[1]) + (m[2] ? ` ${m[2]}` : '');
}

/** Full display name of a value of a grouping, e.g. ("neuropil", "MB_CA_L") -> "Mushroom body calyx (left)". */
export function valueLabel(grouping: string, value: string): string {
  switch (grouping) {
    case 'side': return SIDE_NAMES[value] ?? value;
    case 'region': return superGroupLabel(value);
    case 'neuropil': case 'in_neuropil': case 'out_neuropil': return neuropilLabel(value);
    case 'soma_cluster': return value.replace(/^near (\S+)(.*)$/, (_, np, rest) => `Near ${neuropilLabel(np)}${rest}`);
    case 'flow': return FLOW_NAMES[value] ?? value;
    case 'super_class': return SUPER_CLASS_STYLE[value]?.label ?? value;
    case 'cell_class': return cellClassLabel(value);
    default: return value;
  }
}

const WIRING = new Set(['leiden_coarse', 'leiden_fine', 'infomap', 'conn_type', 'conn_kmeans', 'hub_band']);

/** Short name for a floating label: codes stay codes, with underscores as spaces. */
export function shortLabel(grouping: string, value: string): string {
  if (grouping === 'region') return superGroupShortLabel(value);
  if (WIRING.has(grouping)) return value.split(/ · |, /)[0];   // "M3 · OL_R 94%" -> "M3", "Core, top 1% …" -> "Core"
  if (grouping === 'super_class' || grouping === 'flow' || grouping === 'side') return valueLabel(grouping, value).replace(/ \(.*\)$/, '');
  return value.replace(/_/g, ' ');
}
