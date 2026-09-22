// Loads the files written by scripts/build_data.py and scripts/build_connectome.py.

export interface NeuronMeta {
  count: number;
  release: number;
  rootIds: string[];
  neuropils: { name: string; region: string }[];
}
export type GroupingRow = 'where' | 'what' | 'lineage' | 'wiring';
export interface GroupingMeta {
  id: string;
  label: string;
  row: GroupingRow;
  none: string;          // what value 0 means ("Untyped", "No soma in the brain", ...)
  sided: boolean;        // values already differ per hemisphere (neuropils, regions)
  values: string[];      // values[0] = "" (none)
  counts: number[];
  offset: number;
  bytes: 1 | 2;
}
export interface Grouping extends GroupingMeta {
  codes: Uint8Array | Uint16Array;   // value index per neuron
}
export interface MeshData { name: string; positions: Float32Array; indices: Uint32Array }
export interface Connections {
  outOffsets: Uint32Array; inOffsets: Uint32Array;
  outPartner: Uint32Array; inPartner: Uint32Array;
  outWeight: Uint16Array; inWeight: Uint16Array;
}

async function fetchWithProgress(url: string, onBytes: (n: number) => void = () => {}): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: ${res.status}`);
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

export async function loadAll(onProgress: (loadedMB: number) => void) {
  let loaded = 0;
  const tick = (n: number) => { loaded += n; onProgress(loaded / 1e6); };
  const json = async <T>(url: string) =>
    JSON.parse(new TextDecoder().decode(await fetchWithProgress(url, tick))) as T;

  const [meta, anatomicalBuf, groupingMeta, groupingBuf, meshIndex, meshBuf] = await Promise.all([
    json<NeuronMeta>('data/neurons.json'),
    fetchWithProgress('data/neurons.bin', tick),
    json<GroupingMeta[]>('data/groupings.json'),
    fetchWithProgress('data/groupings.bin', tick),
    json<{ name: string; vertexOffset: number; vertexCount: number; indexOffset: number; indexCount: number }[]>('data/meshes.json'),
    fetchWithProgress('data/meshes.bin', tick),
  ]);

  const n = meta.count;
  const groupings: Grouping[] = groupingMeta.map((g) => ({
    ...g,
    codes: g.bytes === 1 ? new Uint8Array(groupingBuf, g.offset, n) : new Uint16Array(groupingBuf, g.offset, n),
  }));
  const meshes: MeshData[] = meshIndex.map((m) => ({
    name: m.name,
    positions: new Float32Array(meshBuf, m.vertexOffset, m.vertexCount * 3),
    indices: new Uint32Array(meshBuf, m.indexOffset, m.indexCount),
  }));
  return { meta, anatomical: new Float32Array(anatomicalBuf), groupings, meshes };
}

/** A precomputed layout (layout_<id>.bin): float32 xyz per neuron. */
export async function loadLayout(id: string): Promise<Float32Array> {
  return new Float32Array(await fetchWithProgress(`data/layout_${id}.bin`));
}

/** Sensory-to-motor step per neuron (1 = afferent seed, 255 = never reached). */
export async function loadFlowRank(): Promise<Uint8Array> {
  return new Uint8Array(await fetchWithProgress('data/flow_rank.bin'));
}

let connections: Promise<Connections> | null = null;
/** Neuron pairs with >= 5 synapses, per direction, strongest first. ~35 MB, so only loaded when first needed. */
export function loadConnections(n: number): Promise<Connections> {
  connections ??= fetchWithProgress('data/connections.bin').then((buf) => {
    const e = (buf.byteLength - 2 * 4 * (n + 1)) / (2 * 4 + 2 * 2);
    let o = 0;
    const take = <T>(make: (b: ArrayBuffer, o: number, len: number) => T, len: number, size: number) => {
      const out = make(buf, o, len);
      o += len * size;
      return out;
    };
    const u32 = (b: ArrayBuffer, off: number, len: number) => new Uint32Array(b, off, len);
    const u16 = (b: ArrayBuffer, off: number, len: number) => new Uint16Array(b, off, len);
    return {
      outOffsets: take(u32, n + 1, 4), inOffsets: take(u32, n + 1, 4),
      outPartner: take(u32, e, 4), inPartner: take(u32, e, 4),
      outWeight: take(u16, e, 2), inWeight: take(u16, e, 2),
    };
  });
  return connections;
}
