import * as THREE from 'three';
import type { MeshData } from './data';

// Per-neuron draw state, filled by the CPU whenever visibility, focus or selection changes.
export const HIDDEN = 0, SHOWN = 1, DIM = 2, INPUT = 3, OUTPUT = 4;

// One GPU point per neuron. Positions are owned by the CPU (main.ts blends layouts and uploads
// only while something moves); colour and state are per-neuron attributes.
export function createNeuronPoints(positions: Float32Array) {
  const n = positions.length / 3;
  const geom = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
  const color = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
  const state = new THREE.BufferAttribute(new Float32Array(n).fill(SHOWN), 1);
  geom.setAttribute('position', position);
  geom.setAttribute('aColor', color);
  geom.setAttribute('aState', state);

  // Shared by both passes, so one update drives both.
  const uniforms = {
    uSize: { value: 2.2 },
    uScale: { value: 1 },
  };
  // Pass 0 draws shown neurons solid with depth; pass 1 draws dimmed ones as a faint ghost
  // without depth, so dimmed neurons never hide shown ones. Partners of the selected neuron are
  // drawn larger: inputs as rings, outputs as solid discs (shape, not only colour).
  const makePass = (ghost: boolean) => new THREE.ShaderMaterial({
    transparent: ghost,
    depthWrite: !ghost,
    uniforms,
    vertexShader: /* glsl */ `
      uniform float uSize, uScale;
      attribute vec3 aColor;
      attribute float aState;
      varying vec3 vColor;
      varying float vRing;
      void main() {
        vColor = aColor;
        int s = int(aState + 0.5);
        vRing = s == ${INPUT} ? 1.0 : 0.0;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        // Size is in world micrometres, so points grow as you fly in.
        float size = max(1.5, uSize * uScale / -mv.z);
        if (s == ${INPUT} || s == ${OUTPUT}) size = max(7.0, size * 2.2);
        bool draw = ${ghost ? `s == ${DIM}` : `s == ${SHOWN} || s == ${INPUT} || s == ${OUTPUT}`};
        gl_PointSize = draw ? size${ghost ? ' * 0.7' : ''} : 0.0;
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      varying float vRing;
      void main() {
        vec2 d = gl_PointCoord - 0.5;
        float r = dot(d, d);
        if (r > 0.25 || (vRing > 0.5 && r < 0.09)) discard;
        ${ghost
          ? 'gl_FragColor = vec4(vColor, 0.05);'
          : 'gl_FragColor = vec4(vRing > 0.5 ? vColor : vColor * (1.0 - r * 0.8), 1.0);  // gentle sphere shading'}
      }
    `,
  });
  const solid = new THREE.Points(geom, makePass(false));
  const ghost = new THREE.Points(geom, makePass(true));
  // Positions change every frame during transitions; skip CPU-side bounds.
  solid.frustumCulled = ghost.frustumCulled = false;
  ghost.renderOrder = 1;
  const group = new THREE.Group();
  group.add(solid, ghost);
  return { group, uniforms, position, color, state };
}

// Translucent shells with a rim glow, readable from any angle without hiding the points.
export function createShellMaterial(color: string, strength: number) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: { uColor: { value: new THREE.Color(color) }, uOpacity: { value: strength } },
    vertexShader: /* glsl */ `
      varying vec3 vNormal, vView;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vView = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      varying vec3 vNormal, vView;
      void main() {
        float rim = 1.0 - abs(dot(normalize(vNormal), normalize(vView)));
        gl_FragColor = vec4(uColor, uOpacity * (0.08 + 0.92 * pow(rim, 2.5)));
      }
    `,
  });
}

export function meshGeometry(m: MeshData) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
  g.setIndex(new THREE.BufferAttribute(m.indices, 1));
  g.computeVertexNormals();
  return g;
}
