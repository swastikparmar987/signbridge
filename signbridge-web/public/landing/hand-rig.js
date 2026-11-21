/* Rigged GLB hand for the landing film.
   Ported from components/studio/glb-hand-rig.ts and the presets in components/studio/hand-3d-rig.ts
   so the landing page poses the exact same hand the studio uses. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const BONE_ORDER = [
  ['wrist', null],
  ['thumb-metacarpal', 'wrist'],
  ['thumb-phalanx-proximal', 'thumb-metacarpal'],
  ['thumb-phalanx-distal', 'thumb-phalanx-proximal'],
  ['thumb-tip', 'thumb-phalanx-distal'],
  ['index-finger-metacarpal', 'wrist'],
  ['index-finger-phalanx-proximal', 'index-finger-metacarpal'],
  ['index-finger-phalanx-intermediate', 'index-finger-phalanx-proximal'],
  ['index-finger-phalanx-distal', 'index-finger-phalanx-intermediate'],
  ['index-finger-tip', 'index-finger-phalanx-distal'],
  ['middle-finger-metacarpal', 'wrist'],
  ['middle-finger-phalanx-proximal', 'middle-finger-metacarpal'],
  ['middle-finger-phalanx-intermediate', 'middle-finger-phalanx-proximal'],
  ['middle-finger-phalanx-distal', 'middle-finger-phalanx-intermediate'],
  ['middle-finger-tip', 'middle-finger-phalanx-distal'],
  ['ring-finger-metacarpal', 'wrist'],
  ['ring-finger-phalanx-proximal', 'ring-finger-metacarpal'],
  ['ring-finger-phalanx-intermediate', 'ring-finger-phalanx-proximal'],
  ['ring-finger-phalanx-distal', 'ring-finger-phalanx-intermediate'],
  ['ring-finger-tip', 'ring-finger-phalanx-distal'],
  ['pinky-finger-metacarpal', 'wrist'],
  ['pinky-finger-phalanx-proximal', 'pinky-finger-metacarpal'],
  ['pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-proximal'],
  ['pinky-finger-phalanx-distal', 'pinky-finger-phalanx-intermediate'],
  ['pinky-finger-tip', 'pinky-finger-phalanx-distal'],
];

/* 21 joint order used for the glowing constellation (wrist, then thumb, index, middle, ring, pinky) */
export const JOINTS = [
  'wrist',
  'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
  ...['index', 'middle', 'ring', 'pinky'].flatMap(f => [
    `${f}-finger-phalanx-proximal`, `${f}-finger-phalanx-intermediate`, `${f}-finger-phalanx-distal`, `${f}-finger-tip`,
  ]),
];
export const LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

const F = (mcp, pip, dip, spread = 0) => ({ spread, mcp, pip, dip });
export const POSES = {
  OPEN:  { rotation: [0.06, 0.08, 0.02], thumb: F(0.24, 0.14, 0.08, 0.36), index: F(0.07, 0.1, 0.07, 0.11), middle: F(0.05, 0.09, 0.06, 0.02), ring: F(0.07, 0.11, 0.08, -0.06), pinky: F(0.1, 0.13, 0.09, -0.15) },
  HELLO: { rotation: [0.04, 0.15, -0.1], thumb: F(0.22, 0.18, 0.1, 0.32), index: F(0.09, 0.14, 0.08, 0.05), middle: F(0.07, 0.12, 0.08, 0.0), ring: F(0.09, 0.14, 0.09, -0.04), pinky: F(0.12, 0.16, 0.1, -0.09) },
  S:     { rotation: [0.06, 0.1, 0.0], thumb: F(1.5, 0.6, 0.4, 1.2), index: F(1.54, 1.62, 1.24, 0.02), middle: F(1.58, 1.64, 1.26, 0.0), ring: F(1.58, 1.64, 1.26, -0.02), pinky: F(1.54, 1.6, 1.22, -0.04) },
  A:     { rotation: [0.08, 0.1, 0.0], thumb: F(0.22, 0.28, 0.18, 0.34), index: F(1.48, 1.56, 1.15, 0.02), middle: F(1.52, 1.58, 1.18, 0.0), ring: F(1.52, 1.58, 1.18, -0.02), pinky: F(1.48, 1.52, 1.12, -0.05) },
  B:     { rotation: [0.04, 0.1, 0.0], thumb: F(0.95, 1.15, 0.8, 0.12), index: F(0.04, 0.06, 0.04, 0.02), middle: F(0.03, 0.05, 0.04, 0.0), ring: F(0.04, 0.06, 0.05, -0.02), pinky: F(0.06, 0.08, 0.06, -0.05) },
  C:     { rotation: [0.08, -0.45, 0.08], thumb: F(0.55, 0.55, 0.45, 0.42), index: F(0.68, 0.85, 0.65, 0.04), middle: F(0.7, 0.88, 0.68, 0.01), ring: F(0.72, 0.88, 0.68, -0.02), pinky: F(0.75, 0.88, 0.68, -0.05) },
  L:     { rotation: [0.05, 0.08, 0.0], thumb: F(0.15, 0.1, 0.06, 0.78), index: F(0.03, 0.05, 0.04, 0.04), middle: F(1.48, 1.56, 1.18, 0.0), ring: F(1.5, 1.58, 1.18, -0.02), pinky: F(1.48, 1.54, 1.15, -0.05) },
  O:     { rotation: [0.08, -0.52, 0.05], thumb: F(0.75, 0.78, 0.62, 0.34), index: F(0.98, 1.12, 0.9, 0.02), middle: F(1.02, 1.14, 0.92, 0.0), ring: F(1.04, 1.14, 0.92, -0.02), pinky: F(1.05, 1.14, 0.92, -0.05) },
  V:     { rotation: [0.04, 0.12, 0.0], thumb: F(0.98, 1.12, 0.85, 0.18), index: F(0.03, 0.06, 0.04, 0.18), middle: F(0.03, 0.06, 0.04, -0.14), ring: F(1.48, 1.56, 1.16, -0.02), pinky: F(1.48, 1.54, 1.14, -0.06) },
  W:     { rotation: [0.04, 0.12, 0.0], thumb: F(1.0, 1.14, 0.85, 0.18), index: F(0.04, 0.06, 0.04, 0.18), middle: F(0.03, 0.06, 0.04, 0.0), ring: F(0.05, 0.08, 0.05, -0.18), pinky: F(1.48, 1.54, 1.14, -0.06) },
  Y:     { rotation: [0.05, 0.1, 0.04], thumb: F(0.14, 0.12, 0.08, 0.72), index: F(1.48, 1.56, 1.16, 0.02), middle: F(1.52, 1.58, 1.18, 0.0), ring: F(1.48, 1.55, 1.15, -0.02), pinky: F(0.05, 0.08, 0.05, -0.22) },
  E:     { rotation: [0.06, 0.08, 0.0], thumb: F(1.05, 1.25, 0.95, 0.16), index: F(1.25, 1.5, 1.25, 0.03), middle: F(1.28, 1.52, 1.28, 0.0), ring: F(1.28, 1.52, 1.28, -0.02), pinky: F(1.25, 1.48, 1.22, -0.05) },
  TOUCH: { rotation: [0.02, 0.0, 0.0], thumb: F(0.3, 0.2, 0.1, 0.42), index: F(0.14, 0.16, 0.1, 0.04), middle: F(0.12, 0.15, 0.1, 0.0), ring: F(0.14, 0.17, 0.11, -0.04), pinky: F(0.18, 0.2, 0.12, -0.08) },
  LOVE:  { rotation: [0.04, 0.1, 0.0], thumb: F(0.16, 0.12, 0.08, 0.7), index: F(0.03, 0.06, 0.04, 0.08), middle: F(1.48, 1.56, 1.18, 0.0), ring: F(1.48, 1.56, 1.18, -0.02), pinky: F(0.05, 0.08, 0.05, -0.2) },
};
export const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];

const lerp = (a, b, t) => a + (b - a) * t;
const lerpFinger = (a, b, t) => ({ spread: lerp(a.spread, b.spread, t), mcp: lerp(a.mcp, b.mcp, t), pip: lerp(a.pip, b.pip, t), dip: lerp(a.dip, b.dip, t) });

/* Blend two poses with an independent amount per finger: amounts = { thumb, index, middle, ring, pinky, rot } */
export function blendPose(from, to, amounts, out = { rotation: [0, 0, 0] }) {
  for (const f of FINGERS) out[f] = lerpFinger(from[f], to[f], amounts[f]);
  for (let k = 0; k < 3; k++) out.rotation[k] = lerp(from.rotation[k], to.rotation[k], amounts.rot);
  return out;
}

/* Two skins with a fresnel rim: velvet (the signer, a silhouette drawn in light) and porcelain (the listener) */
const SKINS = {
  velvet: { mat: { color: 0x1c1526, roughness: 0.6, sheen: 0.22, sheenRoughness: 0.85, sheenColor: 0xff8a3d, clearcoat: 0.18 }, rim: 0xff9a52 },
  porcelain: { mat: { color: 0xb9a996, roughness: 0.42, sheen: 0.3, sheenRoughness: 0.6, sheenColor: 0xffd9b8, clearcoat: 0.35 }, rim: 0xffe3c8 },
};
function skinMaterial(kind = 'velvet') {
  const k = SKINS[kind] || SKINS.velvet;
  const mat = new THREE.MeshPhysicalMaterial({
    color: k.mat.color, roughness: k.mat.roughness, metalness: 0.0,
    sheen: k.mat.sheen, sheenRoughness: k.mat.sheenRoughness, sheenColor: new THREE.Color(k.mat.sheenColor),
    clearcoat: k.mat.clearcoat, clearcoatRoughness: 0.5,
  });
  const uniforms = { uRim: { value: 1.0 }, uRimColor: { value: new THREE.Color(k.rim) }, uInner: { value: 0.0 },
    uWrist: { value: new THREE.Vector3() }, uAxis: { value: new THREE.Vector3(0, 1, 0) }, uLen: { value: 1.0 } };
  mat.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    /* the forearm dissolves into the dark, so the cut end of the wrist never shows */
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vSbW;')
      .replace('#include <skinning_vertex>', '#include <skinning_vertex>\nvSbW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uRim; uniform vec3 uRimColor; uniform float uInner; uniform vec3 uWrist; uniform vec3 uAxis; uniform float uLen; varying vec3 vSbW;')
      .replace('#include <dithering_fragment>', `
        vec3 vdir = normalize(vViewPosition);
        float fr = pow(1.0 - clamp(dot(vdir, normal), 0.0, 1.0), 2.4);
        gl_FragColor.rgb += uRimColor * fr * uRim;
        gl_FragColor.rgb += vec3(1.0, 0.55, 0.25) * uInner * (0.35 + 0.65 * (1.0 - fr));
        float sbFade = smoothstep(-0.2 * uLen, 0.45 * uLen, dot(vSbW - uWrist, uAxis));
        gl_FragColor.rgb *= sbFade; gl_FragColor.a *= sbFade;
        #include <dithering_fragment>`);
  };
  mat.userData.uniforms = uniforms;
  return mat;
}

const loader = new GLTFLoader();

/* Loop subdivision for a skinned mesh. The WebXR hand is ~2.3k triangles, which looks faceted when it
   fills the screen. Each pass splits every triangle in four and smooths the surface (Loop weights);
   new vertices blend the bone weights of their edge, so the hand still bends exactly like before. */
function subdivideSkinned(src, passes = 2) {
  let pos, idx, sIdx, sW;
  {
    // weld seam duplicates by position so the surface subdivides as one connected skin
    const P = src.attributes.position, J = src.attributes.skinIndex, W = src.attributes.skinWeight, I = src.index.array;
    const key = i => `${P.getX(i).toFixed(5)},${P.getY(i).toFixed(5)},${P.getZ(i).toFixed(5)}`;
    const map = new Map(), remap = new Int32Array(P.count); pos = []; sIdx = []; sW = [];
    for (let i = 0; i < P.count; i++) {
      const k = key(i);
      if (!map.has(k)) { map.set(k, pos.length / 3); pos.push(P.getX(i), P.getY(i), P.getZ(i)); for (let c = 0; c < 4; c++) { sIdx.push(J.getComponent(i, c)); sW.push(W.getComponent(i, c)) } }
      remap[i] = map.get(k);
    }
    idx = Array.from(I, v => remap[v]);
  }
  const blendSkin = (a, b, out) => {
    const m = new Map();
    for (const [v, f] of [[a, 0.5], [b, 0.5]]) for (let c = 0; c < 4; c++) { const j = sIdx[v * 4 + c], w = sW[v * 4 + c] * f; if (w > 0) m.set(j, (m.get(j) || 0) + w) }
    const top = [...m.entries()].sort((x, y) => y[1] - x[1]).slice(0, 4), sum = top.reduce((s, e) => s + e[1], 0) || 1;
    for (let c = 0; c < 4; c++) { out.i.push(top[c] ? top[c][0] : 0); out.w.push(top[c] ? top[c][1] / sum : 0) }
  };
  for (let pass = 0; pass < passes; pass++) {
    const n = pos.length / 3, edges = new Map(), nbr = Array.from({ length: n }, () => new Set());
    const ek = (a, b) => a < b ? a * n + b : b * n + a;
    for (let t = 0; t < idx.length; t += 3) for (let e = 0; e < 3; e++) {
      const a = idx[t + e], b = idx[t + (e + 1) % 3], c = idx[t + (e + 2) % 3], k = ek(a, b);
      if (!edges.has(k)) edges.set(k, { a, b, opp: [] });
      edges.get(k).opp.push(c); nbr[a].add(b); nbr[b].add(a);
    }
    const boundary = new Uint8Array(n);
    edges.forEach(e => { if (e.opp.length < 2) { boundary[e.a] = 1; boundary[e.b] = 1 } });
    const np = [], ni = { i: [], w: [] };
    for (let v = 0; v < n; v++) {
      const N = [...nbr[v]], k = N.length;
      if (boundary[v] || k < 3) { np.push(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]) }
      else {
        const beta = k > 3 ? 3 / (8 * k) : 3 / 16, keep = 1 - k * beta;
        for (let c = 0; c < 3; c++) { let s = 0; for (const u of N) s += pos[u * 3 + c]; np.push(pos[v * 3 + c] * keep + s * beta) }
      }
      for (let c = 0; c < 4; c++) { ni.i.push(sIdx[v * 4 + c]); ni.w.push(sW[v * 4 + c]) }
    }
    const mid = new Map();
    edges.forEach((e, key) => {
      const id = np.length / 3;
      for (let c = 0; c < 3; c++) {
        const ab = pos[e.a * 3 + c] + pos[e.b * 3 + c];
        np.push(e.opp.length === 2 ? ab * 0.375 + (pos[e.opp[0] * 3 + c] + pos[e.opp[1] * 3 + c]) * 0.125 : ab * 0.5);
      }
      blendSkin(e.a, e.b, ni); mid.set(key, id);
    });
    const nidx = [];
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2], ab = mid.get(ek(a, b)), bc = mid.get(ek(b, c)), ca = mid.get(ek(c, a));
      nidx.push(a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca);
    }
    pos = np; sIdx = ni.i; sW = ni.w; idx = nidx;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(sIdx, 4));
  g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sW, 4));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

export async function loadHand(url = '/models/hand-right.glb', skin = 'velvet') {
  const gltf = await loader.loadAsync(url);
  const sceneRoot = gltf.scene;
  const material = skinMaterial(skin);
  sceneRoot.traverse(o => {
    if (o.isMesh) {
      if (o.isSkinnedMesh && o.geometry.index) { const g = subdivideSkinned(o.geometry, 2); o.geometry.dispose(); o.geometry = g }
      else o.geometry.computeVertexNormals();
      o.material = material; o.frustumCulled = false;
    }
  });

  const bones = new Map();
  for (const [name, parentName] of BONE_ORDER) {
    const obj = sceneRoot.getObjectByName(name);
    if (obj) bones.set(name, { bone: obj, restPos: obj.position.clone(), restQuat: obj.quaternion.clone(), parentName });
  }

  const box = new THREE.Box3().setFromObject(sceneRoot);
  const size = new THREE.Vector3(), center = new THREE.Vector3();
  box.getSize(size); box.getCenter(center);
  sceneRoot.position.set(-center.x, -center.y, -center.z);
  const align = new THREE.Group();
  align.add(sceneRoot);
  align.rotation.set(0, -Math.PI / 2, Math.PI);
  align.scale.setScalar(1.9 / Math.max(size.x, size.y, size.z, 0.001));

  const pivot = new THREE.Group(); pivot.add(align);
  const root = new THREE.Group(); root.add(pivot);
  return { root, pivot, bones, material };
}

function localBend(name, pose) {
  const key = FINGERS.find(f => name.startsWith(f));
  if (!key) return [0, 0];
  const s = pose[key];
  if (key === 'thumb') {
    if (name.endsWith('metacarpal')) return [-s.mcp * 0.45, s.spread * 0.45];
    if (name.endsWith('proximal')) return [-s.mcp * 0.72, s.spread * 0.25];
    if (name.endsWith('distal')) return [-s.pip * 0.85, 0];
    return [0, 0];
  }
  if (name.endsWith('proximal')) return [-s.mcp, s.spread];
  if (name.endsWith('intermediate')) return [-s.pip, 0];
  if (name.endsWith('distal')) return [-s.dip, 0];
  return [0, 0];
}

const _e = new THREE.Euler(), _q = new THREE.Quaternion(), _inv = new THREE.Quaternion(), _off = new THREE.Vector3();
const _tp = new THREE.Vector3(), _tq = new THREE.Quaternion();
const cache = new Map();

/* Forward kinematics over the flat bone list (bones are siblings under the armature) */
export function applyPose(rig, pose, blend = 1, extraRot = [0, 0, 0]) {
  rig.pivot.rotation.x += (pose.rotation[0] + extraRot[0] - rig.pivot.rotation.x) * blend;
  rig.pivot.rotation.y += (pose.rotation[1] + extraRot[1] - rig.pivot.rotation.y) * blend;
  rig.pivot.rotation.z += (pose.rotation[2] + extraRot[2] - rig.pivot.rotation.z) * blend;

  for (const [name, parentName] of BONE_ORDER) {
    const st = rig.bones.get(name);
    if (!st) continue;
    _tp.copy(st.restPos); _tq.copy(st.restQuat);
    if (parentName) {
      const ps = rig.bones.get(parentName), pc = cache.get(parentName);
      if (ps && pc) {
        _off.copy(st.restPos).sub(ps.restPos).applyQuaternion(pc.delta);
        _tp.copy(pc.pos).add(_off);
        _tq.copy(pc.delta).multiply(st.restQuat);
      }
    }
    const [cx, sy] = localBend(name, pose);
    if (cx || sy) { _e.set(cx, sy, 0, 'YXZ'); _q.setFromEuler(_e); _tq.multiply(_q); }
    st.bone.position.lerp(_tp, blend);
    st.bone.quaternion.slerp(_tq, blend);

    let c = cache.get(name);
    if (!c) { c = { pos: new THREE.Vector3(), delta: new THREE.Quaternion() }; cache.set(name, c); }
    c.pos.copy(st.bone.position);
    _inv.copy(st.restQuat).invert();
    c.delta.copy(st.bone.quaternion).multiply(_inv);
  }
}
