/* SignBridge landing · 3D film ("The Bridge")
   Two real hands: the signer (velvet, right-hand GLB) and the listener (porcelain, left-hand GLB).
   A river of light carries signs from one to the other. Every value is driven by window.SB.scene,
   which story.js interpolates from scroll keyframes; this file only damps toward it and draws. */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { loadHand, applyPose, blendPose, POSES, JOINTS, LINKS, FINGERS } from './hand-rig.js';

const SB = window.SB;
const root = document.documentElement;
const canvas = document.getElementById('gl');
const mobile = innerWidth < 700;
const FINGER_ORDER = ['pinky', 'ring', 'middle', 'index', 'thumb'];

function fail() {
  root.classList.add('nogl');
  document.getElementById('fallback').hidden = false;
  SB.ready = true;
}
let renderer;
try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' }) }
catch (e) { fail(); throw e }
renderer.setPixelRatio(Math.min(devicePixelRatio, mobile ? 1.25 : 1.75));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;

const BGC = 0x09070D;
const scene = new THREE.Scene();
scene.background = new THREE.Color(BGC);
scene.fog = new THREE.FogExp2(BGC, 0.028);
const CAM_Z = 9, FOV = 30;
const camera = new THREE.PerspectiveCamera(FOV, innerWidth / innerHeight, 0.02, 100);
camera.position.set(0, 0, CAM_Z);

const ORANGE = new THREE.Color(0xFB6A18), AMBER = new THREE.Color(0xFFB27E);

/* ---------- Light: both hands are drawn by their edges ---------- */
scene.add(new THREE.AmbientLight(0x2a2238, 0.55));
const rimL = new THREE.DirectionalLight(0xFFB27E, 3.0); rimL.position.set(-4, 3, -3); scene.add(rimL);
const rimR = new THREE.DirectionalLight(0x9D8CFF, 2.0); rimR.position.set(4, 0.5, -3.5); scene.add(rimR);
const fill = new THREE.DirectionalLight(0xFAF6F0, 0.18); fill.position.set(0.5, 1, 6); scene.add(fill);
const back = new THREE.PointLight(0xFF8A3D, 6, 14, 1.6); back.position.set(0, 0.6, -3.2); scene.add(back);
const palmLight = new THREE.PointLight(0xFF8A3D, 0, 5, 2); scene.add(palmLight);

/* ---------- Textures ---------- */
function radialTex(stops, size = 256) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'), gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  stops.forEach(([o, col]) => gr.addColorStop(o, col));
  g.fillStyle = gr; g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const glowTex = radialTex([[0, 'rgba(255,214,170,1)'], [0.18, 'rgba(255,150,80,.65)'], [0.45, 'rgba(251,106,24,.18)'], [1, 'rgba(251,106,24,0)']]);
const coreTex = radialTex([[0, 'rgba(255,255,255,1)'], [0.3, 'rgba(255,236,214,.8)'], [1, 'rgba(255,200,150,0)']], 128);
const additive = { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending };

/* ---------- God rays behind the signer (the hand occludes them) ---------- */
const rays = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShaderMaterial({
  ...additive,
  uniforms: { uTime: { value: 0 }, uI: { value: 0 }, uColor: { value: new THREE.Color(0xFF8A3D) } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `
    varying vec2 vUv; uniform float uTime, uI; uniform vec3 uColor;
    void main(){
      vec2 p = vUv - 0.5; float r = length(p); float a = atan(p.y, p.x);
      float s1 = pow(0.5 + 0.5 * sin(a * 13.0 + uTime * 0.25 + sin(a * 3.0 - uTime * 0.2) * 1.6), 5.0);
      float s2 = pow(0.5 + 0.5 * sin(a * 29.0 - uTime * 0.4), 9.0) * 0.6;
      float fall = pow(smoothstep(0.5, 0.0, r), 2.2);
      float core = smoothstep(0.16, 0.0, r);
      vec3 col = uColor * ((s1 + s2) * 0.7 + 0.25) * fall + vec3(1.0, 0.8, 0.6) * core * 0.6;
      gl_FragColor = vec4(col * uI, 1.0);
    }`,
}));
scene.add(rays);
const sun = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, ...additive, opacity: 0 })); scene.add(sun);
const core = new THREE.Sprite(new THREE.SpriteMaterial({ map: coreTex, ...additive, opacity: 0 })); scene.add(core);

/* ---------- Shockwaves + the finale ring ---------- */
const ringMat = (k = 2.2) => new THREE.MeshBasicMaterial({ color: ORANGE.clone().multiplyScalar(k), ...additive, opacity: 0, side: THREE.DoubleSide, depthTest: false });
const shocks = [new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 160), ringMat()), new THREE.Mesh(new THREE.RingGeometry(0.99, 1, 160), ringMat())];
shocks.forEach(r => { r.renderOrder = 12; scene.add(r) });
const halo = new THREE.Mesh(new THREE.TorusGeometry(1, 0.012, 12, 220), ringMat(2.6)); halo.renderOrder = 12; scene.add(halo);
const haloGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, ...additive, opacity: 0 })); scene.add(haloGlow);

/* ---------- Dust ---------- */
const DUST = mobile ? 900 : 2000;
const dGeo = new THREE.BufferGeometry();
const dPos = new Float32Array(DUST * 3), dSeed = new Float32Array(DUST);
for (let i = 0; i < DUST; i++) {
  dPos[i * 3] = (Math.random() - 0.5) * 18; dPos[i * 3 + 1] = (Math.random() - 0.5) * 10; dPos[i * 3 + 2] = -8 + Math.random() * 10;
  dSeed[i] = Math.random();
}
dGeo.setAttribute('position', new THREE.BufferAttribute(dPos, 3));
dGeo.setAttribute('aSeed', new THREE.BufferAttribute(dSeed, 1));
const dust = new THREE.Points(dGeo, new THREE.ShaderMaterial({
  ...additive,
  uniforms: { uTime: { value: 0 }, uBurst: { value: 0 }, uCenter: { value: new THREE.Vector3() }, uSize: { value: 30 * renderer.getPixelRatio() }, uO: { value: 0 }, uTint: { value: new THREE.Color(0xFF8A3D) } },
  vertexShader: `
    attribute float aSeed; uniform float uTime, uBurst, uSize; uniform vec3 uCenter; varying float vA; varying float vS;
    void main(){
      vec3 p = position;
      p.x += sin(uTime * 0.11 + aSeed * 40.0) * 0.35;
      p.y += mod(uTime * (0.02 + aSeed * 0.05) + aSeed * 10.0, 10.0) - 5.0;
      p.z += cos(uTime * 0.09 + aSeed * 23.0) * 0.3;
      vec3 d = p - uCenter; float l = max(length(d), 0.2);
      p += d / l * uBurst * (1.2 + aSeed * 2.4) / (0.6 + l * 0.25);
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = uSize * (0.25 + aSeed * aSeed) / -mv.z;
      gl_Position = projectionMatrix * mv;
      vA = (0.35 + 0.65 * fract(aSeed * 7.31)) * (0.55 + 0.45 * sin(uTime * (0.8 + aSeed * 2.0) + aSeed * 60.0));
      vS = aSeed;
    }`,
  fragmentShader: `
    uniform float uO; uniform vec3 uTint; varying float vA; varying float vS;
    void main(){
      float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;
      float a = smoothstep(0.5, 0.0, d);
      vec3 c = mix(vec3(0.95, 0.93, 0.9), uTint, step(0.55, fract(vS * 3.7)));
      gl_FragColor = vec4(c * a * vA * uO, 1.0);
    }`,
}));
dust.frustumCulled = false; scene.add(dust);

/* ---------- The bridge: a river of light from the signer's fingertip to the listener's palm ---------- */
const BN = mobile ? 700 : 1400;
const bGeoP = new THREE.BufferGeometry();
const bSeed = new Float32Array(BN), bSpd = new Float32Array(BN), bLane = new Float32Array(BN * 2), bSize = new Float32Array(BN), bComet = new Float32Array(BN);
for (let i = 0; i < BN; i++) {
  bSeed[i] = Math.random(); bSpd[i] = Math.random();
  const a = Math.random() * Math.PI * 2, r = Math.pow(Math.random(), 1.6);
  bLane[i * 2] = Math.cos(a) * r; bLane[i * 2 + 1] = Math.sin(a) * r;
  bSize[i] = 18 + Math.random() * 28; bComet[i] = i < 6 ? 1 : 0;
}
bGeoP.setAttribute('position', new THREE.BufferAttribute(new Float32Array(BN * 3), 3));
bGeoP.setAttribute('aSeed', new THREE.BufferAttribute(bSeed, 1));
bGeoP.setAttribute('aSpd', new THREE.BufferAttribute(bSpd, 1));
bGeoP.setAttribute('aLane', new THREE.BufferAttribute(bLane, 2));
bGeoP.setAttribute('aSize', new THREE.BufferAttribute(bSize, 1));
bGeoP.setAttribute('aComet', new THREE.BufferAttribute(bComet, 1));
const bridgeMat = new THREE.ShaderMaterial({
  ...additive,
  uniforms: {
    uP0: { value: new THREE.Vector3() }, uP1: { value: new THREE.Vector3() }, uP2: { value: new THREE.Vector3() }, uP3: { value: new THREE.Vector3() },
    uPerp: { value: new THREE.Vector3(0, 1, 0) }, uTime: { value: 0 }, uBridge: { value: 0 }, uBrk: { value: 0 },
    uPR: { value: renderer.getPixelRatio() }, uColA: { value: new THREE.Color(0xFB6A18) }, uColB: { value: new THREE.Color(0xFFE3C8) }, uWidth: { value: 0.09 },
  },
  vertexShader: `
    attribute float aSeed, aSpd, aSize, aComet; attribute vec2 aLane;
    uniform vec3 uP0, uP1, uP2, uP3, uPerp; uniform float uTime, uBridge, uBrk, uPR, uWidth;
    varying float vA; varying float vU; varying float vC;
    vec3 bez(float u){ float v = 1.0 - u; return v*v*v*uP0 + 3.0*v*v*u*uP1 + 3.0*v*u*u*uP2 + u*u*u*uP3; }
    void main(){
      float u = fract(aSeed + uTime * (0.1 + 0.1 * aSpd) * (aComet > 0.5 ? 1.6 : 1.0));
      vec3 p = bez(u);
      float env = sin(u * 3.14159);
      vec3 side = normalize(cross(normalize(uP3 - uP0), vec3(0.0, 0.0, 1.0)));
      p += (uPerp * aLane.y + side * aLane.x * 0.6 + vec3(0.0, 0.0, aLane.x * 0.5)) * uWidth * env * (1.0 + aComet * -0.8);
      p += vec3(sin(uTime * 1.7 + aSeed * 40.0), cos(uTime * 1.3 + aSeed * 31.0), sin(uTime * 1.1 + aSeed * 17.0)) * 0.025 * env;
      float b = max(u - 0.38, 0.0);
      p.y -= uBrk * b * b * 16.0 * (0.5 + aSpd);
      p.x += uBrk * b * aLane.x * 2.2;
      p.z += uBrk * b * aLane.y * 1.5;
      float fade = env * (1.0 - uBrk * smoothstep(0.45, 0.95, u)) * (1.0 + uBrk * 0.4);
      vA = fade * uBridge * (0.45 + 0.55 * aSpd) * (1.0 + aComet * 1.4);
      vU = u; vC = aComet;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = aSize * uPR * (1.0 + aComet * 1.6) / -mv.z;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    uniform vec3 uColA, uColB; varying float vA; varying float vU; varying float vC;
    void main(){
      float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;
      float a = pow(smoothstep(0.5, 0.0, d), 1.6);
      vec3 c = mix(uColA, uColB, smoothstep(0.2, 1.0, vU) * 0.7 + vC * 0.3) * (1.6 + vC * 1.4);
      gl_FragColor = vec4(c * a * vA, 1.0);
    }`,
});
const bridge = new THREE.Points(bGeoP, bridgeMat); bridge.frustumCulled = false; bridge.renderOrder = 9; scene.add(bridge);

/* ---------- Constellation on the signer: the hand, understood ---------- */
const TIP_OF = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const TIPS = Object.values(TIP_OF);
const jGeo = new THREE.IcosahedronGeometry(1, 2);
const joints = JOINTS.map((_, i) => {
  const m = new THREE.Mesh(jGeo, new THREE.MeshBasicMaterial({ color: (TIPS.includes(i) ? AMBER : ORANGE).clone().multiplyScalar(2.4), ...additive, opacity: 0, depthTest: false }));
  m.renderOrder = 11; m.userData.r = TIPS.includes(i) ? 0.022 : i === 0 ? 0.024 : 0.016; scene.add(m); return m;
});
const cyl = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true);
const linkMat = new THREE.MeshBasicMaterial({ color: ORANGE.clone().multiplyScalar(1.7), ...additive, opacity: 0, depthTest: false });
const links = LINKS.map(() => { const m = new THREE.Mesh(cyl, linkMat); m.renderOrder = 10; scene.add(m); return m });
const flares = FINGERS.map(() => { const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, ...additive, opacity: 0, depthTest: false })); s.renderOrder = 13; scene.add(s); return s });

/* ---------- Post ---------- */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bokeh = new BokehPass(scene, camera, { focus: 1, aperture: 0, maxblur: 0.014 }); bokeh.enabled = false; composer.addPass(bokeh);
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.8, 0.55, 0.6); composer.addPass(bloom);
const lens = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, uCA: { value: 0.0015 }, uVig: { value: 0.7 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uCA, uVig; varying vec2 vUv;
    void main(){
      vec2 d = vUv - 0.5;
      vec3 col = vec3(texture2D(tDiffuse, vUv + d * uCA).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - d * uCA).b);
      col *= mix(1.0, smoothstep(0.95, 0.25, length(d * vec2(1.0, 0.85))), uVig);
      gl_FragColor = vec4(col, 1.0);
    }`,
});
composer.addPass(lens);
composer.addPass(new OutputPass());
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight);
});

/* ---------- Hands ---------- */
function makeHand() {
  const g = new THREE.Group(); scene.add(g);
  return { g, rig: null, pose: blendPose(POSES.S, POSES.S, { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, rot: 0 }, { rotation: [0, 0, 0] }),
    tgt: { rotation: [0, 0, 0] }, st: { x: 0, y: 0, s: 1, rx: 0, ry: 0, rz: 0, vis: 0 }, pts: JOINTS.map(() => new THREE.Vector3()),
    palmC: new THREE.Vector3(), palmN: new THREE.Vector3(0, 0, 1), init: false };
}
const H = { L: makeHand(), R: makeHand() };
async function setup(h, url, skin) {
  const rig = await loadHand(url, skin);
  /* centre and size each hand from its joints (a skinned mesh's bounding box is unreliable before
     the first render). Measured detached, so the live group transform can't skew it. */
  applyPose(rig, POSES.OPEN, 1); rig.root.updateMatrixWorld(true);
  const ps = JOINTS.map(n => rig.bones.get(n).bone.getWorldPosition(new THREE.Vector3()));
  const c = ps.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(1 / ps.length);
  const k = 1.6 / Math.max(1e-3, ps[0].distanceTo(ps[12]));
  rig.root.scale.setScalar(k);
  rig.root.position.copy(c).multiplyScalar(-k);
  h.g.add(rig.root); h.rig = rig;
  applyPose(rig, POSES.S, 1);
}
SB.loadP = 0.4;
Promise.all([setup(H.L, '/models/hand-right.glb', 'velvet'), setup(H.R, '/models/hand-left.glb', 'porcelain')])
  .then(() => {
    const m = H.R.rig.material; m.transparent = true; m.opacity = 0;
    H.L.rig.material.transparent = true; // lets the forearm dissolve into the dark
    SB.loadP = 1; SB.ready = true;
  })
  .catch(() => { fail(); canvas.style.opacity = 0 });

/* ---------- Loop ---------- */
const V = THREE.Vector3;
const damp = (a, b, l, dt) => a + (b - a) * (1 - Math.exp(-l * dt));
const clamp01 = v => Math.max(0, Math.min(1, v));
const easeOut = t => 1 - Math.pow(1 - t, 3);
const stag = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, rot: 0 }, kAll = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, rot: 0 };
const tmp = new V(), tmp2 = new V(), tmp3 = new V(), up = new V(0, 1, 0), look = new V(), camT = new V(0, 0, CAM_Z), focusT = new V(), dirV = new V(), macroCam = new V();
const sideA = new V(0.5, 0.25, 0.1), sideB = new V(-0.45, 0.3, 0.15), tint = new THREE.Color(0xFF8A3D), tintT = new THREE.Color();
const G = { ov: 0, glow: 0, rays: 0, bridge: 0, brk: 0, ring: 0, dim: 0 };
let t = 0, last = performance.now(), hidden = false, burstAt = -10, lastBurst = 0;
document.addEventListener('visibilitychange', () => { hidden = document.hidden; last = performance.now() });

const OPENER_L = { x: 0, y: 0.3, s: 0.7, rx: 0, ry: 0.25, rz: 0, vis: 1, from: 'HELLO', to: 'HELLO', t: 1 };
const HIDDEN_R = { x: 1.5, y: -1.3, s: 1, rx: 0, ry: 0, rz: 0, vis: 0, from: 'OPEN', to: 'OPEN', t: 1 };

function poseTarget(h, from, to, tt) {
  FINGER_ORDER.forEach((f, k) => { stag[f] = easeOut(clamp01(tt * 1.6 - k * 0.15)) });
  stag.rot = tt;
  blendPose(POSES[from] || POSES.OPEN, POSES[to] || POSES.OPEN, stag, h.tgt);
}
function place(h, S, dt, k, isL, m, beat, pulse) {
  const halfH = Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * CAM_Z, halfW = halfH * camera.aspect;
  const st = h.st;
  if (!h.init) { Object.assign(st, { x: S.x, y: S.y, s: S.s, rx: S.rx, ry: S.ry, rz: S.rz, vis: S.vis }); h.init = true }
  for (const key of ['x', 'y', 's', 'rx', 'ry', 'rz']) st[key] = damp(st[key], S[key], k, dt);
  st.vis = damp(st.vis, S.vis, 2.4, dt);
  const sway = SB.reduce ? 0 : 1;
  h.g.position.set(st.x * halfW, st.y * halfH + Math.sin(t * 0.9 + (isL ? 0 : 1.7)) * 0.04 * sway, 0);
  h.g.scale.setScalar(st.s * (1 + Math.sin(t * 1.1) * 0.005 * sway + (isL ? beat * 0.012 + pulse * 0.03 : 0)));
  h.g.rotation.set(st.rx - m.y * 0.08 + Math.sin(t * 0.5 + (isL ? 0 : 2)) * 0.03 * sway, st.ry + m.x * 0.12 + Math.sin(t * 0.35) * 0.06 * sway, st.rz + Math.sin(t * 0.27) * 0.02 * sway);
}
function joints3(h) {
  const { rig, pts } = h; if (!rig) return;
  rig.root.updateMatrixWorld(true);
  JOINTS.forEach((name, i) => rig.bones.get(name).bone.getWorldPosition(pts[i]));
  const wu = rig.material.userData.uniforms;
  wu.uWrist.value.copy(pts[0]); wu.uAxis.value.subVectors(pts[9], pts[0]); wu.uLen.value = wu.uAxis.value.length(); wu.uAxis.value.normalize();
  h.palmC.copy(pts[0]).lerp(pts[9], 0.62);
  tmp.subVectors(pts[5], pts[0]); tmp2.subVectors(pts[17], pts[0]);
  h.palmN.crossVectors(tmp, tmp2).normalize();
  if (h.palmN.dot(tmp.subVectors(camera.position, h.palmC)) < 0) h.palmN.negate();
}

function frame(now) {
  requestAnimationFrame(frame);
  if (hidden) return;
  const dt = Math.min(0.05, (now - last) / 1000); last = now; t += dt;
  const S = SB.scene, opening = SB.phase !== 'film', shot = SB.phase === 'choice' ? 0 : SB.shot, shotP = SB.shotP || 0;
  const m = SB.reduce ? { x: 0, y: 0 } : SB.mouse;
  if (SB.burst !== lastBurst) { lastBurst = SB.burst; burstAt = t }
  const bAge = t - burstAt, burstK = bAge < 3 ? Math.exp(-bAge * 1.6) : 0;
  const beatAge = (now - SB.beat) / 1000, beat = beatAge < 0.8 ? Math.exp(-beatAge * 6) + (beatAge > 0.24 ? Math.exp(-(beatAge - 0.24) * 7) * 0.6 : 0) : 0;
  const pulseAge = (now - SB.pulse) / 1000, pulse = pulseAge < 1.2 ? Math.exp(-pulseAge * 3) : 0;
  const ready = H.L.rig && H.R.rig;

  /* targets: the opener frames the signer alone; afterwards the keyframed film takes over */
  const SL = opening || !S ? OPENER_L : S.L, SR = opening || !S ? HIDDEN_R : S.R;
  const goal = opening || !S ? { ov: shot < 3 ? 0.95 : shot === 3 ? 0.95 - shotP * 0.6 : 0.35, glow: 0.3, rays: shot >= 3 ? 0.55 : 0, bridge: 0, brk: 0, ring: 0, dim: 0 } : S;
  if (opening) { SL.from = SL.to = SB.pose }
  const present = ready && !(opening && shot === 0) ? 1 : 0;
  G.pres = damp(G.pres || 0, present, 1.6, dt);
  for (const key of ['ov', 'glow', 'rays', 'bridge', 'brk', 'ring', 'dim']) G[key] = damp(G[key], goal[key], key === 'brk' ? 2.5 : 3.2, dt);
  if (S && !opening) tintT.setRGB(S.tint[0], S.tint[1], S.tint[2]); else tintT.set(0xFF8A3D);
  tint.lerp(tintT, 1 - Math.exp(-2 * dt));

  /* poses (staggered finger by finger, then damped) */
  const kp = 1 - Math.exp(-dt * 7);
  for (const key in kAll) kAll[key] = kp;
  for (const [h, SS] of [[H.L, SL], [H.R, SR]]) {
    poseTarget(h, SS.from, SS.to, SS.t);
    blendPose(h.pose, h.tgt, kAll, h.pose);
  }
  place(H.L, SL, dt, opening ? 2.2 : 3.2, true, m, beat, pulse);
  place(H.R, SR, dt, 3.2, false, m, 0, 0);
  if (H.L.rig) applyPose(H.L.rig, H.L.pose, 1);
  if (H.R.rig) applyPose(H.R.rig, H.R.pose, 1);
  joints3(H.L); joints3(H.R);

  /* camera: gentle parallax, or the opener's macro shots */
  camT.set(m.x * 0.3, m.y * 0.18, CAM_Z); look.set(m.x * 0.05, m.y * 0.03, 0);
  if (opening && shot < 4 && H.L.rig) {
    const P = H.L.pts, ez = shotP * shotP * (3 - 2 * shotP);
    const A = shot <= 1, p = shot === 0 ? 0 : shot >= 3 ? 1 : ez;
    focusT.copy(A ? P[5] : P[12]).lerp(A ? P[8] : P[9], p);
    dirV.copy(H.L.palmN).add(A ? sideA : sideB).normalize();
    const dist = A ? 0.68 - 0.08 * p : 0.85 - 0.15 * p;
    macroCam.copy(focusT).addScaledVector(dirV, dist);
    const w = shot === 3 ? (shotP < 0.5 ? 4 * shotP ** 3 : 1 - Math.pow(-2 * shotP + 2, 3) / 2) : 0;
    tmp3.copy(camT); camT.copy(macroCam).lerp(tmp3, w);
    tmp3.copy(look); look.copy(focusT).lerp(tmp3, w);
    bokeh.enabled = w < 0.98;
    bokeh.uniforms.focus.value = THREE.MathUtils.lerp(dist, camT.distanceTo(look), w);
    bokeh.uniforms.aperture.value = 0.02 * (1 - w);
    camera.position.copy(camT);
  } else {
    bokeh.enabled = false;
    camera.position.x = damp(camera.position.x, camT.x, 3, dt); camera.position.y = damp(camera.position.y, camT.y, 3, dt); camera.position.z = damp(camera.position.z, camT.z, 3, dt);
  }
  camera.lookAt(look);

  /* hand materials: velvet rim breathes with the heartbeat; the listener fades in and out */
  if (H.L.rig) {
    const u = H.L.rig.material.userData.uniforms;
    u.uRim.value = G.pres * (0.32 + beat * 0.35 + G.glow * 0.2) * (1 - G.dim * 0.4);
    u.uInner.value = G.pres * (G.glow * 0.12 + burstK * 0.25 + pulse * 0.08);
    H.L.rig.root.visible = G.pres > 0.001;
  }
  if (H.R.rig) {
    const v = H.R.st.vis * G.pres, mat = H.R.rig.material;
    mat.opacity = clamp01(v * 1.2); mat.userData.uniforms.uRim.value = 0.45 * v; mat.userData.uniforms.uInner.value = G.bridge * 0.08 * v;
    H.R.rig.root.visible = v > 0.01;
  }

  /* palm light, sun and rays on the signer */
  const L = H.L, s = L.st.s, g = G.glow * G.pres + burstK * 0.9;
  sun.position.copy(L.palmC).addScaledVector(L.palmN, 0.22 * s);
  sun.scale.setScalar(s * (0.3 + g * 1.0)); sun.material.opacity = Math.min(1, g * 0.8); sun.material.color.copy(tint).lerp(new THREE.Color(1, 1, 1), 0.5);
  core.position.copy(L.palmC).addScaledVector(L.palmN, 0.24 * s);
  core.scale.setScalar(s * (0.05 + g * 0.18)); core.material.opacity = Math.min(1, g * 0.9);
  palmLight.position.copy(L.palmC).addScaledVector(L.palmN, 0.3 * s); palmLight.intensity = g * 0.9; palmLight.color.copy(tint);
  back.color.copy(tint); back.intensity = (2 + G.rays * 6) * G.pres * (1 - G.dim * 0.6);
  rays.position.copy(L.palmC).addScaledVector(L.palmN, -0.7 * s); rays.quaternion.copy(camera.quaternion);
  rays.scale.setScalar(Math.min(2.2, s) * (0.7 + G.rays * 0.3 + burstK * 0.4));
  rays.material.uniforms.uTime.value = t; rays.material.uniforms.uI.value = (G.rays + burstK * 0.6) * G.pres * (1 - G.dim * 0.7); rays.material.uniforms.uColor.value.copy(tint);

  /* the bridge */
  const P0 = tmp.copy(L.pts[8]), P3 = tmp2.copy(H.R.palmC).addScaledVector(H.R.palmN, 0.12 * H.R.st.s);
  const u = bridgeMat.uniforms, d = tmp3.subVectors(P3, P0), len = d.length();
  const perp = Math.abs(d.y) > Math.abs(d.x) ? new V(1, 0, 0) : new V(0, 1, 0);
  u.uP0.value.copy(P0); u.uP3.value.copy(P3);
  u.uP1.value.copy(P0).addScaledVector(d, 0.33).addScaledVector(perp, len * 0.28);
  u.uP2.value.copy(P0).addScaledVector(d, 0.66).addScaledVector(perp, len * 0.28);
  u.uPerp.value.copy(perp); u.uTime.value = SB.reduce ? 0.5 : t;
  u.uBridge.value = G.bridge * G.pres * (ready ? 1 : 0) * Math.min(1, H.R.st.vis * 1.5 + 0.2); u.uBrk.value = G.brk;
  u.uColA.value.copy(tint).lerp(ORANGE, 0.5);

  /* finale: the bridge collapses into a ring between the two palms */
  const mid = tmp3.copy(L.palmC).lerp(H.R.palmC, 0.5);
  halo.position.copy(mid); halo.quaternion.copy(camera.quaternion);
  const ringR = 0.62 * Math.max(0.5, L.st.s); // sized to the hands, not the gap between them
  halo.scale.setScalar((0.3 + G.ring * 0.7) * ringR + Math.sin(t * 1.4) * 0.02 * G.ring); halo.material.opacity = G.ring * 0.9;
  haloGlow.position.copy(mid); haloGlow.scale.setScalar((0.6 + G.ring * 2.2) * ringR); haloGlow.material.opacity = G.ring * 0.3;
  const shockAt = G.ring > 0.3 ? mid : L.palmC;
  shocks.forEach((r, i) => {
    const a = bAge - i * 0.16;
    r.position.copy(shockAt); r.quaternion.copy(camera.quaternion);
    if (a > 0 && a < 1.8) { const e = easeOut(a / 1.8); r.scale.setScalar(0.15 + e * (i ? 3.2 : 4.6)); r.material.opacity = (1 - e) * (i ? 0.6 : 1) } else r.material.opacity = 0;
  });

  /* constellation on the signer */
  const ov = G.ov * G.pres, thin = opening && shot < 3 ? 0.7 : 1, toCam = new V();
  joints.forEach((j, i) => {
    toCam.subVectors(camera.position, L.pts[i]).normalize();
    j.position.copy(L.pts[i]).addScaledVector(toCam, 0.03 * s);
    const tip = TIPS.includes(i), tw = 1 + Math.sin(t * 3 - i * 0.4) * 0.15 * ov;
    j.scale.setScalar(j.userData.r * s * (0.4 + 0.6 * thin) * tw * (1 + (tip ? burstK + pulse * 0.6 : 0)));
    j.material.opacity = ov;
  });
  links.forEach((l, i) => {
    const [a, b] = LINKS[i], pa = joints[a].position, pb = joints[b].position, ln = pa.distanceTo(pb);
    l.position.copy(pa).add(pb).multiplyScalar(0.5);
    if (ln > 1e-4) l.quaternion.setFromUnitVectors(up, toCam.subVectors(pb, pa).normalize());
    const w = 0.0045 * thin * s; l.scale.set(w, Math.max(ln, 1e-4), w);
  });
  linkMat.opacity = ov * 0.85;
  FINGER_ORDER.forEach((name, k) => {
    const fl = flares[k], age = (now - SB.flare[k]) / 1000;
    fl.position.copy(joints[TIP_OF[name]].position);
    if (age < 0.9) { const e = Math.exp(-age * 4); fl.scale.setScalar(s * (0.08 + (1 - e) * 0.3)); fl.material.opacity = e } else fl.material.opacity = 0;
  });

  /* dust */
  const du = dust.material.uniforms;
  du.uTime.value = SB.reduce ? 0 : t; du.uCenter.value.copy(shockAt); du.uBurst.value = burstK * 1.2;
  du.uO.value = G.pres * (0.5 + G.glow * 0.3) * (1 - G.dim * 0.5); du.uTint.value.copy(tint);

  /* post */
  bloom.strength = 0.75 + G.glow * 0.35 + burstK * 0.7 + G.ring * 0.4;
  lens.uniforms.uCA.value = 0.0012 + burstK * 0.005;
  canvas.style.opacity = (1 - G.dim * 0.6).toFixed(3);
  composer.render();
}
requestAnimationFrame(frame);
