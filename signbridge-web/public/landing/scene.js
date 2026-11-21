/* SignBridge landing · 3D film
   A giant velvet hand (the studio's rigged GLB) holds a light. Scroll opens it, a constellation traces the
   hand as it is understood, the light blooms, and the camera dives through the palm into the world. */
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

function fail() {
  root.classList.add('nogl');
  document.getElementById('fallback').hidden = false;
  SB.ready = true;
}

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
} catch (e) { fail(); throw e }

renderer.setPixelRatio(Math.min(devicePixelRatio, mobile ? 1.25 : 1.75));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;

const scene = new THREE.Scene();
const BG = new THREE.Color(0x100D19);
scene.background = BG;
scene.fog = new THREE.FogExp2(0x100D19, 0.035);
const camera = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.02, 100);
camera.position.set(0, 0, 3.2);

const ORANGE = new THREE.Color(0xFB6A18), AMBER = new THREE.Color(0xFFB27E), CREAM = new THREE.Color(0xFAF6F0), VIOLET = new THREE.Color(0x9D8CFF);

/* ---------- Light: the hand is drawn by its edges ---------- */
scene.add(new THREE.AmbientLight(0x2a2238, 0.6));
const rimL = new THREE.DirectionalLight(0xFFB27E, 3.2); rimL.position.set(-3, 2.5, -2.5); scene.add(rimL);
const rimR = new THREE.DirectionalLight(0x9D8CFF, 2.0); rimR.position.set(3.2, 0.5, -3); scene.add(rimR);
const fill = new THREE.DirectionalLight(0xFAF6F0, 0.1); fill.position.set(0.5, 1, 5); scene.add(fill);
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

/* ---------- God rays behind the hand (the hand occludes them) ---------- */
const rays = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.ShaderMaterial({
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
      vec3 col = uColor * ((s1 + s2) * 0.7 + 0.25) * fall + vec3(1.0, 0.8, 0.6) * core * 0.8;
      gl_FragColor = vec4(col * uI, 1.0);
    }`,
}));
scene.add(rays);

/* ---------- Palm light ---------- */
const sun = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, ...additive, opacity: 0 })); scene.add(sun);
const core = new THREE.Sprite(new THREE.SpriteMaterial({ map: coreTex, ...additive, opacity: 0 })); scene.add(core);

/* ---------- Shockwave rings ---------- */
const ringMat = () => new THREE.MeshBasicMaterial({ color: ORANGE.clone().multiplyScalar(2.2), ...additive, opacity: 0, side: THREE.DoubleSide, depthTest: false });
const rings = [new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 160), ringMat()), new THREE.Mesh(new THREE.RingGeometry(0.99, 1, 160), ringMat())];
rings.forEach(r => { r.renderOrder = 12; scene.add(r) });

/* ---------- Dust: motes drifting through the light ---------- */
const DUST = mobile ? 900 : 2200;
const dGeo = new THREE.BufferGeometry();
const dPos = new Float32Array(DUST * 3), dSeed = new Float32Array(DUST);
for (let i = 0; i < DUST; i++) {
  dPos[i * 3] = (Math.random() - 0.5) * 14; dPos[i * 3 + 1] = (Math.random() - 0.5) * 9; dPos[i * 3 + 2] = -7 + Math.random() * 9;
  dSeed[i] = Math.random();
}
dGeo.setAttribute('position', new THREE.BufferAttribute(dPos, 3));
dGeo.setAttribute('aSeed', new THREE.BufferAttribute(dSeed, 1));
const dust = new THREE.Points(dGeo, new THREE.ShaderMaterial({
  ...additive,
  uniforms: { uTime: { value: 0 }, uBurst: { value: 0 }, uCenter: { value: new THREE.Vector3() }, uSize: { value: 28 * renderer.getPixelRatio() }, uO: { value: 0 }, uWarm: { value: 0 } },
  vertexShader: `
    attribute float aSeed; uniform float uTime, uBurst, uSize; uniform vec3 uCenter; varying float vA; varying float vS;
    void main(){
      vec3 p = position;
      p.x += sin(uTime * 0.11 + aSeed * 40.0) * 0.35;
      p.y += mod(uTime * (0.02 + aSeed * 0.05) + aSeed * 9.0, 9.0) - 4.5 - position.y * 0.0;
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
    uniform float uO, uWarm; varying float vA; varying float vS;
    void main(){
      float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;
      float a = smoothstep(0.5, 0.0, d);
      vec3 warm = mix(vec3(1.0, 0.55, 0.22), vec3(1.0, 0.86, 0.7), fract(vS * 3.7));
      vec3 cool = mix(vec3(0.62, 0.55, 1.0), vec3(0.98, 0.96, 0.94), fract(vS * 5.3));
      vec3 c = mix(cool, warm, clamp(uWarm + step(0.82, vS), 0.0, 1.0));
      gl_FragColor = vec4(c * a * vA * uO, 1.0);
    }`,
}));
dust.frustumCulled = false;
scene.add(dust);

/* ---------- Lantern words: what people finally hear ---------- */
const WORDS = ['hello', 'thank you', 'I love you', 'yes', 'me too', 'good morning', 'see you soon', 'one oat latte, please', 'I understand', 'welcome', 'friend', 'family', 'nice to meet you', 'proud of you', 'how are you?', 'I am here'];
function wordTex(text) {
  const c = document.createElement('canvas'), g = c.getContext('2d');
  const font = '600 64px Onest, "Plus Jakarta Sans", system-ui, sans-serif';
  g.font = font; const w = Math.ceil(g.measureText(text).width) + 120;
  c.width = w; c.height = 160;
  g.font = font; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.shadowColor = 'rgba(251,106,24,0.9)'; g.shadowBlur = 36; g.fillStyle = 'rgba(255,214,180,1)';
  g.fillText(text, w / 2, 84);
  g.shadowBlur = 0; g.fillStyle = 'rgba(255,250,244,1)'; g.fillText(text, w / 2, 84);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return { t, aspect: w / 160 };
}
const lanterns = [];
const LCOUNT = mobile ? 12 : 20;
for (let i = 0; i < LCOUNT; i++) {
  const { t, aspect } = wordTex(WORDS[i % WORDS.length]);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, ...additive, opacity: 0, fog: true }));
  const h = 0.22 + Math.random() * 0.2;
  s.scale.set(h * aspect, h, 1);
  s.userData = { x: (Math.random() - 0.5) * 16, y: (Math.random() - 0.5) * 10, z: -5 - Math.random() * 8, sp: 0.08 + Math.random() * 0.12, ph: Math.random() * 6.28 };
  scene.add(s); lanterns.push(s);
}

/* ---------- Constellation: the hand, understood ---------- */
const overlay = new THREE.Group(); scene.add(overlay);
const TIP_OF = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const TIPS = Object.values(TIP_OF);
const jGeo = new THREE.IcosahedronGeometry(1, 2);
const joints = JOINTS.map((_, i) => {
  const m = new THREE.Mesh(jGeo, new THREE.MeshBasicMaterial({ color: (TIPS.includes(i) ? AMBER : ORANGE).clone().multiplyScalar(2.4), ...additive, opacity: 0, depthTest: false }));
  m.renderOrder = 11; m.userData.r = TIPS.includes(i) ? 0.022 : i === 0 ? 0.024 : 0.016; overlay.add(m); return m;
});
const bGeo = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true);
const bMat = new THREE.MeshBasicMaterial({ color: ORANGE.clone().multiplyScalar(1.7), ...additive, opacity: 0, depthTest: false });
const links = LINKS.map(() => { const m = new THREE.Mesh(bGeo, bMat.clone()); m.renderOrder = 10; overlay.add(m); return m });
const flares = FINGERS.map(() => { const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, ...additive, opacity: 0, depthTest: false })); s.renderOrder = 13; scene.add(s); return s });

/* ---------- Post: bloom + a cinematic lens ---------- */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
/* shallow depth of field for the opener's macro shots */
const bokeh = new BokehPass(scene, camera, { focus: 1, aperture: 0, maxblur: 0.014 });
bokeh.enabled = false;
composer.addPass(bokeh);
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.9, 0.55, 0.62);
composer.addPass(bloom);
const lens = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uCA: { value: 0.002 }, uRadial: { value: 0 }, uVig: { value: 0.9 }, uCenter: { value: new THREE.Vector2(0.5, 0.5) } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uTime, uCA, uRadial, uVig; uniform vec2 uCenter; varying vec2 vUv;
    void main(){
      vec2 d = vUv - uCenter; vec3 col = vec3(0.0);
      if (uRadial > 0.002) {
        for (int i = 0; i < 14; i++) {
          float s = 1.0 - uRadial * float(i) / 14.0; vec2 p = uCenter + d * s;
          col += vec3(texture2D(tDiffuse, p + d * uCA).r, texture2D(tDiffuse, p).g, texture2D(tDiffuse, p - d * uCA).b);
        }
        col /= 14.0;
      } else {
        col = vec3(texture2D(tDiffuse, vUv + d * uCA).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - d * uCA).b);
      }
      vec2 q = vUv - 0.5;
      col *= mix(1.0, smoothstep(0.95, 0.25, length(q * vec2(1.0, 0.85))), uVig);
      gl_FragColor = vec4(col, 1.0);
    }`,
});
composer.addPass(lens);
composer.addPass(new OutputPass());

function resize() {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight);
}
addEventListener('resize', resize);

/* ---------- Hand ---------- */
const handRoot = new THREE.Group(); scene.add(handRoot);
let rig = null;
SB.loadP = 0.4;
loadHand().then(r => {
  rig = r; handRoot.add(rig.root);
  // centre the open hand on its own palm so every framing is predictable
  applyPose(rig, POSES.OPEN, 1); rig.root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(rig.root), c = new THREE.Vector3(); box.getCenter(c);
  rig.root.position.set(-c.x, -c.y, -c.z);
  applyPose(rig, POSES.S, 1);
  SB.loadP = 1; SB.ready = true;
}).catch(() => { fail(); canvas.style.opacity = 0 });

/* ---------- Loop ---------- */
const V = THREE.Vector3;
const damp = (a, b, l, dt) => a + (b - a) * (1 - Math.exp(-l * dt));
const clamp01 = v => Math.max(0, Math.min(1, v));
const smooth = (a, b, v) => { const t = clamp01((v - a) / (b - a)); return t * t * (3 - 2 * t) };
const easeOut = t => 1 - Math.pow(1 - t, 3), easeIn = t => t * t * t;

const pose = { rotation: [0, 0, 0] };
blendPose(POSES.S, POSES.S, { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, rot: 0 }, pose);
const amounts = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, rot: 0 };
const all = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, rot: 0 };
const pts = JOINTS.map(() => new V());
const focusT = new V(), dirV = new V(), macroCam = new V(), sideA = new V(0.5, 0.25, 0.1), sideB = new V(-0.45, 0.3, 0.15);
const centroid = new V(), palmC = new V(), palmN = new V(), tmp = new V(), tmp2 = new V(), camTarget = new V(), look = new V(), up = new V(0, 1, 0);
const st = { x: 0, y: 0, s: 1, dim: 0, ov: 0, world: SB.film >= SB.T.world, glow: 0 };
let burstAt = -10, lastBurst = SB.burst, t = 0, last = performance.now(), hidden = false;
document.addEventListener('visibilitychange', () => { hidden = document.hidden; last = performance.now() });

function frame(now) {
  requestAnimationFrame(frame);
  if (hidden) return;
  const dt = Math.min(0.05, (now - last) / 1000); last = now; t += dt;
  const T = SB.T, f = SB.film, intro = SB.intro, opening = SB.phase !== 'film', shot = SB.phase === 'choice' ? 0 : SB.shot, shotP = SB.shotP || 0,
    pres = (st.pres = damp(st.pres || 0, rig && !(opening && shot === 0) ? 1 : 0, 1.6, dt)), m = SB.reduce ? { x: 0, y: 0 } : SB.mouse;
  const open = SB.open(f), trace = smooth(T.trace[0], T.trace[1], f), sunP = smooth(T.sun[0], T.sun[1], f);
  const dive = easeIn(smooth(T.dive[0], T.dive[1], f));
  const world = f >= T.world;
  if (SB.burst !== lastBurst) { lastBurst = SB.burst; burstAt = t }
  const bAge = t - burstAt, burstK = bAge < 3 ? Math.exp(-bAge * 1.6) : 0;
  const beatAge = (now - SB.beat) / 1000, beat = beatAge < 0.8 ? Math.exp(-beatAge * 6) + (beatAge > 0.24 ? Math.exp(-(beatAge - 0.24) * 7) * 0.6 : 0) : 0;
  const pulseAge = (now - SB.pulse) / 1000, pulse = pulseAge < 1.2 ? Math.exp(-pulseAge * 3) : 0;
  const wide = camera.aspect < 0.8 ? 1.6 : camera.aspect < 1.2 ? 0.7 : 0;

  /* ----- pose ----- */
  if (!world && !opening) {
    SB.FINGER_ORDER.forEach((name, k) => { amounts[name] = damp(amounts[name], easeOut(SB.finger(k, f)), 9, dt) });
    amounts.rot = open;
    blendPose(POSES.S, POSES.OPEN, amounts, pose);
  } else {
    const k = 1 - Math.exp(-dt * 7);
    for (const key in all) all[key] = k;
    blendPose(pose, POSES[SB.pose] || POSES.OPEN, all, pose);
  }

  /* ----- placement ----- */
  if (world !== st.world) {
    st.world = world;
    if (world) { // teleport behind the flash
      const halfH = Math.tan(THREE.MathUtils.degToRad(17.5)) * 7, halfW = halfH * camera.aspect;
      st.x = SB.target.x * halfW; st.y = SB.target.y * halfH; st.s = SB.target.s;
    }
  }
  if (!world) {
    const base = 3.9 + wide + open * 0.6;
    st.cx = damp(st.cx ?? centroid.x, centroid.x, 3, dt); st.cy = damp(st.cy ?? centroid.y, centroid.y, 3, dt);
    const ly = st.cy - 0.05;
    camTarget.set(st.cx + m.x * 0.08, ly + m.y * 0.05, base);
    handRoot.position.set(0, -0.08, 0);
    handRoot.scale.setScalar(1 + Math.sin(t * 1.1) * 0.006 + beat * 0.012);
    handRoot.rotation.set(-m.y * 0.06 - sunP * 0.12, 0.6 * (1 - open) + Math.sin(t * 0.3) * 0.04 + m.x * 0.1, 0.14 * (1 - open) + Math.sin(t * 0.23) * 0.015);
    st.dim = damp(st.dim, 0, 4, dt);
    look.set(st.cx, ly, 0);
    /* title layout: the hand above the title card; finishOpener() grows it into the giant fist */
    const L = 1 - intro;
    if (L > 0.0001) {
      const nod = 0, thank = 0, wave = 0;
      const lp = tmp.set(0, (mobile ? 0.72 : 0.8) + Math.sin(t * 1.2) * 0.03, 0);
      handRoot.position.lerp(lp, L);
      handRoot.scale.setScalar(THREE.MathUtils.lerp(handRoot.scale.x, (mobile ? 0.5 : 0.6) * (1 + pulse * 0.05), L));
      handRoot.rotation.set(
        THREE.MathUtils.lerp(handRoot.rotation.x, -m.y * 0.1 + nod - thank * 0.6, L),
        THREE.MathUtils.lerp(handRoot.rotation.y, Math.sin(t * 0.4) * 0.15 + m.x * 0.2, L),
        THREE.MathUtils.lerp(handRoot.rotation.z, wave, L));
      camTarget.lerp(tmp2.set(m.x * 0.1, 0.1 + m.y * 0.06, 6.2 + wide * 1.5), L);
      look.lerp(tmp2.set(0, 0.1, 0), L);
    }
    /* opener: macro shots across the glowing hand, then a fast pull back to the title framing */
    if (opening && shot < 4) {
      const ez = shotP * shotP * (3 - 2 * shotP);
      const A = shot <= 1, p = shot === 0 ? 0 : shot >= 3 ? 1 : ez;
      const a = A ? pts[5] : pts[12], b = A ? pts[8] : pts[9];
      focusT.copy(a).lerp(b, p);
      dirV.copy(palmN).add(A ? sideA : sideB).normalize();
      const dist = A ? 0.68 - 0.08 * p : 0.85 - 0.15 * p;
      macroCam.copy(focusT).addScaledVector(dirV, dist);
      const w = shot === 3 ? (shotP < 0.5 ? 4 * shotP ** 3 : 1 - Math.pow(-2 * shotP + 2, 3) / 2) : 0;
      tmp2.copy(camTarget); camTarget.copy(macroCam).lerp(tmp2, w);
      tmp2.copy(look); look.copy(focusT).lerp(tmp2, w);
      bokeh.enabled = w < 0.98;
      bokeh.uniforms.focus.value = THREE.MathUtils.lerp(dist, camTarget.distanceTo(look), w);
      bokeh.uniforms.aperture.value = 0.02 * (1 - w);
    } else bokeh.enabled = false;
  } else {
    const halfH = Math.tan(THREE.MathUtils.degToRad(17.5)) * 7, halfW = halfH * camera.aspect, tg = SB.target;
    st.x = damp(st.x, tg.x * halfW, 2.6, dt); st.y = damp(st.y, tg.y * halfH, 2.6, dt);
    st.s = damp(st.s, tg.s, 2.4, dt); st.dim = damp(st.dim, tg.dim || 0, 3, dt);
    handRoot.position.set(st.x, st.y + Math.sin(t * 0.9) * 0.05, 0);
    handRoot.scale.setScalar(st.s * (1 + pulse * 0.04));
    const spin = Math.max(-1, Math.min(1, (SB.vel || 0) / 3000));
    handRoot.rotation.set(-m.y * 0.18 + Math.sin(t * 0.5) * 0.04, Math.sin(t * 0.35) * 0.2 + m.x * 0.35 + spin * 0.4, Math.sin(t * 0.27) * 0.04 - spin * 0.08);
    camTarget.set(m.x * 0.25, m.y * 0.15, 7);
    look.set(0, 0, 0);
  }
  if (rig) { applyPose(rig, pose, 1); rig.root.updateMatrixWorld(true) }

  /* ----- joints in world space ----- */
  if (rig) {
    JOINTS.forEach((name, i) => rig.bones.get(name).bone.getWorldPosition(pts[i]));
    centroid.set(0, 0, 0); pts.forEach(p => centroid.add(p)); centroid.multiplyScalar(1 / pts.length);
    palmC.copy(pts[0]).lerp(pts[9], 0.62);
    tmp.subVectors(pts[5], pts[0]); tmp2.subVectors(pts[17], pts[0]);
    palmN.crossVectors(tmp, tmp2).normalize();
    if (palmN.dot(tmp.subVectors(camera.position, palmC)) < 0) palmN.negate();
  } else { palmC.set(0, 0, 0); palmN.set(0, 0, 1) }

  /* ----- camera (and the dive into the palm) ----- */
  if (!world && dive > 0) {
    tmp.copy(palmC).addScaledVector(palmN, 0.08);
    camTarget.lerp(tmp, dive);
    look.lerp(palmC, Math.min(1, dive * 3));
  }
  if ((!world && dive > 0) || opening) camera.position.copy(camTarget);
  else { camera.position.x = damp(camera.position.x, camTarget.x, 4, dt); camera.position.y = damp(camera.position.y, camTarget.y, 4, dt); camera.position.z = world ? damp(camera.position.z, camTarget.z, 3, dt) : camTarget.z }
  if (world && Math.abs(camera.position.z - 7) > 2) camera.position.copy(camTarget);
  camera.fov = 35 + (world ? 0 : dive * dive * 45);
  camera.updateProjectionMatrix();
  camera.lookAt(look);

  /* ----- the light in the palm ----- */
  const glowTarget = world ? 0.32 + pulse * 0.6 : 0.16 + open * 0.55 + sunP * 1.1 + beat * 0.12;
  st.glow = damp(st.glow, glowTarget * pres * (0.45 + 0.55 * intro), 5, dt);
  const g = st.glow + burstK * 1.4;
  sun.position.copy(palmC).addScaledVector(palmN, 0.22 * (world ? st.s : 1));
  sun.scale.setScalar((world ? st.s : 1) * (0.35 + g * 1.3));
  sun.material.opacity = Math.min(1, g * 0.85);
  core.position.copy(palmC).addScaledVector(palmN, 0.24 * (world ? st.s : 1));
  core.scale.setScalar((world ? st.s : 1) * (0.06 + sunP * 0.32 + burstK * 0.5) * (world ? 0.6 : 1));
  core.material.opacity = world ? 0.35 + pulse * 0.5 : Math.min(1, sunP * 1.2 + burstK);
  palmLight.position.copy(palmC).addScaledVector(palmN, 0.3);
  palmLight.intensity = g * 0.9;

  rays.position.copy(palmC).addScaledVector(palmN, -0.6);
  rays.quaternion.copy(camera.quaternion);
  rays.scale.setScalar(world ? st.s * 0.7 : 1 + sunP * 0.4 + burstK * 0.6);
  rays.material.uniforms.uTime.value = t;
  rays.material.uniforms.uI.value = (world ? 0.22 + pulse * 0.3 : 0.12 + open * 0.45 + sunP * 0.7 + burstK * 0.9) * pres * (0.5 + 0.5 * intro) + (opening ? (shot === 3 ? shotP * 0.55 : shot === 4 ? 0.55 : 0) * pres : 0);

  /* ----- shockwave ----- */
  rings.forEach((r, i) => {
    const a = bAge - i * 0.16;
    r.position.copy(palmC); r.quaternion.copy(camera.quaternion);
    if (a > 0 && a < 1.8) { const e = easeOut(a / 1.8); r.scale.setScalar(0.15 + e * (i ? 3.2 : 4.6)); r.material.opacity = (1 - e) * (i ? 0.6 : 1) }
    else r.material.opacity = 0;
  });

  /* ----- constellation ----- */
  const ovTarget = world ? 0.5 + pulse * 0.5 : opening ? (shot < 3 ? 0.95 : shot === 3 ? 0.95 - shotP * 0.6 : 0.35) : trace;
  st.ov = damp(st.ov, ovTarget * pres, 4, dt);
  const thin = opening && shot < 3 ? 0.7 : opening && shot === 3 ? 0.7 + 0.3 * shotP : 1; // finer lines up close
  const toCam = tmp2;
  joints.forEach((j, i) => {
    toCam.subVectors(camera.position, pts[i]).normalize();
    j.position.copy(pts[i]).addScaledVector(toCam, 0.03);
    const order = i / JOINTS.length, appear = world || opening ? 1 : Math.max(1 - intro, clamp01((trace * 1.25 - order) / 0.2));
    const tip = TIPS.includes(i);
    const twinkle = 1 + Math.sin(t * 3 - i * 0.4) * 0.15 * st.ov;
    j.scale.setScalar(j.userData.r * (0.4 + 0.6 * thin) * (world ? st.s * 0.9 : 1) * easeOut(appear) * twinkle * (1 + (tip ? burstK : 0)));
    j.material.opacity = st.ov * appear;
  });
  links.forEach((l, i) => {
    const [a, b] = LINKS[i];
    const order = i / LINKS.length, appear = world || opening ? 1 : Math.max(1 - intro, clamp01((trace * 1.25 - order) / 0.2));
    tmp.copy(joints[a].position); const end = toCam.copy(joints[a].position).lerp(joints[b].position, easeOut(appear));
    const len = tmp.distanceTo(end);
    l.position.copy(tmp).add(end).multiplyScalar(0.5);
    if (len > 1e-4) l.quaternion.setFromUnitVectors(up, end.sub(tmp).normalize());
    const w = 0.0045 * thin * (world ? st.s : 1);
    l.scale.set(w, Math.max(len, 1e-4), w);
    l.material.opacity = st.ov * 0.85 * Math.min(1, appear * 3);
  });
  SB.FINGER_ORDER.forEach((name, k) => {
    const fl = flares[k], age = (now - SB.flare[k]) / 1000;
    fl.position.copy(joints[TIP_OF[name]].position);
    if (age < 0.9) { const e = Math.exp(-age * 4); fl.scale.setScalar(0.12 + (1 - e) * 0.5); fl.material.opacity = e } else fl.material.opacity = 0;
  });

  /* ----- dust + lanterns ----- */
  const du = dust.material.uniforms;
  du.uTime.value = SB.reduce ? 0 : t; du.uCenter.value.copy(palmC);
  du.uBurst.value = burstK * 1.2;
  du.uO.value = pres * (0.55 + open * 0.3 + burstK * 0.6);
  du.uWarm.value = world ? 0.45 : 0.2 + open * 0.7;
  lanterns.forEach(s => {
    const u = s.userData, y = ((u.y + t * u.sp + 5) % 10 + 10) % 10 - 5;
    s.position.set(u.x + Math.sin(t * 0.2 + u.ph) * 0.4, y, u.z);
    const edge = smooth(-5, -3.6, y) * (1 - smooth(3.6, 5, y));
    tmp.copy(s.position).project(camera);
    const clear = mobile ? 1 : 0.12 + 0.88 * smooth(0.04, 0.32, tmp.x); // keep the copy column clean
    s.material.opacity = damp(s.material.opacity, world ? edge * 0.55 * clear * (SB.lanterns ?? 1) : 0, 2, dt);
  });

  /* ----- hand material ----- */
  if (rig) {
    const u = rig.material.userData.uniforms;
    u.uRim.value = pres * (0.28 + beat * 0.35 + open * 0.25 + (world ? 0.25 : 0)) * (1 - st.dim * 0.4);
    u.uInner.value = (world ? 0.06 + pulse * 0.15 : sunP * 0.22 + burstK * 0.3) * pres;
    rig.root.visible = pres > 0.001;
  }

  /* ----- post ----- */
  bloom.strength = world ? 0.75 : 0.85 + sunP * 0.5 + burstK * 0.9 + dive * 0.8;
  lens.uniforms.uCA.value = 0.0015 + (world ? 0 : dive * 0.03 + burstK * 0.006);
  lens.uniforms.uRadial.value = world ? 0 : dive * 0.42;
  lens.uniforms.uVig.value = world ? 0.55 : 0.95;
  if (!world && dive > 0) { tmp.copy(palmC).project(camera); lens.uniforms.uCenter.value.set(tmp.x * 0.5 + 0.5, tmp.y * 0.5 + 0.5) }
  else lens.uniforms.uCenter.value.set(0.5, 0.5);

  canvas.style.opacity = (1 - st.dim * 0.7).toFixed(3);
  composer.render();
}
requestAnimationFrame(frame);
