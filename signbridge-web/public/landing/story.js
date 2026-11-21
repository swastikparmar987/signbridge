/* SignBridge landing · story director ("The Bridge")
   Owns the opener, scroll, chapter text, sound cues, and the keyframed 3D state the scene reads (window.SB.scene). */
(function () {
  const root = document.documentElement;
  const hasGsap = !!(window.gsap && window.ScrollTrigger);
  if (!hasGsap) root.classList.add('reduce');
  const reduce = root.classList.contains('reduce');
  const Sound = window.SBSound;
  const isMobile = () => innerWidth < 900;
  const clamp01 = v => Math.max(0, Math.min(1, v));
  const smooth = t => { t = clamp01(t); return t * t * (3 - 2 * t) };
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

  const SB = window.SB = {
    reduce, isMobile, phase: 'choice', shot: 0, shotP: 0, pose: 'HELLO', intro: 0, entered: false, ready: false,
    beat: -10, burst: 0, pulse: -10, flare: [-10, -10, -10, -10, -10], mouse: { x: 0, y: 0 }, vel: 0,
    scene: null, events: {},
  };

  /* ================= 1 · The keyframed film =================
     Each chapter lists keyframes at local progress p (0..1 of its runway). Missing fields carry forward,
     so a keyframe only says what changes. Mobile patches merge by index. */
  const BASE = {
    L: { x: 0, y: 0, s: 1, rx: 0, ry: 0, rz: 0, pose: 'S', vis: 1 },
    R: { x: 1.5, y: -1.3, s: 1, rx: 0, ry: 0, rz: 0, pose: 'OPEN', vis: 0 },
    bridge: 0, brk: 0, ov: 0, glow: 0.2, rays: 0.3, tint: '#FF8A3D', ring: 0, dim: 0,
  };
  const FILM = {
    story: [
      [0, { L: { x: -0.2, y: 0.5, s: 3.1, ry: 0.55, rz: 0.12, pose: 'S' }, glow: 0.25, rays: 0.3 }],
      [0.08, {}],
      [0.34, { L: { x: 0, y: 0.02, s: 2.5, ry: 0, rz: 0, pose: 'OPEN' }, glow: 1, rays: 0.62 }],
      [0.56, { L: { x: -0.4, y: 0.3, s: 1.15, ry: -0.35, pose: 'HELLO' }, R: { x: 0.46, y: 0.28, s: 1.05, ry: -0.35, vis: 1, pose: 'HELLO' }, bridge: 1, glow: 0.55, rays: 0.32, ov: 0.3 }],
      [1, { L: { x: -0.44 }, R: { x: 0.47 } }],
    ],
    silence: [
      [0, {}],
      [0.4, { brk: 1, bridge: 1, L: { x: 0.02, y: 0.22, ry: -0.2, pose: 'B' }, R: { x: 0.62, y: 0.12, ry: 1.4, vis: 0.35, pose: 'OPEN' }, tint: '#7A86FF', rays: 0.1, glow: 0.16, ov: 0 }],
      [1, {}],
    ],
    how: [
      [0, { brk: 0, bridge: 0.9, L: { x: -0.22, y: 0.2, s: 1.4, ry: -0.3, pose: 'HELLO' }, R: { x: 0.66, y: -0.18, s: 0.85, ry: -0.35, vis: 1, pose: 'HELLO' }, ov: 1, tint: '#FF8A3D', glow: 0.4, rays: 0.22 }],
      [0.3, {}], [0.37, { L: { pose: 'LOVE' }, R: { pose: 'LOVE' } }], [0.63, {}], [0.7, { L: { pose: 'B' }, R: { pose: 'OPEN' } }], [1, {}],
    ],
    moments: [
      [0, { L: { x: -0.4, y: 0.24, s: 1.1, ry: -0.3, pose: 'C' }, R: { x: 0.42, y: 0.22, s: 1, ry: -0.35, vis: 1, pose: 'OPEN' }, ov: 0.35, bridge: 0.85, tint: '#FF9A4D', rays: 0.24 }],
      [0.2, {}], [0.27, { L: { pose: 'LOVE' }, R: { pose: 'LOVE' }, tint: '#FF6F86' }], [0.45, {}], [0.52, { L: { pose: 'HELLO' }, R: { pose: 'HELLO' }, tint: '#8FB4FF' }],
      [0.7, {}], [0.77, { L: { pose: 'B' }, R: { pose: 'B' }, tint: '#6FE3C8' }], [1, {}],
    ],
    learn: [
      [0, { L: { x: 0.42, y: 0.02, s: 1.45, ry: 0.15, rz: 0, pose: 'LEARN' }, R: { x: 1.8, y: -0.2, vis: 0 }, bridge: 0, ov: 0.6, tint: '#FF8A3D', glow: 0.4, rays: 0.26 }],
      [1, {}],
    ],
    promise: [[0, { L: { x: 0, y: 0.05, s: 1, ry: 0, pose: 'OPEN' }, dim: 0.75, ov: 0.25, rays: 0.2, glow: 0.3 }], [1, {}]],
    faq: [[0, { L: { x: -0.62, y: -0.05, s: 0.9, pose: 'OPEN' }, dim: 0.8 }], [1, {}]],
    finale: [
      [0, { L: { x: -0.62, y: 0.42, s: 0.95, ry: -0.6, pose: 'TOUCH' }, R: { x: 0.62, y: 0.42, s: 0.95, ry: -0.6, vis: 1, pose: 'TOUCH' }, bridge: 0.8, dim: 0, ov: 0.3, glow: 0.4, rays: 0.2 }],
      [0.58, { L: { x: -0.2, pose: 'LOVE' }, R: { x: 0.2, pose: 'LOVE' }, bridge: 0, ring: 1, glow: 0.55, rays: 0.38 }],
      [1, {}],
    ],
  };
  const FILM_M = {
    story: [{ L: { x: -0.3, s: 1.5, y: 0.3 } }, {}, { L: { s: 1.15, y: 0.1 } }, { L: { x: -0.06, y: 0.6, s: 0.6, ry: -0.2 }, R: { x: 0.08, y: 0.16, s: 0.56, ry: 0.2 } }, { L: { x: -0.06 }, R: { x: 0.08 } }],
    silence: [{}, { L: { x: -0.08, y: 0.6 }, R: { x: 0.14, y: 0.18 } }],
    how: [{ L: { x: 0, y: 0.6, s: 0.7 }, R: { x: 0.55, y: 0.58, s: 0.42 } }],
    moments: [{ L: { x: -0.28, y: 0.5, s: 0.6 }, R: { x: 0.3, y: 0.5, s: 0.55 } }],
    learn: [{ L: { x: 0, y: 0.52, s: 0.72 } }],
    promise: [{ L: { y: 0.1, s: 0.7 } }],
    faq: [{ L: { x: 0, y: 0, s: 0.7 } }],
    finale: [{ L: { x: -0.5, y: 0.45, s: 0.6 }, R: { x: 0.5, y: 0.45, s: 0.6 } }, { L: { x: -0.14 }, R: { x: 0.14 } }],
  };

  const chapters = [...document.querySelectorAll('main > section')];
  const merge = (a, b) => { const o = { ...a }; for (const k in b) o[k] = (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k])) ? { ...(a[k] || {}), ...b[k] } : b[k]; return o };
  const hex = h => { const n = parseInt(h.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255] };
  let TL = []; // [{ y, state }]
  function build() {
    const vh = innerHeight, mob = isMobile();
    let state = merge(BASE, {});
    TL = [];
    chapters.forEach(el => {
      const key = el.id === 'story' ? 'story' : el.id || el.className.split(' ')[1];
      const kf = FILM[key]; if (!kf) return;
      const top = el.getBoundingClientRect().top + scrollY, h = el.offsetHeight;
      const sticky = !!el.querySelector(':scope > .stage');
      kf.forEach(([p, patch], i) => {
        state = merge(state, patch);
        const m = mob && FILM_M[key] && FILM_M[key][i];
        const s = m ? merge(state, m) : state;
        if (m) state = merge(state, Object.fromEntries(Object.entries(m).filter(([k]) => k === 'L' || k === 'R')));
        const y = sticky ? top + p * Math.max(1, h - vh) : top - vh * 0.5 + p * h;
        TL.push({ y, state: JSON.parse(JSON.stringify(s)) });
      });
      el._run = { top, h, sticky };
    });
    TL.sort((a, b) => a.y - b.y);
  }
  const lerp = (a, b, t) => a + (b - a) * t;
  let learnPose = 'HELLO';
  const hand = (A, B, t) => ({
    x: lerp(A.x, B.x, t), y: lerp(A.y, B.y, t), s: lerp(A.s, B.s, t), rx: lerp(A.rx, B.rx, t), ry: lerp(A.ry, B.ry, t), rz: lerp(A.rz, B.rz, t), vis: lerp(A.vis, B.vis, t),
    from: A.pose === 'LEARN' ? learnPose : A.pose, to: B.pose === 'LEARN' ? learnPose : B.pose, t,
  });
  function sample(y) {
    if (!TL.length) return null;
    let i = 0; while (i < TL.length - 1 && TL[i + 1].y <= y) i++;
    const last = i === TL.length - 1 || y <= TL[0].y;
    const A = TL[i].state, B = last ? A : TL[i + 1].state;
    const t = last ? 0 : smooth((y - TL[i].y) / Math.max(1, TL[i + 1].y - TL[i].y));
    const out = { L: hand(A.L, B.L, t), R: hand(A.R, B.R, t) };
    for (const k of ['bridge', 'brk', 'ov', 'glow', 'rays', 'ring', 'dim']) out[k] = lerp(A[k], B[k], t);
    const ca = hex(A.tint), cb = hex(B.tint); out.tint = ca.map((v, j) => lerp(v, cb[j], t));
    return out;
  }

  /* ================= 2 · Opener: "Before the first word" ================= */
  const gate = document.getElementById('gate'), note = document.getElementById('gateNote');
  const letterbox = document.getElementById('letterbox'), flare = document.getElementById('flare');
  const titleCard = document.getElementById('titleCard'), skipBtn = document.getElementById('skipIntro');
  const SHOTS = [1.6, 2.2, 1.4, 1.8, 2.4];
  (function waitReady() {
    if (SB.ready) { note.textContent = 'Headphones recommended'; return }
    note.textContent = 'Preparing the story';
    setTimeout(waitReady, 200);
  })();
  document.getElementById(Sound.saved() === 'off' ? 'enterQuiet' : 'enterSound').focus({ preventScroll: true });
  /* the lockup leans toward the pointer; the silence button carries a cursor spotlight */
  const lockup = document.getElementById('lockup');
  if (!reduce && matchMedia('(hover:hover)').matches) {
    gate.addEventListener('pointermove', e => {
      const x = e.clientX / innerWidth - 0.5, y = e.clientY / innerHeight - 0.5;
      lockup.style.transform = `perspective(900px) rotateY(${(x * 10).toFixed(2)}deg) rotateX(${(-y * 8).toFixed(2)}deg) translate3d(${(x * 10).toFixed(1)}px,${(y * 8).toFixed(1)}px,0)`;
    }, { passive: true });
    gate.addEventListener('pointerleave', () => { lockup.style.transform = '' });
    const q = document.getElementById('enterQuiet');
    q.addEventListener('pointermove', e => { const r = q.getBoundingClientRect(); q.style.setProperty('--mx', (e.clientX - r.left) + 'px'); q.style.setProperty('--my', (e.clientY - r.top) + 'px') });
  }

  let openerDone = false;
  function enter(withSound) {
    Sound.set(withSound);
    if (withSound) Sound.whoosh();
    scrollTo(0, 0);
    gate.classList.add('go');                       // the logo flares first
    setTimeout(() => gate.classList.add('out'), 380);
    setTimeout(() => gate.remove(), 1500);
    if (reduce || !hasGsap) { finishOpener(); return }
    SB.phase = 'opener';
    document.body.classList.add('opening');
    letterbox.classList.add('on');
    setTimeout(() => flare.classList.add('go'), 350);
    Sound.murmur(true);
    let clock = 0, last = performance.now(), waited = 0, cue = -1;
    (function tick(now) {
      if (openerDone) return;
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      if (clock + dt > SHOTS[0] && !SB.ready && waited < 10) { waited += dt } else clock += dt;
      let i = 0, acc = 0; while (i < SHOTS.length - 1 && clock > acc + SHOTS[i]) { acc += SHOTS[i]; i++ }
      SB.shot = i; SB.shotP = clamp01((clock - acc) / SHOTS[i]);
      SB.pose = i < 3 ? 'HELLO' : 'S';
      if (i !== cue) {
        cue = i;
        if (i === 4) {
          Sound.murmur(false); Sound.riser(0);
          setTimeout(() => Sound.titleHit(), 260);
          titleCard.classList.add('on');
          setTimeout(() => titleCard.classList.add('line'), 1100);
        }
      }
      if (i === 3) Sound.riser(0.2 + SB.shotP * 0.6);
      if (i === 4 && SB.shotP >= 1) { finishOpener(); return }
      requestAnimationFrame(tick);
    })(last);
  }
  function finishOpener() {
    if (openerDone) return;
    openerDone = true;
    Sound.murmur(false); Sound.riser(0);
    SB.phase = 'film'; SB.shot = 4; SB.shotP = 1;
    document.body.classList.remove('locked', 'opening');
    document.body.classList.add('entered');
    letterbox.classList.remove('on');
    titleCard.classList.add('out');
    setTimeout(() => { titleCard.remove(); flare.remove() }, 1200);
    SB.entered = true; SB.intro = 1;
    scrollTo(0, 0);
    build();
    if (lenis) lenis.start();
    setTimeout(() => Sound.heartbeat(true), 900);
  }
  skipBtn.addEventListener('click', finishOpener);
  addEventListener('keydown', e => { if (e.key === 'Escape' && SB.phase === 'opener') finishOpener() });
  document.getElementById('enterSound').addEventListener('click', () => enter(true));
  document.getElementById('enterQuiet').addEventListener('click', () => enter(false));
  document.querySelectorAll('[data-sound-toggle]').forEach(b => b.addEventListener('click', () => { Sound.set(!Sound.on); if (Sound.on) Sound.chime(5) }));
  addEventListener('keydown', e => { if ((e.key === 'm' || e.key === 'M') && SB.entered && !e.target.closest('input,textarea')) Sound.set(!Sound.on) });

  /* ================= 3 · Smooth scroll ================= */
  let lenis = null;
  if (hasGsap) {
    gsap.registerPlugin(ScrollTrigger);
    if (!reduce && window.Lenis) {
      try {
        lenis = new Lenis({ lerp: 0.075, smoothWheel: true, wheelMultiplier: 0.9 });
        lenis.stop();
        lenis.on('scroll', ScrollTrigger.update);
        gsap.ticker.add(t => lenis.raf(t * 1000)); gsap.ticker.lagSmoothing(0);
      } catch (e) { lenis = null }
    }
  }
  document.querySelectorAll('a[href^="#"]').forEach(a => a.addEventListener('click', e => {
    const id = a.getAttribute('href'), el = document.querySelector(id); if (!el) return;
    e.preventDefault();
    const target = el.id === 'story' ? 0 : el.getBoundingClientRect().top + scrollY + (el._run && el._run.sticky ? innerHeight * 0.4 : 0);
    if (lenis) lenis.scrollTo(target, { duration: 1.8 }); else scrollTo({ top: target, behavior: reduce ? 'auto' : 'smooth' });
  }));

  /* ================= 4 · Text: split words for blur-rise ================= */
  function split(el, cls = 'w') {
    const words = [];
    const walk = node => [...node.childNodes].forEach(n => {
      if (n.nodeType === 3) {
        const frag = document.createDocumentFragment();
        n.textContent.split(/(\s+)/).forEach(w => {
          if (!w) return;
          if (/^\s+$/.test(w)) { frag.append(' '); return }
          const s = document.createElement('span'); s.className = cls; s.textContent = w; frag.append(s); words.push(s);
        });
        n.replaceWith(frag);
      } else if (n.nodeType === 1 && n.tagName !== 'BR') walk(n);
    });
    walk(el);
    words.forEach((w, i) => w.style.setProperty('--i', i));
    return words;
  }
  document.querySelectorAll('[data-split], .cap').forEach(el => split(el));
  const promiseWords = split(document.getElementById('promise'));

  /* ================= 5 · Chapter director ================= */
  const wins = chapters.map(ch => [...ch.querySelectorAll('[data-win]')].map(el => { const [a, b] = el.dataset.win.split(',').map(Number); return { el, a, b } }));
  const heard = document.querySelector('.heard'), heardWord = document.getElementById('heardWord');
  const rail = [...document.querySelectorAll('.rail li')], mdots = [...document.querySelectorAll('.m-dots i')];
  const progressBar = document.getElementById('progressBar');
  const WORDS = ['Hello.', 'I love you.', 'Thank you.'];
  const POSE_OF = ['HELLO', 'LOVE', 'B'];
  let typer = 0, speakT = 0;
  function typeWord(text) {
    clearInterval(typer); clearTimeout(speakT);
    let i = 0; heardWord.textContent = ''; heard.classList.add('speaking');
    if (reduce) { heardWord.textContent = text; return }
    typer = setInterval(() => {
      heardWord.textContent = text.slice(0, ++i); Sound.letter();
      if (i >= text.length) { clearInterval(typer); speakT = setTimeout(() => heard.classList.remove('speaking'), 1400) }
    }, 55);
  }
  const mark = { beat: -1, moment: -1, fingers: [0, 0, 0, 0, 0], bridge: false, brk: false, meet: false, count: false };
  let lastBeat = 0, lastY = -1;
  const localP = ch => { const r = ch._run; if (!r) return -1; return r.sticky ? (scrollY - r.top) / Math.max(1, r.h - innerHeight) : (scrollY - r.top + innerHeight) / (r.h + innerHeight) };

  function direct(now) {
    requestAnimationFrame(direct);
    if (!TL.length) build();
    const y = reduce ? scrollY : scrollY;
    SB.scene = sample(y + 0) || SB.scene;
    if (y === lastY && SB.scene && !SB._learnDirty) { tickAmbient(now); return }
    lastY = y; SB._learnDirty = false;

    chapters.forEach((ch, ci) => {
      const p = localP(ch);
      wins[ci].forEach(w => w.el.classList.toggle('on', SB.entered && p >= w.a && p < w.b));
      const id = ch.id || ch.className.split(' ')[1];
      if (!SB.entered) return;
      if (id === 'story') {
        document.body.classList.toggle('world', p > 0.5 || p > 1);
        for (let k = 0; k < 5; k++) {
          const th = 0.1 + k * 0.045, up = p > th;
          if (up && !mark.fingers[k]) { mark.fingers[k] = 1; Sound.fingerTick(k); SB.flare[k] = now }
          if (!up && p < th - 0.03) mark.fingers[k] = 0;
        }
        if (p > 0.46 && p < 1.2 && !mark.bridge) { mark.bridge = true; Sound.bridgeOn(); SB.burst++ }
        if (p < 0.4) mark.bridge = false;
      }
      if (id === 'silence') {
        if (p > 0.15 && p < 1 && !mark.brk) { mark.brk = true; Sound.breakApart() }
        if (p < 0.1) mark.brk = false;
      }
      if (id === 'how' && p > -0.3 && p < 1.3) {
        const b = p < 0.34 ? 0 : p < 0.67 ? 1 : 2;
        if (b !== mark.beat && p >= -0.05) { mark.beat = b; typeWord(WORDS[b]); Sound.chime(b * 2 + 3); SB.pulse = now }
        rail.forEach((li, k) => li.classList.toggle('on', k === b));
      }
      if (id === 'moments' && p > -0.2 && p < 1.2) {
        const i = Math.max(0, Math.min(3, Math.floor(p * 4)));
        if (i !== mark.moment && p >= 0) { mark.moment = i; Sound.sceneChord(i); SB.pulse = now }
        mdots.forEach((d, k) => d.style.setProperty('--p', clamp01(p * 4 - k).toFixed(3)));
      }
      if (id === 'learn' && p > 0.05 && !mark.count) { mark.count = true; countUp() }
      if (ch.classList.contains('promise')) {
        const n = Math.round(clamp01((p - 0.1) / 0.55) * promiseWords.length);
        promiseWords.forEach((w, i) => w.classList.toggle('lit', reduce || i < n));
      }
      if (id === 'finale') {
        if (p > 0.5 && !mark.meet) { mark.meet = true; Sound.meet(); SB.burst++ }
        if (p < 0.4) mark.meet = false;
      }
    });
    if (!chapters.some(ch => ch.id === 'story' && localP(ch) <= 1.05) && SB.entered) document.body.classList.add('world');
    const max = document.documentElement.scrollHeight - innerHeight;
    progressBar.style.setProperty('--p', (scrollY / Math.max(1, max)).toFixed(4));
    document.body.classList.toggle('end', scrollY > max - innerHeight * 0.6);
    Sound.padBright(0.3 + clamp01(scrollY / Math.max(1, max)) * 0.7);
    tickAmbient(now);
  }
  function tickAmbient(now) {
    /* heartbeat while the fist is still closed */
    const hero = chapters[0], p = localP(hero);
    if (SB.entered && p < 0.07 && now - lastBeat > 1300) { lastBeat = now; SB.beat = now; Sound.heartbeat(false) }
  }
  requestAnimationFrame(direct);
  addEventListener('resize', () => { build(); lastY = -1 });
  addEventListener('load', () => { build(); lastY = -1 });
  if (document.fonts) document.fonts.ready.then(() => { build(); lastY = -1 });

  /* ================= 6 · Learn ================= */
  const chips = [...document.querySelectorAll('.chip-btn')], nowEl = document.getElementById('nowSigning');
  chips.forEach((c, i) => c.addEventListener('click', () => {
    chips.forEach(o => { o.classList.toggle('on', o === c); o.setAttribute('aria-pressed', String(o === c)) });
    learnPose = c.dataset.pose; SB._learnDirty = true; SB.pulse = performance.now();
    nowEl.textContent = c.dataset.name; Sound.chime(i);
  }));
  const countEl = document.getElementById('count');
  function countUp() {
    if (reduce || !hasGsap) return;
    const to = +countEl.dataset.to, o = { v: 0 }; let lt = 0;
    gsap.to(o, { v: to, duration: 2, ease: 'expo.out', onUpdate() { countEl.textContent = Math.round(o.v).toLocaleString('en-US'); const n = performance.now(); if (n - lt > 70) { lt = n; Sound.tick(o.v / to) } } });
  }

  /* ================= 7 · Reveals, nav, pointer, interface sound ================= */
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target) } }), { threshold: 0.2 });
  document.querySelectorAll('[data-reveal]').forEach(el => io.observe(el));
  const links = [...document.querySelectorAll('.nav-links a')];
  const navIo = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    links.forEach(a => a.setAttribute('aria-current', String(a.getAttribute('href') === '#' + e.target.id)));
  }), { rootMargin: '-45% 0px -45% 0px' });
  ['story', 'silence', 'how', 'moments', 'learn', 'faq', 'finale'].forEach(id => { const el = document.getElementById(id); if (el) navIo.observe(el) });
  addEventListener('pointermove', e => { SB.mouse.x = e.clientX / innerWidth * 2 - 1; SB.mouse.y = -(e.clientY / innerHeight * 2 - 1) }, { passive: true });
  document.querySelectorAll('[data-sfx]').forEach((el, i) => {
    el.addEventListener('pointerenter', () => Sound.hover(i));
    if (!['enterSound', 'enterQuiet', 'soundToggle'].includes(el.id)) el.addEventListener('click', () => Sound.click());
  });
  document.querySelectorAll('.qa summary').forEach((s, i) => s.addEventListener('click', () => Sound.chime(i + 2)));
  if (reduce) heardWord.textContent = WORDS[0];
})();
