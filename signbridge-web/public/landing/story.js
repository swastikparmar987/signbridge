/* SignBridge landing · story director
   Owns scroll, chapters, captions, sound cues and the shared state the 3D scene reads (window.SB). */
(function () {
  const root = document.documentElement;
  const hasGsap = !!(window.gsap && window.ScrollTrigger);
  if (!hasGsap) root.classList.add('reduce');
  const reduce = root.classList.contains('reduce');
  const Sound = window.SBSound;
  const isMobile = () => innerWidth < 900;
  const clamp01 = v => Math.max(0, Math.min(1, v));
  const smooth = (a, b, v) => { const t = clamp01((v - a) / (b - a)); return t * t * (3 - 2 * t) };
  const parse = s => { try { return JSON.parse(s) } catch (e) { return null } };
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

  /* Film timeline, as fractions of the film's scroll length */
  const T = { open: [0.1, 0.5], trace: [0.5, 0.66], sun: [0.54, 0.72], burst: 0.7, dive: [0.72, 0.9], world: 0.9, hero: 0.93 };
  const FINGER_ORDER = ['pinky', 'ring', 'middle', 'index', 'thumb'];

  const SB = window.SB = {
    T, FINGER_ORDER, reduce,
    film: reduce ? 1 : 0, intro: reduce ? 1 : 0, entered: false, ready: false, loadP: 0,
    target: { x: 0, y: 0, s: 1, dim: 0 }, pose: 'HELLO', lanterns: 1,
    beat: -10, burst: 0, pulse: -10, flare: [-10, -10, -10, -10, -10],
    mouse: { x: 0, y: 0 }, vel: 0,
    open(f = SB.film) { return smooth(T.open[0], T.open[1], f) },
    finger(k, f = SB.film) { return clamp01(SB.open(f) * 1.6 - k * 0.15) },
    isMobile,
  };

  /* ---------------- Loading: the hand signs its first words while the counter runs ---------------- */
  const gate = document.getElementById('gate'), actions = document.getElementById('gateActions');
  const bar = document.getElementById('gateBar'), num = document.getElementById('gateNum'), status = document.getElementById('gateStatus');
  const wordEl = document.getElementById('gateWord'), kicker = document.getElementById('gateKicker'), spell = document.getElementById('gateSpell');
  const spellTiles = [...spell.querySelectorAll('i')];
  /* each step: when it starts (0..1 of loading), the hand pose, what we show, and an optional motion */
  const SEQ = [
    { at: 0.00, pose: 'HELLO', word: 'Hello', motion: 'wave' },
    { at: 0.15, pose: 'L', word: 'Love', spell: 0 },
    { at: 0.24, pose: 'O', word: 'Love', spell: 1 },
    { at: 0.33, pose: 'V', word: 'Love', spell: 2 },
    { at: 0.42, pose: 'E', word: 'Love', spell: 3 },
    { at: 0.53, pose: 'LOVE', word: 'I love you' },
    { at: 0.69, pose: 'B', word: 'Thank you', motion: 'thank' },
    { at: 0.84, pose: 'S', word: 'Yes', motion: 'nod' },
  ];
  const STATUS = ['Preparing the story', 'Learning your hands', 'Finding the words', 'Almost there'];
  SB.phase = 'loading'; SB.motion = 'wave'; SB.pose = 'HELLO';
  const LOAD_MS = reduce ? 3200 : 6400;
  const t0 = performance.now();
  let shown = 0, gateOpen = false, step = -1;
  function showStep(i) {
    const st = SEQ[i], prev = SEQ[step];
    step = i; SB.pose = st.pose; SB.motion = st.motion || null; SB.pulse = performance.now();
    if (!prev || prev.word !== st.word) {
      wordEl.textContent = st.word; wordEl.classList.remove('swap'); void wordEl.offsetWidth; wordEl.classList.add('swap');
    }
    kicker.textContent = st.spell != null ? 'Spelling' : 'Signing';
    spell.classList.toggle('on', st.spell != null);
    spellTiles.forEach((t, k) => { t.classList.toggle('now', k === st.spell); t.classList.toggle('done', st.spell != null && k < st.spell) });
  }
  (function loadLoop() {
    if (gateOpen) return;
    const timeP = clamp01((performance.now() - t0) / LOAD_MS);
    const target = Math.min(timeP, SB.ready ? 1 : 0.9);
    shown += (target - shown) * 0.1;
    if (target >= 1 && shown > 0.995) shown = 1;
    bar.style.setProperty('--p', shown.toFixed(4));
    num.textContent = String(Math.floor(shown * 100));
    status.textContent = STATUS[Math.min(STATUS.length - 1, Math.floor(shown * STATUS.length))];
    let i = 0; while (i + 1 < SEQ.length && shown >= SEQ[i + 1].at) i++;
    if (i !== step) showStep(i);
    if (shown >= 1 || performance.now() - t0 > LOAD_MS + 8000) {
      gateOpen = true; num.textContent = '100'; bar.style.setProperty('--p', 1);
      status.textContent = 'Ready';
      SB.motion = null; SB.pose = 'S';
      gate.classList.add('ready');
      setTimeout(() => {
        actions.classList.add('on');
        document.getElementById(Sound.saved() === 'off' ? 'enterQuiet' : 'enterSound').focus({ preventScroll: true });
      }, 500);
      return;
    }
    requestAnimationFrame(loadLoop);
  })();

  function enter(withSound) {
    Sound.set(withSound);
    if (withSound) Sound.whoosh();
    document.body.classList.remove('locked');
    document.body.classList.add('entered');
    SB.entered = true;
    scrollTo(0, 0);
    if (lenis) lenis.start();
    SB.phase = 'film'; SB.motion = null;
    if (hasGsap && !reduce) {
      gsap.to(gate, { opacity: 0, duration: 1.1, ease: 'power2.inOut', onComplete: () => gate.remove() });
      /* the small signing hand grows into the giant fist */
      gsap.to(SB, { intro: 1, duration: 2.8, ease: 'power3.inOut', delay: 0.1 });
      Sound.heartbeat(true);
    } else { gate.remove(); SB.intro = 1 }
  }
  document.getElementById('enterSound').addEventListener('click', () => enter(true));
  document.getElementById('enterQuiet').addEventListener('click', () => enter(false));
  document.querySelectorAll('[data-sound-toggle]').forEach(b => b.addEventListener('click', () => { Sound.set(!Sound.on); if (Sound.on) Sound.chime(5) }));
  addEventListener('keydown', e => { if ((e.key === 'm' || e.key === 'M') && SB.entered && !e.target.closest('input,textarea')) Sound.set(!Sound.on) });

  /* ---------------- Smooth scroll ---------------- */
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
  const scrollToEl = (target, immediate) => {
    if (lenis) lenis.scrollTo(target, { duration: 1.8, immediate });
    else if (typeof target === 'number') scrollTo({ top: target, behavior: reduce || immediate ? 'auto' : 'smooth' });
    else target.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth' });
  };
  document.querySelectorAll('a[href^="#"]').forEach(a => a.addEventListener('click', e => {
    const id = a.getAttribute('href'); const el = document.querySelector(id); if (!el) return;
    e.preventDefault(); scrollToEl(id === '#top' ? 0 : el);
  }));

  /* ---------------- Caption word split ---------------- */
  function splitWords(el, cls = 'w') {
    const words = [];
    const walk = node => [...node.childNodes].forEach(n => {
      if (n.nodeType === 3) {
        const frag = document.createDocumentFragment();
        n.textContent.split(/(\s+)/).forEach(w => {
          if (!w) return;
          if (/^\s+$/.test(w)) { frag.appendChild(document.createTextNode(' ')); return }
          const s = document.createElement('span'); s.className = cls; s.textContent = w; frag.appendChild(s); words.push(s);
        });
        n.replaceWith(frag);
      } else if (n.nodeType === 1 && n.tagName !== 'BR') walk(n);
    });
    walk(el);
    return words;
  }
  const caps = [...document.querySelectorAll('.cap')];
  caps.forEach(c => splitWords(c.querySelector('.cap-text')).forEach((w, i) => w.style.transitionDelay = (i * 0.07) + 's'));
  const CAP_WIN = [[0.012, 0.1], [0.16, 0.29], [0.32, 0.46], [0.54, 0.69]];

  /* ---------------- Hand placement per section ---------------- */
  const hero = document.getElementById('hero');
  let learnPose = 'HELLO', activeKey = null;
  function placeFrom(el, key, poseOverride) {
    const d = parse((isMobile() && el.dataset.handM) || el.dataset.hand); if (!d) return;
    Object.assign(SB.target, { x: d.x, y: d.y, s: d.s, dim: d.dim || 0 });
    const pose = poseOverride || d.pose;
    if (pose && SB.pose !== pose) SB.pose = pose;
    if (activeKey !== key && activeKey !== null && SB.film >= T.world) Sound.whoosh();
    SB.lanterns = key === 'hero' ? 1 : 0.22;
    activeKey = key;
  }
  const MOMENTS_PLACE = { x: 0.5, y: -0.02, s: 1.0, dim: 0 }, MOMENTS_PLACE_M = { x: 0, y: 0.5, s: 0.5, dim: 0 };
  let momentIdx = -1;
  function placeMoments() {
    Object.assign(SB.target, isMobile() ? MOMENTS_PLACE_M : MOMENTS_PLACE);
    const scene = scenes[Math.max(0, momentIdx)];
    if (scene) SB.pose = scene.dataset.pose;
    SB.lanterns = 0.35;
    activeKey = 'moments';
  }

  /* ---------------- Film director ---------------- */
  const filmEl = document.querySelector('.film'), cue = document.getElementById('cue'), flash = document.getElementById('flash');
  const armed = { finger: [true, true, true, true, true], burst: true, dive: true, arrive: true };
  let lastF = SB.film, beats = 0, lastBeat = 0, filmActive = true;

  if (hasGsap && !reduce) {
    ScrollTrigger.create({
      trigger: filmEl, start: 'top top', end: 'bottom bottom',
      onUpdate: s => { SB.film = s.progress },
      onToggle: s => { filmActive = s.isActive },
    });
  }

  function direct(now) {
    const f = SB.film, jump = Math.abs(f - lastF) > 0.15;
    /* captions */
    caps.forEach((c, i) => {
      const [a, b] = CAP_WIN[i], on = f > a && f < b && SB.entered;
      c.classList.toggle('on', on);
      const local = clamp01((f - a) / (b - a));
      c.style.opacity = on ? Math.min(1, (b - f) / 0.02).toFixed(3) : 0;
      c.style.transform = `translate3d(0,${((0.5 - local) * 36).toFixed(1)}px,0)`;
    });
    cue.style.opacity = f < 0.012 && SB.entered ? 1 : 0;

    /* heartbeat while the hand is still closed */
    if (SB.entered && f < T.open[0] && now - lastBeat > 1250) { lastBeat = now; SB.beat = now; Sound.heartbeat(beats++ < 3) }

    /* tension rises as it opens */
    Sound.riser(f < T.burst ? SB.open(f) : 0);

    /* each finger releases with a tick and a flare */
    FINGER_ORDER.forEach((_, k) => {
      const p = SB.finger(k, f);
      if (armed.finger[k] && p > 0.62) { armed.finger[k] = false; if (!jump) { Sound.fingerTick(k); SB.flare[k] = now } }
      if (!armed.finger[k] && p < 0.3) armed.finger[k] = true;
    });
    if (armed.burst && f > T.burst) { armed.burst = false; SB.burst++; if (!jump) Sound.bloom() }
    if (!armed.burst && f < T.burst - 0.04) armed.burst = true;
    if (armed.dive && f > T.dive[0] + 0.02) { armed.dive = false; if (!jump) Sound.dive() }
    if (!armed.dive && f < T.dive[0] - 0.02) armed.dive = true;
    if (armed.arrive && f > T.world) { armed.arrive = false; if (!jump) Sound.arrive() }
    if (!armed.arrive && f < T.world - 0.03) armed.arrive = true;

    /* the flash that carries us through the palm */
    if (!reduce) flash.style.opacity = (smooth(0.8, 0.895, f) * (1 - smooth(0.895, 0.975, f)) * 0.96).toFixed(3);

    /* arrival in the world */
    const inWorld = f >= T.world || !filmActive;
    document.body.classList.toggle('world', SB.entered && inWorld);
    hero.classList.toggle('on', f >= T.hero);
    if (filmActive && !reduce) {
      if (f >= T.world) placeFrom(hero, 'hero');
      else activeKey = 'film';
    }
    lastF = f;
  }
  (function loop(now) { direct(now); requestAnimationFrame(loop) })(performance.now());

  document.getElementById('skipStory').addEventListener('click', () => {
    const end = filmEl.offsetTop + filmEl.offsetHeight - innerHeight;
    scrollToEl(filmEl.offsetTop + (end - filmEl.offsetTop) * 0.965, true);
  });

  /* ---------------- Moments ---------------- */
  const scenes = [...document.querySelectorAll('.scene')], dots = [...document.querySelectorAll('.scene-dots i')];
  let typer = 0;
  function typeLine(el) {
    clearInterval(typer);
    const text = el.dataset.say; let i = 0; el.textContent = '';
    if (reduce) { el.textContent = text; return }
    typer = setInterval(() => { el.textContent = text.slice(0, ++i); if (i % 3 === 0) Sound.tick(i / text.length); if (i >= text.length) clearInterval(typer) }, 42);
  }
  function setMoment(i) {
    if (i === momentIdx) return;
    momentIdx = i;
    scenes.forEach((s, k) => { s.classList.toggle('on', k === i); s.classList.toggle('past', k < i) });
    SB.pose = scenes[i].dataset.pose; SB.pulse = performance.now();
    typeLine(scenes[i].querySelector('.said-text'));
    Sound.chime(i * 2 + 1);
  }
  if (reduce) scenes.forEach(s => { s.querySelector('.said-text').textContent = s.querySelector('.said-text').dataset.say });

  /* ---------------- Section triggers ---------------- */
  const sections = [...document.querySelectorAll('main > section[data-hand]')];
  if (hasGsap) {
    ScrollTrigger.create({
      trigger: '#moments', start: 'top 55%', end: 'bottom 45%',
      onToggle: s => { if (s.isActive) placeMoments() },
    });
    if (!reduce) {
      ScrollTrigger.create({
        trigger: '#moments', start: 'top top', end: 'bottom bottom',
        onUpdate: s => {
          const p = s.progress * 4;
          setMoment(Math.min(3, Math.floor(p)));
          dots.forEach((d, k) => d.style.setProperty('--p', clamp01(p - k).toFixed(3)));
        },
        onEnter: () => setMoment(0),
      });
    }
    sections.forEach(sec => ScrollTrigger.create({
      trigger: sec, start: 'top 55%', end: 'bottom 45%',
      onToggle: s => { if (s.isActive) placeFrom(sec, sec.id || sec.className, sec.id === 'learn' ? learnPose : null) },
    }));
    ScrollTrigger.create({ start: 0, end: 'max', onUpdate: s => { SB.vel = s.getVelocity(); if (SB.film >= 1) Sound.padBright(0.4 + s.progress * 0.6) } });
  }
  if (reduce) { placeFrom(hero, 'hero'); document.body.classList.add('world'); hero.classList.add('on') }
  addEventListener('resize', () => { if (activeKey === 'moments') placeMoments() });

  /* ---------------- Tagline: words light up in reading order ---------------- */
  const tagWords = splitWords(document.getElementById('tagline'));
  if (hasGsap && !reduce) {
    ScrollTrigger.create({
      trigger: '.tagline', start: 'top 72%', end: 'center 42%',
      onUpdate: s => { const n = Math.round(s.progress * tagWords.length); tagWords.forEach((w, i) => { if (i < n) w.classList.add('lit') }) },
    });
  }

  /* ---------------- Reveals (once) ---------------- */
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target) } }), { threshold: 0.15 });
  document.querySelectorAll('[data-reveal]').forEach(el => io.observe(el));

  /* ---------------- Learn ---------------- */
  const chips = [...document.querySelectorAll('.chip')], now = document.getElementById('nowSigning');
  chips.forEach((c, i) => c.addEventListener('click', () => {
    chips.forEach(o => { o.classList.toggle('on', o === c); o.setAttribute('aria-pressed', String(o === c)) });
    learnPose = c.dataset.pose; SB.pose = learnPose; SB.pulse = performance.now();
    now.textContent = c.dataset.name;
    Sound.chime(i);
  }));
  const counter = document.querySelector('.count');
  if (counter && hasGsap && !reduce) {
    const to = +counter.dataset.to, o = { v: 0 }; let lt = 0;
    counter.textContent = '0';
    new IntersectionObserver((es, obs) => es.forEach(e => {
      if (!e.isIntersecting) return; obs.disconnect();
      gsap.to(o, { v: to, duration: 2.4, ease: 'expo.out', onUpdate() { counter.textContent = Math.round(o.v).toLocaleString('en-US'); const n = performance.now(); if (n - lt > 60) { lt = n; Sound.tick(o.v / to) } } });
    }), { threshold: 0.6 }).observe(counter);
  }

  /* ---------------- Nav: current section ---------------- */
  const links = [...document.querySelectorAll('.nav-links a')];
  const navIo = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    links.forEach(a => a.setAttribute('aria-current', String(a.getAttribute('href') === '#' + e.target.id)));
  }), { rootMargin: '-45% 0px -45% 0px' });
  ['moments', 'how', 'learn', 'faq'].forEach(id => { const el = document.getElementById(id); if (el) navIo.observe(el) });

  /* ---------------- Pointer + interface sounds ---------------- */
  addEventListener('pointermove', e => { SB.mouse.x = e.clientX / innerWidth * 2 - 1; SB.mouse.y = -(e.clientY / innerHeight * 2 - 1) }, { passive: true });
  document.querySelectorAll('[data-sfx]').forEach((el, i) => {
    el.addEventListener('pointerenter', () => Sound.hover(i));
    if (!['enterSound', 'enterQuiet', 'soundToggle'].includes(el.id)) el.addEventListener('click', () => Sound.click());
  });
  document.querySelectorAll('.qa summary').forEach((s, i) => s.addEventListener('click', () => Sound.chime(i + 2)));
})();
