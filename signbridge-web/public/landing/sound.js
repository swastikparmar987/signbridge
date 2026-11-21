/* SignBridge landing sound design. Fully synthesized with Web Audio, no files.
   Every sound has a visual twin on screen, so the story is complete in silence. */
(function () {
  const store = { get(k) { try { return localStorage.getItem(k) } catch (e) { return null } }, set(k, v) { try { localStorage.setItem(k, v) } catch (e) {} } };
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const SCALE = [220, 246.94, 277.18, 329.63, 369.99, 440, 493.88, 554.37, 659.25, 739.99];

  let ctx = null, master, wet, noise, pad = null, riser = null, enabled = false;

  function impulse(sec, decay) {
    const len = Math.floor(ctx.sampleRate * sec), b = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let c = 0; c < 2; c++) { const d = b.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay) }
    return b;
  }
  function ensure() {
    if (ctx) return true;
    const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return false;
    ctx = new AC();
    const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -14; comp.ratio.value = 4;
    master = ctx.createGain(); master.gain.value = 0; master.connect(comp); comp.connect(ctx.destination);
    const conv = ctx.createConvolver(); conv.buffer = impulse(4, 2.2);
    wet = ctx.createGain(); wet.gain.value = 0.9; wet.connect(conv); conv.connect(master);
    noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const nd = noise.getChannelData(0); for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
    return true;
  }
  const send = (node, amt) => { const s = ctx.createGain(); s.gain.value = amt; node.connect(s); s.connect(wet) };

  function voice(freq, o = {}) {
    if (!enabled || !ctx) return;
    const { type = 'sine', dur = 0.35, gain = 0.08, attack = 0.005, rev = 0.35, when = 0, glide = 0 } = o;
    const t = ctx.currentTime + when, osc = ctx.createOscillator(), g = ctx.createGain();
    osc.type = type; osc.frequency.setValueAtTime(freq, t);
    if (glide) osc.frequency.exponentialRampToValueAtTime(freq * glide, t + dur);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + attack); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.connect(g); g.connect(master); if (rev) send(g, rev);
    osc.start(t); osc.stop(t + dur + 0.05);
  }
  function noiseHit(o = {}) {
    if (!enabled || !ctx) return;
    const { from = 300, to = 3000, q = 1.2, dur = 0.8, gain = 0.08, attack = 0.2, rev = 0.4, type = 'bandpass' } = o;
    const t = ctx.currentTime, src = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    src.buffer = noise; f.type = type; f.Q.value = q;
    f.frequency.setValueAtTime(from, t); f.frequency.exponentialRampToValueAtTime(to, t + dur * 0.8);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + attack); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f); f.connect(g); g.connect(master); if (rev) send(g, rev);
    src.start(t); src.stop(t + dur + 0.05);
  }

  function startPad() {
    if (pad) return;
    const out = ctx.createGain(); out.gain.value = 0;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 260; lp.Q.value = 0.6;
    const lfo = ctx.createOscillator(), lg = ctx.createGain(); lfo.frequency.value = 0.05; lg.gain.value = 120; lfo.connect(lg); lg.connect(lp.frequency); lfo.start();
    [55, 82.41, 110, 164.81, 220, 277.18].forEach((f, i) => {
      const o = ctx.createOscillator(); o.type = i % 2 ? 'triangle' : 'sawtooth'; o.frequency.value = f; o.detune.value = (i - 2.5) * 6;
      const g = ctx.createGain(); g.gain.value = i === 0 ? 0.5 : 0.22; o.connect(g); g.connect(lp); o.start();
    });
    lp.connect(out); out.connect(master); send(out, 0.7);
    out.gain.linearRampToValueAtTime(0.05, ctx.currentTime + 5);
    pad = { out, lp };
  }
  function startRiser() {
    if (riser) return;
    const src = ctx.createBufferSource(); src.buffer = noise; src.loop = true;
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 2.2; bp.frequency.value = 200;
    const tone = ctx.createOscillator(); tone.type = 'sine'; tone.frequency.value = 110;
    const tg = ctx.createGain(); tg.gain.value = 0.25;
    const g = ctx.createGain(); g.gain.value = 0;
    src.connect(bp); bp.connect(g); tone.connect(tg); tg.connect(g); g.connect(master); send(g, 0.6);
    src.start(); tone.start();
    riser = { g, bp, tone };
  }

  const vibrate = p => { if (!reduce && navigator.vibrate) try { navigator.vibrate(p) } catch (e) {} };

  const Sound = {
    get on() { return enabled },
    set(on) {
      if (!ensure()) { enabled = false; return }
      enabled = on; if (ctx.state === 'suspended') ctx.resume();
      master.gain.cancelScheduledValues(ctx.currentTime);
      master.gain.setTargetAtTime(on ? 0.9 : 0, ctx.currentTime, 0.15);
      if (on) { startPad(); startRiser() }
      store.set('sb-sound', on ? 'on' : 'off');
      document.querySelectorAll('[data-sound-toggle]').forEach(b => { b.setAttribute('aria-pressed', String(on)); b.querySelector('.lbl') && (b.querySelector('.lbl').textContent = on ? 'Sound on' : 'Sound off') });
    },
    saved() { return store.get('sb-sound') },
    /* lub dub: a body heartbeat for the closed hand */
    heartbeat(haptic) {
      voice(62, { dur: 0.32, gain: 0.22, attack: 0.008, rev: 0.15, glide: 0.62 });
      voice(54, { dur: 0.36, gain: 0.15, attack: 0.008, rev: 0.15, glide: 0.6, when: 0.24 });
      if (haptic) vibrate([18, 120, 12]);
    },
    /* continuous tension tied to how far the hand has opened */
    riser(p) {
      if (!riser || !ctx) return;
      const t = ctx.currentTime, on = p > 0.001 && p < 0.999;
      riser.g.gain.setTargetAtTime(on ? 0.02 + p * 0.07 : 0, t, 0.2);
      riser.bp.frequency.setTargetAtTime(200 + p * 2600, t, 0.2);
      riser.tone.frequency.setTargetAtTime(110 + p * 110, t, 0.3);
    },
    fingerTick(i) {
      voice(SCALE[(4 - i) + 3] * 2, { type: 'triangle', dur: 0.9, gain: 0.05, rev: 0.6 });
      voice(SCALE[(4 - i) + 3] * 4, { dur: 0.25, gain: 0.015, rev: 0.5 });
      noiseHit({ from: 4000, to: 6000, q: 4, dur: 0.08, gain: 0.03, attack: 0.002, rev: 0.2 });
      vibrate(12);
    },
    bloom() {
      voice(41, { dur: 2.6, gain: 0.28, attack: 0.02, rev: 0.4, glide: 0.7 });
      [0, 2, 4, 5, 7, 9].forEach((n, k) => voice(SCALE[n] * (k > 3 ? 2 : 1), { type: k % 2 ? 'sine' : 'triangle', dur: 4.5, gain: 0.05, attack: 0.04 + k * 0.02, rev: 0.85, when: k * 0.05 }));
      noiseHit({ from: 800, to: 9000, q: 0.6, dur: 2.2, gain: 0.05, attack: 0.05, rev: 0.9, type: 'highpass' });
      if (pad) pad.lp.frequency.setTargetAtTime(1100, ctx.currentTime, 0.6);
      vibrate([40, 60, 90]);
    },
    dive() {
      noiseHit({ from: 180, to: 5200, q: 1.1, dur: 1.6, gain: 0.12, attack: 0.9, rev: 0.6 });
      voice(110, { type: 'sawtooth', dur: 1.5, gain: 0.03, attack: 1.0, rev: 0.6, glide: 4 });
      vibrate(30);
    },
    arrive() {
      [5, 7, 9].forEach((n, k) => voice(SCALE[n] * 2, { dur: 3.2, gain: 0.04, attack: 0.3, rev: 0.9, when: k * 0.18 }));
      if (pad) pad.lp.frequency.setTargetAtTime(700, ctx.currentTime, 1.2);
    },
    hover(i = 0) { voice(SCALE[(i % 5) + 5] * 2, { dur: 0.16, gain: 0.02, rev: 0.25 }) },
    click() { voice(SCALE[5], { type: 'triangle', dur: 0.5, gain: 0.06 }); voice(SCALE[8], { dur: 0.7, gain: 0.04, when: 0.07 }) },
    chime(i = 0) { const f = SCALE[i % SCALE.length]; voice(f * 2, { type: 'triangle', dur: 1.4, gain: 0.055, rev: 0.7 }); voice(f * 3, { dur: 0.9, gain: 0.02, rev: 0.7, when: 0.04 }); vibrate(10) },
    whoosh() { noiseHit({ from: 260, to: 2600, q: 1.4, dur: 0.8, gain: 0.06, attack: 0.18 }) },
    tick(p) { voice(600 + p * 900, { type: 'square', dur: 0.03, gain: 0.006, rev: 0.05 }) },
    padBright(v) { if (pad) pad.lp.frequency.setTargetAtTime(380 + Math.min(1, v) * 900, ctx.currentTime, 0.4) },
    vibrate,
  };
  window.SBSound = Sound;
})();
