import type { FeedbackEvent } from "./bus"

/**
 * Optional synthesized sound cues (Web Audio, no files). Off unless the user
 * turns "Sound effects" on. Cues are short and soft so they never compete with
 * spoken TTS output.
 */

// A major pentatonic, two octaves
const SCALE = [220, 246.94, 277.18, 329.63, 369.99, 440, 493.88, 554.37, 659.25, 739.99]

let ctx: AudioContext | null = null
let master: GainNode | null = null
let wet: GainNode | null = null

function impulse(c: AudioContext, seconds: number, decay: number) {
  const len = Math.floor(c.sampleRate * seconds)
  const buf = c.createBuffer(2, len, c.sampleRate)
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch)
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay)
  }
  return buf
}

function ensure(): boolean {
  if (ctx) {
    if (ctx.state === "suspended") void ctx.resume()
    return true
  }
  if (typeof window === "undefined") return false
  const AC =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AC) return false
  ctx = new AC()
  const comp = ctx.createDynamicsCompressor()
  comp.threshold.value = -18
  comp.ratio.value = 4
  master = ctx.createGain()
  master.gain.value = 0.7
  master.connect(comp)
  comp.connect(ctx.destination)
  const verb = ctx.createConvolver()
  verb.buffer = impulse(ctx, 2.2, 2.6)
  wet = ctx.createGain()
  wet.gain.value = 0.6
  wet.connect(verb)
  verb.connect(master)
  return true
}

type VoiceOpts = {
  type?: OscillatorType
  dur?: number
  gain?: number
  attack?: number
  send?: number
  when?: number
  glide?: number
}

function voice(freq: number, o: VoiceOpts = {}) {
  if (!ctx || !master || !wet) return
  const { type = "sine", dur = 0.35, gain = 0.06, attack = 0.006, send = 0.35, when = 0, glide = 0 } = o
  const t = ctx.currentTime + when
  const osc = ctx.createOscillator()
  const g = ctx.createGain()
  osc.type = type
  osc.frequency.setValueAtTime(freq, t)
  if (glide) osc.frequency.exponentialRampToValueAtTime(freq * glide, t + dur)
  g.gain.setValueAtTime(0.0001, t)
  g.gain.exponentialRampToValueAtTime(gain, t + attack)
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur)
  osc.connect(g)
  g.connect(master)
  if (send) {
    const s = ctx.createGain()
    s.gain.value = send
    g.connect(s)
    s.connect(wet)
  }
  osc.start(t)
  osc.stop(t + dur + 0.05)
}

function letterIndex(label: string) {
  const c = label.toUpperCase().charCodeAt(0)
  return Number.isFinite(c) ? (c - 65 + 26) % SCALE.length : 0
}

/** Must be called from a user gesture at least once so browsers allow audio. */
export function unlockSound() {
  ensure()
}

export function playFor(event: FeedbackEvent) {
  if (!ensure()) return
  switch (event.type) {
    case "letter":
      voice(SCALE[letterIndex(event.label)] * 2, { type: "triangle", dur: 0.5, gain: 0.045, send: 0.45 })
      break
    case "gloss":
      voice(SCALE[5], { type: "triangle", dur: 0.7, gain: 0.06 })
      voice(SCALE[8], { dur: 0.9, gain: 0.045, when: 0.08, send: 0.5 })
      break
    case "sentence":
      ;[0, 2, 4, 7].forEach((n, k) => voice(SCALE[n], { type: "triangle", dur: 1.6, gain: 0.04, send: 0.65, when: k * 0.07 }))
      break
    case "handFound":
      voice(SCALE[7] * 2, { dur: 0.14, gain: 0.018, send: 0.2 })
      break
    case "strokeStart":
      voice(SCALE[3] * 2, { dur: 0.12, gain: 0.015, send: 0.15 })
      break
    case "noSign":
    case "lowConfidence":
      voice(196, { type: "sine", dur: 0.35, gain: 0.035, glide: 0.85, send: 0.2 })
      break
    case "error":
      voice(164.81, { type: "triangle", dur: 0.25, gain: 0.04, send: 0.15 })
      voice(146.83, { type: "triangle", dur: 0.35, gain: 0.04, when: 0.16, send: 0.15 })
      break
    case "modeChange":
      voice(SCALE[4] * 2, { dur: 0.18, gain: 0.03, send: 0.3 })
      break
    case "ui":
      if (event.kind === "start") {
        voice(SCALE[5], { type: "triangle", dur: 0.5, gain: 0.05 })
        voice(SCALE[9], { dur: 0.6, gain: 0.035, when: 0.07, send: 0.5 })
      } else {
        voice(SCALE[6] * 2, { dur: 0.1, gain: 0.02, send: 0.15 })
      }
      break
    default:
      break
  }
}
