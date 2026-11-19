/**
 * Deaf-first feedback bus.
 *
 * Recognition code emits events here; the FeedbackLayer turns each one into a
 * visual signal first, a vibration second and (only if enabled) a sound.
 */

export type FeedbackPoint = { x: number; y: number } // normalized 0..1 in screen space

export type FeedbackEvent =
  | { type: "handFound"; at?: FeedbackPoint }
  | { type: "handLost" }
  | { type: "strokeStart"; at?: FeedbackPoint }
  | { type: "captureProgress"; progress: number }
  | { type: "letter"; label: string; confidence: number; at?: FeedbackPoint }
  | { type: "gloss"; label: string; confidence: number; at?: FeedbackPoint }
  | { type: "sentence"; text: string }
  | { type: "noSign" }
  | { type: "lowConfidence" }
  | { type: "error"; kind: string }
  | { type: "speakStart" }
  | { type: "speakEnd" }
  | { type: "modeChange"; mode: string }
  | { type: "ui"; kind: "tap" | "start" | "toggle" }

type Listener = (event: FeedbackEvent) => void

const listeners = new Set<Listener>()

export function emitFeedback(event: FeedbackEvent) {
  listeners.forEach((fn) => {
    try {
      fn(event)
    } catch {
      // A broken effect must never break recognition.
    }
  })
}

export function onFeedback(fn: Listener) {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

// Dev-only: trigger effects from the browser console, e.g.
// __sbFeedback({ type: "gloss", label: "HELLO", confidence: 92, at: { x: 0.5, y: 0.4 } })
if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  ;(window as unknown as { __sbFeedback?: typeof emitFeedback }).__sbFeedback = emitFeedback
}
