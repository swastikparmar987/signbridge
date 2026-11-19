import type { FeedbackEvent } from "./bus"

/**
 * Distinct vibration patterns so each event can be told apart by touch alone.
 * Silently does nothing where the Vibration API is unavailable (e.g. iOS Safari).
 */
const PATTERNS: Partial<Record<FeedbackEvent["type"], number | number[]>> = {
  handFound: 12,
  strokeStart: 8,
  letter: 15,
  gloss: [30, 40, 30],
  sentence: [70, 50, 25, 50, 70],
  noSign: [18, 60, 18],
  error: [25, 45, 25, 45, 25],
  modeChange: 10,
  ui: 8,
}

let lastAt = 0

export function vibrateFor(event: FeedbackEvent) {
  if (typeof navigator === "undefined" || typeof navigator.vibrate !== "function") return
  const pattern = PATTERNS[event.type]
  if (pattern == null) return
  const now = performance.now()
  // Light events shouldn't buzz continuously.
  if (typeof pattern === "number" && now - lastAt < 120) return
  lastAt = now
  try {
    navigator.vibrate(pattern)
  } catch {
    // ignore
  }
}
