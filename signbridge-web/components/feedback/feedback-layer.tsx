"use client"

import * as React from "react"
import { AnimatePresence, motion, useReducedMotion } from "framer-motion"
import { Volume2 } from "lucide-react"
import { onFeedback, type FeedbackEvent, type FeedbackPoint } from "@/lib/feedback/bus"
import { useSignBridgeStore } from "@/lib/signbridge-store"

/**
 * Deaf-first visual feedback for the live studio.
 *
 * - Edge glow ring: a peripheral signal the signer can see while watching their hands.
 *   amber breathing = looking for hands · green = hand tracked · orange = reading a sign
 *   bright bloom = word recognized · red double pulse = no sign / error
 * - Recognition burst + word pop at the hand when a word or letter is committed.
 * - "Speaking aloud" waveform whenever text-to-speech is playing, so the signer
 *   knows the hearing person is hearing it.
 *
 * Flashes stay at or below 2 per second and never cover the screen (WCAG 2.3.1).
 */

type Pulse = { id: number; tone: "word" | "letter" | "sentence" | "warn" }
type Pop = { id: number; label: string; confidence: number; at: FeedbackPoint; small: boolean }

interface FeedbackLayerProps {
  mirrored: boolean
  isStreaming: boolean
  isHandDetected: boolean
  isStrokeActive: boolean
  isSpeaking: boolean
  hasCameraError: boolean
}

const BURST_COLORS = ["#FFA66D", "#FF8A3D", "#FB6A18", "#FFF1C9", "#FAF6F0"]

export function FeedbackLayer({
  mirrored,
  isStreaming,
  isHandDetected,
  isStrokeActive,
  isSpeaking,
  hasCameraError,
}: FeedbackLayerProps) {
  const { preferences } = useSignBridgeStore()
  const reduceMotion = useReducedMotion()
  const level = preferences.effectsLevel === "full" && reduceMotion ? "subtle" : preferences.effectsLevel
  const enabled = preferences.visualAlerts && level !== "off"

  const [pulse, setPulse] = React.useState<Pulse | null>(null)
  const [pops, setPops] = React.useState<Pop[]>([])
  const [capture, setCapture] = React.useState(0)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const particlesRef = React.useRef<
    Array<{ x: number; y: number; vx: number; vy: number; life: number; max: number; r: number; color: string; ring?: boolean }>
  >([])
  const rafRef = React.useRef<number | null>(null)
  const idRef = React.useRef(0)
  const lastPulseAtRef = React.useRef(0)

  const cfgRef = React.useRef({ mirrored, level, enabled })
  React.useEffect(() => {
    cfgRef.current = { mirrored, level, enabled }
  }, [mirrored, level, enabled])

  const toScreen = React.useCallback((p: FeedbackPoint) => {
    const x = cfgRef.current.mirrored ? 1 - p.x : p.x
    return { x: Math.min(0.92, Math.max(0.08, x)), y: Math.min(0.8, Math.max(0.12, p.y)) }
  }, [])

  // Particle animation loop; only runs while particles are alive.
  const tickRef = React.useRef<() => void>(() => {})
  React.useEffect(() => {
    tickRef.current = function tick() {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    ctx.globalCompositeOperation = "lighter"
    const list = particlesRef.current
    for (let i = list.length - 1; i >= 0; i--) {
      const p = list[i]
      p.life += 1
      const t = p.life / p.max
      if (t >= 1) {
        list.splice(i, 1)
        continue
      }
      if (p.ring) {
        ctx.beginPath()
        ctx.arc(p.x * w, p.y * h, p.r + t * 120, 0, Math.PI * 2)
        ctx.strokeStyle = `rgba(255, 138, 61, ${(1 - t) * 0.85})`
        ctx.lineWidth = 3 * (1 - t) + 0.5
        ctx.stroke()
        continue
      }
      p.x += p.vx
      p.y += p.vy
      p.vy += 0.00018
      p.vx *= 0.985
      p.vy *= 0.985
      ctx.beginPath()
      ctx.arc(p.x * w, p.y * h, p.r * (1 - t * 0.6), 0, Math.PI * 2)
      ctx.fillStyle = p.color
      ctx.globalAlpha = 1 - t
      ctx.fill()
      ctx.globalAlpha = 1
    }
    ctx.globalCompositeOperation = "source-over"
    if (list.length > 0) {
      rafRef.current = requestAnimationFrame(tick)
    } else {
      rafRef.current = null
    }
    }
  }, [])

  const burst = React.useCallback(
    (at: FeedbackPoint, strength: number) => {
      const { level: lv } = cfgRef.current
      const p = toScreen(at)
      const list = particlesRef.current
      list.push({ x: p.x, y: p.y, vx: 0, vy: 0, life: 0, max: 42, r: 16, color: "", ring: true })
      if (lv === "full") {
        const n = Math.round(14 + strength * 22)
        for (let i = 0; i < n; i++) {
          const a = Math.random() * Math.PI * 2
          const sp = 0.004 + Math.random() * 0.012 * (0.6 + strength)
          list.push({
            x: p.x,
            y: p.y,
            vx: Math.cos(a) * sp,
            vy: Math.sin(a) * sp * 1.3 - 0.003,
            life: 0,
            max: 40 + Math.random() * 30,
            r: 1.6 + Math.random() * 3.2,
            color: BURST_COLORS[i % BURST_COLORS.length],
          })
        }
      }
      if (rafRef.current == null) rafRef.current = requestAnimationFrame(() => tickRef.current())
    },
    [toScreen]
  )

  React.useEffect(() => {
    const firePulse = (tone: Pulse["tone"]) => {
      const now = performance.now()
      // Never flash faster than 2 per second.
      if (now - lastPulseAtRef.current < 500) return
      lastPulseAtRef.current = now
      setPulse({ id: ++idRef.current, tone })
    }

    const off = onFeedback((e: FeedbackEvent) => {
      if (!cfgRef.current.enabled) {
        if (e.type === "captureProgress") setCapture(e.progress >= 100 ? 0 : e.progress)
        return
      }
      switch (e.type) {
        case "captureProgress":
          setCapture(e.progress >= 100 ? 0 : e.progress)
          break
        case "gloss":
        case "letter": {
          const small = e.type === "letter"
          firePulse(small ? "letter" : "word")
          const at = e.at ?? { x: 0.5, y: 0.45 }
          burst(at, small ? 0.25 : Math.min(1, e.confidence / 100))
          const id = ++idRef.current
          setPops((prev) => [...prev.slice(-2), { id, label: e.label, confidence: e.confidence, at: toScreen(at), small }])
          window.setTimeout(() => setPops((prev) => prev.filter((p) => p.id !== id)), small ? 1100 : 1700)
          break
        }
        case "sentence":
          firePulse("sentence")
          break
        case "noSign":
        case "lowConfidence":
        case "error":
          firePulse("warn")
          break
        default:
          break
      }
    })
    return () => {
      off()
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
    }
  }, [burst, toScreen])

  // Base edge-ring state
  let ringClass = ""
  let ringStyle: React.CSSProperties = {}
  if (enabled && isStreaming) {
    if (hasCameraError) {
      ringStyle = { boxShadow: "inset 0 0 0 3px rgba(248,113,113,0.7), inset 0 0 60px rgba(248,113,113,0.22)" }
    } else if (capture > 0 || isStrokeActive) {
      const width = 2.5 + (capture / 100) * 4
      ringStyle = {
        boxShadow: `inset 0 0 0 ${width}px rgba(255,138,61,0.9), inset 0 0 ${50 + capture}px rgba(251,106,24,0.32)`,
      }
      ringClass = "sb-edge-breathe-fast"
    } else if (isHandDetected) {
      ringStyle = { boxShadow: "inset 0 0 0 2.5px rgba(52,211,153,0.6), inset 0 0 46px rgba(52,211,153,0.16)" }
    } else {
      ringStyle = { boxShadow: "inset 0 0 0 2px rgba(251,106,24,0.32), inset 0 0 40px rgba(251,106,24,0.1)" }
      ringClass = "sb-edge-breathe"
    }
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-[5] rounded-[inherit]">
      {/* Edge ring */}
      <div
        aria-hidden="true"
        className={["absolute inset-0 rounded-[inherit] transition-[box-shadow] duration-500", ringClass].join(" ")}
        style={ringStyle}
      />

      {/* Event pulse */}
      {enabled && pulse && (
        <div
          key={pulse.id}
          aria-hidden="true"
          className={[
            "absolute inset-0 rounded-[inherit]",
            pulse.tone === "warn"
              ? "sb-edge-pulse-warn"
              : pulse.tone === "sentence"
                ? "sb-edge-pulse-sentence"
                : pulse.tone === "letter"
                  ? "sb-edge-pulse-letter"
                  : "sb-edge-pulse-word",
          ].join(" ")}
        />
      )}

      {/* Particle canvas */}
      <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 h-full w-full" />

      {/* Word pop at the hand, then drifts down toward the caption */}
      <AnimatePresence>
        {enabled &&
          pops.map((pop) => {
            const pct = Math.max(0, Math.min(100, pop.confidence))
            const r = pop.small ? 15 : 20
            const c = 2 * Math.PI * r
            return (
              <motion.div
                key={pop.id}
                aria-hidden="true"
                className="absolute -translate-x-1/2 -translate-y-1/2"
                initial={{ left: `${pop.at.x * 100}%`, top: `${pop.at.y * 100}%`, opacity: 0, scale: 0.6 }}
                animate={
                  level === "full"
                    ? {
                        left: [`${pop.at.x * 100}%`, `${pop.at.x * 100}%`, "50%"],
                        top: [`${pop.at.y * 100}%`, `${pop.at.y * 100 - 6}%`, "86%"],
                        opacity: [0, 1, 0],
                        scale: [0.6, 1.08, 0.7],
                      }
                    : { opacity: [0, 1, 0], scale: [0.95, 1, 1] }
                }
                transition={{ duration: pop.small ? 1.0 : 1.6, times: [0, 0.35, 1], ease: [0.2, 0.8, 0.2, 1] }}
              >
                <div
                  className={[
                    "flex items-center whitespace-nowrap rounded-full border border-amber-300/40 bg-[#15121E]/80 shadow-[0_12px_50px_-8px_rgba(251,106,24,0.7)] backdrop-blur-md",
                    pop.small ? "gap-2 py-1.5 pl-1.5 pr-4" : "gap-3 py-2 pl-2 pr-6",
                  ].join(" ")}
                >
                  {/* Confidence meter */}
                  <span className="relative grid place-items-center" style={{ width: r * 2 + 6, height: r * 2 + 6 }}>
                    <svg width={r * 2 + 6} height={r * 2 + 6} className="absolute inset-0 -rotate-90">
                      <circle cx={r + 3} cy={r + 3} r={r} fill="none" stroke="rgba(255,255,255,0.14)" strokeWidth={4} />
                      <circle
                        cx={r + 3}
                        cy={r + 3}
                        r={r}
                        fill="none"
                        stroke="#FF8A3D"
                        strokeWidth={4}
                        strokeLinecap="round"
                        strokeDasharray={c}
                        strokeDashoffset={c * (1 - pct / 100)}
                      />
                    </svg>
                    <span className={["relative font-mono font-semibold text-amber-200", pop.small ? "text-[9px]" : "text-[11px]"].join(" ")}>
                      {Math.round(pct)}%
                    </span>
                  </span>
                  <span
                    className={[
                      "font-display font-extrabold tracking-tight text-white drop-shadow-[0_2px_18px_rgba(251,106,24,0.6)]",
                      pop.small ? "text-2xl" : "text-3xl sm:text-4xl",
                    ].join(" ")}
                  >
                    {pop.label}
                  </span>
                </div>
              </motion.div>
            )
          })}
      </AnimatePresence>

      {/* Speaking-aloud indicator: the visual equivalent of TTS audio */}
      <AnimatePresence>
        {isSpeaking && (
          <motion.div
            role="status"
            aria-live="polite"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute left-1/2 top-16 sm:top-20 -translate-x-1/2 flex items-center gap-2.5 rounded-full border border-amber-400/40 bg-[#1B1726]/85 px-4 py-2 text-xs font-semibold text-amber-100 shadow-[0_10px_40px_-10px_rgba(251,106,24,0.6)] backdrop-blur-xl"
          >
            <Volume2 className="h-4 w-4 text-amber-300" />
            <span>Speaking aloud</span>
            <span className="flex h-4 items-end gap-[3px]" aria-hidden="true">
              {[0, 1, 2, 3, 4].map((i) => (
                <i key={i} className="sb-wave-bar block w-[3px] rounded-full bg-amber-300" style={{ animationDelay: `${i * -0.18}s` }} />
              ))}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
