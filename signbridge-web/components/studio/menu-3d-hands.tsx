"use client"

import * as React from "react"
import { useReducedMotion } from "framer-motion"
import type { RecognitionMode } from "@/lib/signbridge-store"

export interface ModePointerRef {
  x: number
  y: number
}

interface InteractiveMenuHandProps {
  mode: RecognitionMode
  isHovered: boolean
  isSelected: boolean
  pointerRef: React.MutableRefObject<Record<RecognitionMode, ModePointerRef>>
}

const SPELL_LETTERS = ["D", "A", "S", "L"] as const

export function InteractiveMenuHand({
  mode,
  isHovered,
  isSelected,
  pointerRef,
}: InteractiveMenuHandProps) {
  const prefersReduced = useReducedMotion()
  const stageRef = React.useRef<HTMLDivElement | null>(null)
  const primaryRef = React.useRef<HTMLDivElement | null>(null)
  const secondaryRef = React.useRef<HTMLDivElement | null>(null)
  const ringOuterRef = React.useRef<SVGGElement | null>(null)
  const ringInnerRef = React.useRef<SVGGElement | null>(null)
  const sheenRef = React.useRef<HTMLDivElement | null>(null)

  const hoveredRef = React.useRef(isHovered)
  const selectedRef = React.useRef(isSelected)
  const [activeLetterIdx, setActiveLetterIdx] = React.useState(0)

  React.useEffect(() => {
    hoveredRef.current = isHovered
  }, [isHovered])

  React.useEffect(() => {
    selectedRef.current = isSelected
  }, [isSelected])

  // Subtle letter pulse in Fingerspelling mode when active/hovered
  React.useEffect(() => {
    if (mode !== "Fingerspelling" || prefersReduced) return
    const intervalMs = isHovered ? 950 : isSelected ? 1400 : 2200
    const id = window.setInterval(() => {
      setActiveLetterIdx((idx) => (idx + 1) % SPELL_LETTERS.length)
    }, intervalMs)
    return () => window.clearInterval(id)
  }, [mode, isHovered, isSelected, prefersReduced])

  // 60fps interpolated 3D perspective stage driven without React re-renders
  React.useEffect(() => {
    if (prefersReduced) return

    let rafId = 0
    let smoothPx = 0
    let smoothPy = 0
    let orbitAngle = 0
    let lastTime = performance.now()
    const startTime = lastTime

    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000)
      lastTime = now
      const t = (now - startTime) / 1000

      const target = pointerRef.current[mode] ?? { x: 0, y: 0 }
      const lerpFactor = 1 - Math.exp(-dt * 10)
      smoothPx += (target.x - smoothPx) * lerpFactor
      smoothPy += (target.y - smoothPy) * lerpFactor

      const hov = hoveredRef.current
      const sel = selectedRef.current
      const energy = hov ? 1.0 : sel ? 0.55 : 0.25

      const rotY = smoothPx * (hov ? 16 : 8)
      const rotX = -smoothPy * (hov ? 12 : 6)

      if (stageRef.current) {
        const liftZ = hov ? 20 : sel ? 10 : 0
        const scale = hov ? 1.07 : sel ? 1.03 : 1.0
        stageRef.current.style.transform = `perspective(900px) rotateX(${rotX.toFixed(2)}deg) rotateY(${rotY.toFixed(2)}deg) translateZ(${liftZ}px) scale(${scale})`
      }

      if (sheenRef.current) {
        const sx = 50 + smoothPx * 28
        const sy = 42 + smoothPy * 24
        sheenRef.current.style.background = `radial-gradient(circle at ${sx.toFixed(1)}% ${sy.toFixed(1)}%, rgba(255, 224, 166, ${(hov ? 0.26 : sel ? 0.17 : 0.08).toFixed(2)}), transparent 62%)`
      }

      if (mode === "Word" && primaryRef.current) {
        const wave = Math.sin(t * (hov ? 3.2 : 1.45)) * (hov ? 4.4 : 1.6)
        const bobY = Math.sin(t * 1.9) * (hov ? -4.5 : -2.2)
        primaryRef.current.style.transform = `translate3d(0px, ${bobY.toFixed(2)}px, 14px) rotateZ(${wave.toFixed(2)}deg)`
      } else if (mode === "Continuous") {
        const speed = hov ? 48 : sel ? 26 : 14
        orbitAngle = (orbitAngle + dt * speed) % 360

        if (ringOuterRef.current) {
          ringOuterRef.current.style.transform = `rotate(${orbitAngle.toFixed(2)}deg)`
        }
        if (ringInnerRef.current) {
          ringInnerRef.current.style.transform = `rotate(${(-orbitAngle * 1.35).toFixed(2)}deg)`
        }

        const sep = hov ? 6 : sel ? 2.5 : 0
        const phaseA = Math.sin(t * (2.1 + energy * 0.8))
        const phaseB = Math.cos(t * (2.1 + energy * 0.8))

        if (primaryRef.current) {
          const lx = -19 - sep + smoothPx * 4
          const ly = -7 + phaseA * (3.2 + energy * 2)
          const lrot = -10 + phaseA * 3.5
          primaryRef.current.style.transform = `translate3d(${lx.toFixed(2)}px, ${ly.toFixed(2)}px, 16px) rotateZ(${lrot.toFixed(2)}deg)`
        }
        if (secondaryRef.current) {
          const rx = 20 + sep - smoothPx * 4
          const ry = 10 + phaseB * (3.2 + energy * 2)
          const rrot = 9 - phaseB * 3.5
          secondaryRef.current.style.transform = `translate3d(${rx.toFixed(2)}px, ${ry.toFixed(2)}px, 24px) rotateZ(${rrot.toFixed(2)}deg)`
        }
      } else if (mode === "Fingerspelling" && primaryRef.current) {
        const bobY = Math.sin(t * 2.15) * (hov ? -4.5 : -2.2)
        const tilt = Math.sin(t * 1.6) * (hov ? 2.8 : 1.1)
        primaryRef.current.style.transform = `translate3d(0px, ${bobY.toFixed(2)}px, 16px) rotateZ(${tilt.toFixed(2)}deg)`
      }

      rafId = window.requestAnimationFrame(tick)
    }

    rafId = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(rafId)
  }, [mode, pointerRef, prefersReduced])

  const glowIntensity = isHovered
    ? "drop-shadow(0 16px 28px rgba(251, 106, 24, 0.36))"
    : isSelected
      ? "drop-shadow(0 14px 24px rgba(251, 106, 24, 0.25))"
      : "drop-shadow(0 10px 18px rgba(0, 0, 0, 0.45))"

  return (
    <div
      className="relative flex h-full w-full items-center justify-center overflow-visible"
      style={{ transformStyle: "preserve-3d" }}
    >
      {/* Dynamic cursor-tracking warm clay rim sheen */}
      <div
        ref={sheenRef}
        aria-hidden="true"
        className="pointer-events-none absolute inset-1 rounded-2xl transition-opacity duration-300"
      />

      {/* 3D Perspective Stage */}
      <div
        ref={stageRef}
        className="relative flex h-full w-full items-center justify-center transition-transform duration-200 ease-out will-change-transform"
        style={{ transformStyle: "preserve-3d" }}
      >
        {mode === "Word" && (
          <div ref={primaryRef} className="relative flex items-center justify-center will-change-transform">
            <img
              src="/hands/words-hand.png"
              alt="Single sculpted 3D open palm for isolated word signs"
              draggable={false}
              decoding="async"
              className="h-[126px] sm:h-[138px] w-auto object-contain select-none transition-[filter] duration-300"
              style={{ filter: glowIntensity }}
            />
          </div>
        )}

        {mode === "Continuous" && (
          <div className="relative flex h-full w-full items-center justify-center">
            {/* Opposing golden orbital motion arcs behind the dual clay hands */}
            <svg
              aria-hidden="true"
              viewBox="0 0 200 150"
              className={[
                "pointer-events-none absolute inset-0 m-auto h-[138px] w-[184px] transition-opacity duration-300",
                isHovered || isSelected ? "opacity-95" : "opacity-65",
              ].join(" ")}
            >
              <defs>
                <linearGradient id="sbOrbitGradA" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#FFC9A5" stopOpacity="0.95" />
                  <stop offset="55%" stopColor="#FB6A18" stopOpacity="0.65" />
                  <stop offset="100%" stopColor="#FB6A18" stopOpacity="0.05" />
                </linearGradient>
                <linearGradient id="sbOrbitGradB" x1="100%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#FF8A3D" stopOpacity="0.85" />
                  <stop offset="100%" stopColor="#FB6A18" stopOpacity="0.08" />
                </linearGradient>
              </defs>
              <g ref={ringOuterRef} style={{ transformOrigin: "100px 75px" }}>
                <circle
                  cx="100"
                  cy="75"
                  r="58"
                  fill="none"
                  stroke="url(#sbOrbitGradA)"
                  strokeWidth={isHovered ? "2.4" : "1.8"}
                  strokeDasharray="115 62 42 54"
                  strokeLinecap="round"
                />
              </g>
              <g ref={ringInnerRef} style={{ transformOrigin: "100px 75px" }}>
                <circle
                  cx="100"
                  cy="75"
                  r="45"
                  fill="none"
                  stroke="url(#sbOrbitGradB)"
                  strokeWidth="1.5"
                  strokeDasharray="68 78"
                  strokeLinecap="round"
                />
              </g>
            </svg>

            <div
              ref={primaryRef}
              className="absolute flex items-center justify-center will-change-transform"
              style={{ transform: "translate3d(-19px, -7px, 16px) rotateZ(-10deg)" }}
            >
              <img
                src="/hands/cont-left.png"
                alt="Left continuous signing 3D sculpted hand"
                draggable={false}
                decoding="async"
                className="h-[88px] sm:h-[96px] w-auto object-contain select-none transition-[filter] duration-300"
                style={{ filter: glowIntensity }}
              />
            </div>

            <div
              ref={secondaryRef}
              className="absolute flex items-center justify-center will-change-transform"
              style={{ transform: "translate3d(20px, 10px, 24px) rotateZ(9deg)" }}
            >
              <img
                src="/hands/cont-right.png"
                alt="Right continuous signing 3D sculpted hand"
                draggable={false}
                decoding="async"
                className="h-[88px] sm:h-[96px] w-auto object-contain select-none transition-[filter] duration-300"
                style={{ filter: glowIntensity }}
              />
            </div>
          </div>
        )}

        {mode === "Fingerspelling" && (
          <div className="relative flex h-full w-full items-center justify-center">
            <div ref={primaryRef} className="relative flex items-center justify-center will-change-transform">
              <img
                src="/hands/fingerspelling-hand.png"
                alt="Sculpted 3D ASL fingerspelling D-shape hand"
                draggable={false}
                decoding="async"
                className="h-[126px] sm:h-[138px] w-auto object-contain select-none transition-[filter] duration-300"
                style={{ filter: glowIntensity }}
              />
            </div>

            {/* Subtle floating ASL letter pulse badge beside the index posture */}
            <div
              aria-hidden="true"
              className={[
                "pointer-events-none absolute top-2 right-4 sm:right-6 flex h-7 min-w-7 items-center justify-center rounded-full border px-2 font-mono text-[11px] font-bold tracking-wider backdrop-blur-md transition-all duration-300",
                isHovered || isSelected
                  ? "border-amber-300/60 bg-amber-400/20 text-amber-200 shadow-[0_0_16px_rgba(251,106,24,0.35)] scale-105"
                  : "border-white/15 bg-white/8 text-white/60 scale-95",
              ].join(" ")}
            >
              {SPELL_LETTERS[activeLetterIdx]}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
