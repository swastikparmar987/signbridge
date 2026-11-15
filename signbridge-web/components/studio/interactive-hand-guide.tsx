"use client"

import * as React from "react"
import * as THREE from "three"
import { RotateCcw, Minimize2, Maximize2 } from "lucide-react"
import { useReducedMotion } from "framer-motion"
import type { RecognitionMode } from "@/lib/signbridge-store"
import { addWarmStudioLights, HAND_POSES, type HandGuidePreset } from "./hand-3d-rig"
import { applyGlbHandPose, loadGlbHandRig, type GlbHandRig } from "./glb-hand-rig"

interface InteractiveHandGuideProps {
  mode: RecognitionMode
  activeLabel?: string | null
}

const GUIDE_PRESETS: HandGuidePreset[] = [
  "OPEN",
  "HELLO",
  "LOVE",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "I",
  "L",
  "O",
  "S",
  "U",
  "V",
  "W",
  "Y",
]

function resolvePresetFromContext(mode: RecognitionMode, activeLabel?: string | null): HandGuidePreset {
  if (activeLabel) {
    const clean = activeLabel.trim().toUpperCase()
    if (clean in HAND_POSES) {
      return clean as HandGuidePreset
    }
    if (clean === "HI" || clean === "WAVE") return "HELLO"
    if (clean === "ILY" || clean === "I LOVE YOU") return "LOVE"
  }
  if (mode === "Fingerspelling") return "D"
  if (mode === "Continuous") return "HELLO"
  return "OPEN"
}

export function InteractiveHandGuide({ mode, activeLabel }: InteractiveHandGuideProps) {
  const prefersReduced = useReducedMotion()
  const [expanded, setExpanded] = React.useState(false)
  const [manualPreset, setManualPreset] = React.useState<HandGuidePreset | null>(null)
  const [modelReady, setModelReady] = React.useState(false)

  const currentPreset: HandGuidePreset = React.useMemo(
    () => manualPreset ?? resolvePresetFromContext(mode, activeLabel),
    [manualPreset, mode, activeLabel]
  )

  const currentPresetRef = React.useRef<HandGuidePreset>(currentPreset)
  React.useEffect(() => {
    currentPresetRef.current = currentPreset
  }, [currentPreset])

  const dragOrbitRef = React.useRef({ x: 0.08, y: 0.2, dragging: false, lastX: 0, lastY: 0 })
  const mountRef = React.useRef<HTMLDivElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)

  React.useEffect(() => {
    const container = mountRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return

    let disposed = false
    let renderer: THREE.WebGLRenderer | null = null
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
        powerPreference: "low-power",
      })
    } catch {
      return
    }

    renderer.setClearColor(0x000000, 0)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))

    const scene = new THREE.Scene()
    addWarmStudioLights(scene)

    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 20)
    camera.position.set(0, 0.08, 4.1)

    const syncSize = () => {
      if (!renderer || !container) return
      const w = Math.max(40, container.clientWidth)
      const h = Math.max(40, container.clientHeight)
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    syncSize()

    const ro = new ResizeObserver(() => syncSize())
    ro.observe(container)

    let glbRig: GlbHandRig | null = null
    let rafId = 0
    let last = performance.now()

    loadGlbHandRig({ skinColor: 0xebb683, sheenColor: 0xffe4ca })
      .then((rig) => {
        if (disposed) {
          rig.dispose()
          return
        }
        glbRig = rig
        scene.add(rig.root)
        applyGlbHandPose(rig, HAND_POSES[currentPresetRef.current], 1)
        setModelReady(true)
      })
      .catch(() => {})

    const animate = (now: number) => {
      rafId = requestAnimationFrame(animate)
      if (!renderer || !glbRig) return
      const dt = Math.min(0.05, Math.max(0.001, (now - last) / 1000))
      last = now
      const t = now / 1000

      const preset = HAND_POSES[currentPresetRef.current] ?? HAND_POSES.OPEN
      const orbit = dragOrbitRef.current

      if (!orbit.dragging && !prefersReduced) {
        orbit.y += (Math.sin(t * 0.95) * 0.42 - orbit.y) * Math.min(1, dt * 1.4)
        orbit.x += (0.06 - orbit.x) * Math.min(1, dt * 1.4)
      }

      glbRig.root.position.y = prefersReduced ? 0 : Math.sin(t * 1.8) * 0.035
      applyGlbHandPose(glbRig, preset, Math.min(1, dt * 8), [orbit.x, orbit.y, 0])
      renderer.render(scene, camera)
    }

    rafId = requestAnimationFrame(animate)

    return () => {
      disposed = true
      cancelAnimationFrame(rafId)
      ro.disconnect()
      glbRig?.dispose()
      renderer?.dispose()
    }
  }, [prefersReduced])

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!expanded) return
    e.currentTarget.setPointerCapture(e.pointerId)
    dragOrbitRef.current.dragging = true
    dragOrbitRef.current.lastX = e.clientX
    dragOrbitRef.current.lastY = e.clientY
  }

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const orbit = dragOrbitRef.current
    if (!orbit.dragging) return
    const dx = e.clientX - orbit.lastX
    const dy = e.clientY - orbit.lastY
    orbit.lastX = e.clientX
    orbit.lastY = e.clientY
    orbit.y = Math.max(-1.8, Math.min(1.8, orbit.y + dx * 0.012))
    orbit.x = Math.max(-1.0, Math.min(1.0, orbit.x + dy * 0.012))
  }

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    dragOrbitRef.current.dragging = false
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
  }

  return (
    <div
      className={[
        "rounded-2xl border border-white/15 bg-[#191525]/88 text-[#FAF6F0] shadow-[0_18px_40px_-10px_rgba(0,0,0,0.75)] backdrop-blur-xl transition-all duration-300 overflow-hidden",
        expanded ? "w-[264px]" : "w-[104px]",
      ].join(" ")}
    >
      {expanded && (
        <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
            <span className="text-[11px] font-semibold tracking-wide text-white/90">
              3D Sign Guide · {currentPreset}
            </span>
          </div>
          <div className="flex items-center gap-1">
            {manualPreset && (
              <button
                type="button"
                title="Sync to live prediction"
                onClick={() => setManualPreset(null)}
                className="rounded-md p-1 text-white/60 hover:bg-white/10 hover:text-white"
              >
                <RotateCcw className="h-3 w-3" />
              </button>
            )}
            <button
              type="button"
              aria-label="Minimize 3D sign guide"
              onClick={() => setExpanded(false)}
              className="rounded-md p-1 text-white/60 hover:bg-white/10 hover:text-white"
            >
              <Minimize2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}

      <div
        ref={mountRef}
        onClick={() => {
          if (!expanded) setExpanded(true)
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        className={[
          "relative w-full select-none flex items-center justify-center",
          expanded ? "h-[168px] cursor-grab active:cursor-grabbing" : "h-[88px] cursor-pointer",
        ].join(" ")}
      >
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-4 rounded-full bg-amber-400/12 blur-xl"
        />
        <canvas ref={canvasRef} className="relative z-10 h-full w-full" />
        {!modelReady && (
          <span className="absolute inset-0 flex items-center justify-center text-[11px] text-white/45">
            3D
          </span>
        )}
        {expanded && (
          <span className="pointer-events-none absolute bottom-1.5 right-2.5 text-[10px] text-white/40">
            Drag to rotate
          </span>
        )}
      </div>

      {!expanded ? (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="flex w-full items-center justify-center gap-1 border-t border-white/10 px-2 py-1.5 text-[11px] font-medium text-white/80 hover:bg-white/5 hover:text-white"
        >
          <span>Sign guide</span>
          <Maximize2 className="h-2.5 w-2.5 opacity-60" />
        </button>
      ) : (
        <div className="border-t border-white/10 p-2.5">
          <div className="flex flex-wrap gap-1">
            {GUIDE_PRESETS.map((preset) => {
              const active = currentPreset === preset
              return (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setManualPreset(preset)}
                  className={[
                    "rounded-md px-1.5 py-0.5 text-[10px] font-semibold transition",
                    active
                      ? "bg-amber-400 text-[#18120C]"
                      : "bg-white/8 text-white/70 hover:bg-white/15 hover:text-white",
                  ].join(" ")}
                >
                  {preset}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
