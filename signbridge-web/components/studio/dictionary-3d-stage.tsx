"use client"

import * as React from "react"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { Play, Pause, RotateCcw, Route } from "lucide-react"
import {
  applyGlbHandPose,
  loadGlbHandRig,
  type GlbHandRig
} from "@/components/studio/glb-hand-rig"
import {
  analyzeDictionarySequence,
  interpolateFrameLandmarks,
  landmarks21ToHandPose,
  trailToWorldPoints,
  type DictionarySequenceAnalysis
} from "@/lib/landmarks-to-hand-rig"

export type CameraPreset = "front" | "side" | "top" | "signer"

export interface Dictionary3DStageProps {
  gloss: string
  glossIndexLabel?: string
  twoHand: boolean
  frames: number[][][]
  handshape?: string
  isLoading?: boolean
  onFrameChange?: (frameIdx: number) => void
}

const PRESET_VIEWS: Record<
  CameraPreset,
  { label: string; pos: [number, number, number]; target: [number, number, number] }
> = {
  front: { label: "Front", pos: [0, 0.1, 2.38], target: [0, -0.02, 0] },
  side: { label: "Side", pos: [2.05, 0.18, 0.72], target: [0, -0.02, 0] },
  top: { label: "Top", pos: [0, 2.15, 0.68], target: [0, 0, 0] },
  signer: { label: "Signer POV", pos: [0, 0.3, -2.08], target: [0, 0, 0] }
}

const TOTAL_FRAMES = 32
const TARGET_HAND_HEIGHT = 0.66

export function Dictionary3DStage({
  gloss,
  glossIndexLabel,
  twoHand,
  frames,
  handshape,
  isLoading = false,
  onFrameChange
}: Dictionary3DStageProps) {
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const [preset, setPreset] = React.useState<CameraPreset>("front")
  const [isPlaying, setIsPlaying] = React.useState(true)
  const [looping, setLooping] = React.useState(true)
  const [speed, setSpeed] = React.useState<0.5 | 1 | 1.5>(1)
  const [showTrail, setShowTrail] = React.useState(true)
  const [frameUI, setFrameUI] = React.useState(0)
  const [rigsReady, setRigsReady] = React.useState(false)
  const sequenceKey = `${gloss}:${twoHand}:${frames.length}`
  const [prevSequenceKey, setPrevSequenceKey] = React.useState(sequenceKey)

  if (sequenceKey !== prevSequenceKey) {
    setPrevSequenceKey(sequenceKey)
    setFrameUI(0)
    setIsPlaying(true)
  }

  // Live playback mutable state read directly inside requestAnimationFrame
  const playCursorRef = React.useRef(0)
  const isPlayingRef = React.useRef(true)
  const loopingRef = React.useRef(true)
  const speedRef = React.useRef(1)
  const showTrailRef = React.useRef(true)
  const framesRef = React.useRef<number[][][]>(frames)
  const twoHandRef = React.useRef(twoHand)
  const analysisRef = React.useRef<DictionarySequenceAnalysis | null>(null)
  const worldScaleRef = React.useRef(2.2)
  const cameraTweenRef = React.useRef<{
    active: boolean
    pos: THREE.Vector3
    target: THREE.Vector3
  }>({
    active: false,
    pos: new THREE.Vector3(...PRESET_VIEWS.front.pos),
    target: new THREE.Vector3(...PRESET_VIEWS.front.target)
  })
  const onFrameChangeRef = React.useRef(onFrameChange)

  React.useEffect(() => {
    onFrameChangeRef.current = onFrameChange
  }, [onFrameChange])

  React.useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

  React.useEffect(() => {
    loopingRef.current = looping
  }, [looping])

  React.useEffect(() => {
    speedRef.current = speed
  }, [speed])

  React.useEffect(() => {
    showTrailRef.current = showTrail
  }, [showTrail])

  const trailMeshRebuildRef = React.useRef<(() => void) | null>(null)

  React.useEffect(() => {
    framesRef.current = frames
    twoHandRef.current = twoHand
    playCursorRef.current = 0
    if (frames.length > 0) {
      const analysis = analyzeDictionarySequence(frames, twoHand)
      analysisRef.current = analysis
      worldScaleRef.current = (TARGET_HAND_HEIGHT * 0.54) / analysis.palmSpan
    } else {
      analysisRef.current = null
    }
    isPlayingRef.current = true
    trailMeshRebuildRef.current?.()
  }, [frames, twoHand, gloss])

  const handleSelectPreset = (nextPreset: CameraPreset) => {
    setPreset(nextPreset)
    const spec = PRESET_VIEWS[nextPreset]
    cameraTweenRef.current = {
      active: true,
      pos: new THREE.Vector3(...spec.pos),
      target: new THREE.Vector3(...spec.target)
    }
  }

  const handleScrub = (val: number) => {
    playCursorRef.current = val
    const rounded = Math.round(val)
    setFrameUI(rounded)
    onFrameChangeRef.current?.(rounded)
  }

  const handleTogglePlay = () => {
    if (!isPlaying && playCursorRef.current >= Math.max(0, framesRef.current.length - 1.1)) {
      playCursorRef.current = 0
      setFrameUI(0)
    }
    setIsPlaying((prev) => !prev)
  }

  React.useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let disposed = false
    let rafId: number | null = null

    const width = Math.max(container.clientWidth, 320)
    const height = Math.max(container.clientHeight, 320)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(36, width / height, 0.05, 40)
    camera.position.set(...PRESET_VIEWS.front.pos)

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(width, height)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.06
    renderer.domElement.style.width = "100%"
    renderer.domElement.style.height = "100%"
    renderer.domElement.style.display = "block"
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.enablePan = false
    controls.minDistance = 1.15
    controls.maxDistance = 4.2
    controls.target.set(...PRESET_VIEWS.front.target)
    const onControlStart = () => {
      cameraTweenRef.current.active = false
    }
    controls.addEventListener("start", onControlStart)

    // Warm gallery sculpture lighting
    const hemi = new THREE.HemisphereLight(0xfff5e8, 0x1a1326, 0.98)
    scene.add(hemi)

    const keyLight = new THREE.DirectionalLight(0xfff0dc, 2.05)
    keyLight.position.set(1.6, 2.4, 2.2)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0xc4b5fd, 0.65)
    fillLight.position.set(-2.1, 1.0, 1.5)
    scene.add(fillLight)

    const amberRim = new THREE.DirectionalLight(0xfb6a18, 0.92)
    amberRim.position.set(-0.7, -0.5, -2.1)
    scene.add(amberRim)

    const underPool = new THREE.PointLight(0xfb6a18, 0.85, 2.8)
    underPool.position.set(0, -0.55, 0.25)
    scene.add(underPool)

    // Subtle dark architectural floor grid
    const grid = new THREE.GridHelper(3.4, 22, 0x4b3860, 0x251e34)
    grid.position.y = -0.58
    const gridMats = Array.isArray(grid.material) ? grid.material : [grid.material]
    gridMats.forEach((m) => {
      m.transparent = true
      m.opacity = 0.38
    })
    scene.add(grid)

    // Trajectory arc ribbon & active bead group
    const trailGroup = new THREE.Group()
    scene.add(trailGroup)
    const trailDisposables: Array<{ dispose: () => void }> = []

    const clearTrails = () => {
      while (trailGroup.children.length > 0) {
        trailGroup.remove(trailGroup.children[0])
      }
      while (trailDisposables.length > 0) {
        trailDisposables.pop()?.dispose()
      }
    }

    const rebuildTrails = () => {
      clearTrails()
      const analysis = analysisRef.current
      if (!analysis) return
      const scale = worldScaleRef.current
      const addTrail = (rawPts: [number, number, number][], colorHex: number) => {
        if (rawPts.length < 3) return
        const worldPts = trailToWorldPoints(rawPts, analysis, scale)
        // Deduplicate closely overlapping consecutive points for smooth Catmull-Rom
        const deduped: THREE.Vector3[] = [worldPts[0]]
        for (let i = 1; i < worldPts.length; i++) {
          if (worldPts[i].distanceToSquared(deduped[deduped.length - 1]) > 1e-5) {
            deduped.push(worldPts[i])
          }
        }
        if (deduped.length < 3) return
        const curve = new THREE.CatmullRomCurve3(deduped)
        const geo = new THREE.TubeGeometry(curve, 44, 0.0055, 8, false)
        const mat = new THREE.MeshBasicMaterial({
          color: colorHex,
          transparent: true,
          opacity: 0.42
        })
        const mesh = new THREE.Mesh(geo, mat)
        trailDisposables.push(geo, mat)
        trailGroup.add(mesh)
      }
      addTrail(analysis.wristTrail0, 0xfb6a18)
      if (analysis.twoHand && analysis.wristTrail1.length >= 3) {
        addTrail(analysis.wristTrail1, 0x38bdf8)
      }
    }
    trailMeshRebuildRef.current = rebuildTrails
    rebuildTrails()

    let rightRig: GlbHandRig | null = null
    let leftRig: GlbHandRig | null = null

    const normalizeRigHeight = (rig: GlbHandRig) => {
      rig.root.updateMatrixWorld(true)
      const box = new THREE.Box3().setFromObject(rig.root)
      const h = Math.max(0.05, box.max.y - box.min.y)
      rig.root.scale.setScalar(TARGET_HAND_HEIGHT / h)
    }

    Promise.all([
      loadGlbHandRig({
        url: "/models/hand-right.glb",
        skinColor: 0xe4b898,
        sheenColor: 0xfbe2cc
      }),
      loadGlbHandRig({
        url: "/models/hand-left.glb",
        skinColor: 0xe4b898,
        sheenColor: 0xfbe2cc
      })
    ])
      .then(([rRig, lRig]) => {
        if (disposed) {
          rRig.dispose()
          lRig.dispose()
          return
        }
        normalizeRigHeight(rRig)
        normalizeRigHeight(lRig)
        rRig.root.visible = false
        lRig.root.visible = false
        scene.add(rRig.root)
        scene.add(lRig.root)
        rightRig = rRig
        leftRig = lRig
        setRigsReady(true)
      })
      .catch(() => {})

    const ro = new ResizeObserver((entries) => {
      if (disposed || !entries[0]) return
      const { width: w, height: h } = entries[0].contentRect
      if (w <= 0 || h <= 0) return
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    })
    ro.observe(container)

    let lastTime = performance.now()
    let lastUiFrame = -1

    const animate = () => {
      if (disposed) return
      rafId = requestAnimationFrame(animate)

      const now = performance.now()
      const dt = Math.min(0.05, Math.max(0.001, (now - lastTime) / 1000))
      lastTime = now

      const seq = framesRef.current
      const analysis = analysisRef.current

      if (seq.length > 0 && isPlayingRef.current) {
        const nextCursor = playCursorRef.current + dt * 14.5 * speedRef.current
        const maxFrame = seq.length - 1
        if (nextCursor >= maxFrame) {
          if (loopingRef.current) {
            playCursorRef.current = nextCursor % Math.max(1, maxFrame)
          } else {
            playCursorRef.current = maxFrame
            isPlayingRef.current = false
            setIsPlaying(false)
          }
        } else {
          playCursorRef.current = nextCursor
        }
      }

      const rounded = Math.round(playCursorRef.current)
      if (rounded !== lastUiFrame) {
        lastUiFrame = rounded
        setFrameUI(rounded)
        onFrameChangeRef.current?.(rounded)
      }

      trailGroup.visible = showTrailRef.current

      if (rightRig && leftRig && seq.length > 0 && analysis) {
        const blend = Math.min(0.45, dt * 16)
        const { slot0, slot1 } = interpolateFrameLandmarks(seq, playCursorRef.current, analysis)
        const scale = worldScaleRef.current

        const pose0 = slot0
          ? landmarks21ToHandPose(slot0, analysis.slot0Side, analysis, scale)
          : null
        const pose1 =
          analysis.twoHand && slot1
            ? landmarks21ToHandPose(slot1, analysis.slot1Side, analysis, scale)
            : null

        let activeRight = false
        let activeLeft = false

        const applyToRig = (poseData: ReturnType<typeof landmarks21ToHandPose>) => {
          const targetRig = poseData.side === "right" ? rightRig! : leftRig!
          if (poseData.side === "right") activeRight = true
          else activeLeft = true
          targetRig.root.visible = true
          applyGlbHandPose(targetRig, poseData.pose, blend)
          targetRig.pivot.quaternion.slerp(poseData.pivotQuat, blend)
          targetRig.root.position.lerp(poseData.worldPos, blend)
        }

        if (pose0?.active) applyToRig(pose0)
        if (pose1?.active) applyToRig(pose1)

        if (!activeRight) rightRig.root.visible = false
        if (!activeLeft) leftRig.root.visible = false
      }

      if (cameraTweenRef.current.active) {
        const tBlend = Math.min(1, dt * 7.5)
        camera.position.lerp(cameraTweenRef.current.pos, tBlend)
        controls.target.lerp(cameraTweenRef.current.target, tBlend)
        if (camera.position.distanceTo(cameraTweenRef.current.pos) < 0.01) {
          cameraTweenRef.current.active = false
        }
      }

      controls.update()
      renderer.render(scene, camera)
    }

    rafId = requestAnimationFrame(animate)

    return () => {
      disposed = true
      if (rafId !== null) cancelAnimationFrame(rafId)
      ro.disconnect()
      controls.removeEventListener("start", onControlStart)
      controls.dispose()
      clearTrails()
      rightRig?.dispose()
      leftRig?.dispose()
      gridMats.forEach((m) => m.dispose())
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  const frameTotal = Math.max(frames.length || TOTAL_FRAMES, 1)

  return (
    <div
      className="relative flex flex-col w-full h-full min-h-[480px] rounded-[24px] border border-white/10 overflow-hidden shadow-[0_24px_64px_-16px_rgba(0,0,0,0.75)]"
      style={{
        background:
          "radial-gradient(circle at 50% 42%, rgba(251,106,24,0.14) 0%, rgba(28,23,40,0.92) 48%, rgba(16,13,24,0.98) 100%)"
      }}
    >
      {/* Top Overlay: Sign Title & Camera POV Presets */}
      <div className="pointer-events-none absolute top-0 left-0 right-0 z-10 flex flex-wrap items-start justify-between gap-3 p-5">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#FAF6F0]">
              {gloss || "Select a Gloss"}
            </h2>
            {handshape && (
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 border border-amber-400/30 text-amber-300">
                {handshape}
              </span>
            )}
          </div>
          <p className="text-xs sm:text-sm text-[#ACA5BC] mt-0.5">
            {twoHand ? "Two-Handed Dynamic Sign" : "Single-Handed Sign"}
            {glossIndexLabel ? ` · ${glossIndexLabel}` : ""}
          </p>
        </div>

        <div className="pointer-events-auto flex items-center gap-1 p-1 rounded-full bg-[#171321]/85 border border-white/12 backdrop-blur-md">
          {(Object.keys(PRESET_VIEWS) as CameraPreset[]).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => handleSelectPreset(key)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors cursor-pointer ${
                preset === key
                  ? "bg-white/16 text-[#FAF6F0] border border-white/20"
                  : "text-[#9A93AB] hover:text-[#FAF6F0]"
              }`}
            >
              {PRESET_VIEWS[key].label}
            </button>
          ))}
        </div>
      </div>

      {/* Three.js Canvas Mount */}
      <div
        ref={containerRef}
        className="w-full flex-1 cursor-grab active:cursor-grabbing"
        aria-label={`Interactive 3D hand animation for ${gloss}`}
      />

      {/* Loading Indicator */}
      {(isLoading || !rigsReady) && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-[#14101D]/45 backdrop-blur-[2px]">
          <div className="px-4 py-2 rounded-full bg-[#1E192B]/90 border border-white/12 text-xs font-medium text-[#E5DFEF]">
            {!rigsReady ? "Loading 3D hand rig..." : `Loading ${gloss}...`}
          </div>
        </div>
      )}

      {/* Bottom Floating Playback & Frame Scrubber Dock */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 w-[min(94%,580px)]">
        <div className="flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-[#1A1525]/88 border border-white/12 backdrop-blur-xl shadow-lg">
          <button
            type="button"
            onClick={handleTogglePlay}
            aria-label={isPlaying ? "Pause animation" : "Play animation"}
            className="w-9 h-9 shrink-0 rounded-full bg-[#FB6A18] hover:bg-[#FF8A3D] text-[#14101B] flex items-center justify-center shadow-[0_0_20px_rgba(251,106,24,0.45)] transition-colors cursor-pointer"
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
          </button>

          <div className="flex-1 flex flex-col gap-1 min-w-0">
            <input
              type="range"
              min={0}
              max={frameTotal - 1}
              step={0.1}
              value={frameUI}
              onChange={(e) => {
                setIsPlaying(false)
                handleScrub(parseFloat(e.target.value))
              }}
              aria-label="Scrub sign animation frame"
              className="w-full h-1.5 rounded-full accent-[#FB6A18] bg-white/15 cursor-pointer"
            />
            <div className="flex items-center justify-between gap-2 text-[11px] font-medium text-[#9B94AC] tabular-nums">
              <span className="whitespace-nowrap shrink-0">
                Frame {Math.min(frameTotal, frameUI + 1)} / {frameTotal}
              </span>
              <span className="hidden sm:inline whitespace-nowrap text-[#7E7890]">
                Drag to orbit · Scroll to zoom
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => setLooping((p) => !p)}
            title="Toggle loop playback"
            aria-label="Toggle loop playback"
            className={`p-2 rounded-full border transition-colors cursor-pointer ${
              looping
                ? "bg-amber-500/18 border-amber-400/35 text-amber-300"
                : "bg-white/5 border-white/10 text-[#8F88A0] hover:text-white"
            }`}
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={() => setShowTrail((p) => !p)}
            title="Toggle 3D trajectory path"
            aria-label="Toggle 3D trajectory path"
            className={`p-2 rounded-full border transition-colors cursor-pointer ${
              showTrail
                ? "bg-amber-500/18 border-amber-400/35 text-amber-300"
                : "bg-white/5 border-white/10 text-[#8F88A0] hover:text-white"
            }`}
          >
            <Route className="w-3.5 h-3.5" />
          </button>

          <div className="flex items-center gap-0.5 p-0.5 rounded-full bg-white/6 border border-white/10">
            {([0.5, 1, 1.5] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSpeed(s)}
                className={`px-2 py-1 rounded-full text-[11px] font-semibold transition-colors cursor-pointer ${
                  speed === s ? "bg-white/18 text-[#FAF6F0]" : "text-[#8E87A0] hover:text-white"
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
