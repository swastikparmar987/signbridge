"use client"

import * as React from "react"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import {
  Camera,
  ChevronDown,
  Infinity as InfinityIcon,
  Hand,
  Languages,
  Pause,
  Play,
  Volume2,
  Maximize2,
  Minimize2,
  Copy,
  Check,
  RotateCcw,
  RefreshCw,
  SignalHigh
} from "lucide-react"
import { RecognitionMode } from "@/lib/signbridge-store"
import { useAslRecognition } from "@/lib/use-asl-recognition"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"

const MODE_OPTIONS: Array<{
  id: RecognitionMode
  label: string
  subtitle: string
}> = [
  {
    id: "Continuous",
    label: "Continuous",
    subtitle: "Continuous signing → Real-time text"
  },
  {
    id: "Word",
    label: "Word",
    subtitle: "Word capture → Isolated ASL sign"
  },
  {
    id: "Fingerspelling",
    label: "Fingerspelling",
    subtitle: "Fingerspelling → Alphabet A–Z to text"
  }
]

function ModeIcon({ id }: { id: RecognitionMode }) {
  if (id === "Continuous") {
    return <InfinityIcon className="w-4 h-4 shrink-0" strokeWidth={2.2} />
  }
  if (id === "Word") {
    return <Hand className="w-4 h-4 shrink-0" strokeWidth={2.1} />
  }
  return (
    <span className="w-4 h-4 rounded-full border border-current text-[10px] font-extrabold inline-flex items-center justify-center shrink-0">
      A
    </span>
  )
}

export function LiveRecognitionCard({
  onExpandStudio
}: {
  onExpandStudio?: () => void
}) {
  const prefersReduced = useReducedMotion()
  const {
    videoRef,
    canvasRef,
    mode,
    setMode,
    modelVariant,
    setModelVariant,
    devices,
    selectedDeviceId,
    selectCamera,
    isStreaming,
    isPaused,
    cameraError,
    fps,
    isHandDetected,
    recognizedText,
    activePrediction,
    topCandidates,
    captureState,
    captureProgress,
    startWordCapture,
    selectCandidate,
    isSpeaking,
    retryCamera,
    togglePause,
    clearOutput,
    copyOutput,
    speakOutput
  } = useAslRecognition("Continuous")

  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (mode !== "Word" || !isStreaming || isPaused) return
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return
      if (e.code === "Space") {
        e.preventDefault()
        startWordCapture()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [mode, isStreaming, isPaused, startWordCapture])

  const previewWrapperRef = React.useRef<HTMLDivElement | null>(null)
  const [isFullscreen, setIsFullscreen] = React.useState(false)
  const [copied, setCopied] = React.useState(false)

  React.useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement))
    }
    document.addEventListener("fullscreenchange", onFullscreenChange)
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange)
  }, [])

  const toggleFullscreen = async () => {
    const el = previewWrapperRef.current
    if (!el) return
    try {
      if (!document.fullscreenElement) {
        await el.requestFullscreen()
      } else {
        await document.exitFullscreen()
      }
    } catch {
      // Fullscreen unavailable in embedded/headless frame
    }
  }

  const handleCopy = async () => {
    await copyOutput()
    if (recognizedText) {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    }
  }

  const activeSubtitle =
    MODE_OPTIONS.find((m) => m.id === mode)?.subtitle ||
    "Continuous signing → Real-time text"

  return (
    <motion.section
      initial={prefersReduced ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: prefersReduced ? 0 : 0.25 }}
      aria-label="Live ASL Recognition"
      className="sb-surface-card rounded-[26px] p-4 sm:p-5 flex flex-col gap-3.5 min-w-0 w-full"
    >
      {/* Card Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 min-w-0">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#FF8C1A] to-[#E85D04] text-white flex items-center justify-center shadow-[0_8px_18px_-6px_rgba(255,121,0,0.55)] shrink-0">
            <Camera className="w-5 h-5" strokeWidth={2.2} />
          </div>
          <div className="min-w-0">
            <h2 className="text-[17px] sm:text-lg font-extrabold tracking-tight text-[#FAF6F0] leading-snug truncate">
              Live ASL Recognition
            </h2>
            <p className="text-xs font-medium text-[#ACA5BC] truncate">
              {activeSubtitle}
            </p>
          </div>
        </div>

        {/* Model Vocabulary Switcher + Camera Selection Dropdown */}
        <div className="flex flex-wrap items-center gap-2 self-stretch sm:self-auto shrink-0 min-w-0">
          {mode !== "Fingerspelling" && (
            <div
              role="group"
              aria-label="Vocabulary scope"
              className="inline-flex items-center rounded-xl bg-[#231D32] border border-white/10 p-0.5 text-[11px] font-bold"
            >
              <button
                type="button"
                onClick={() => setModelVariant("production_exp13")}
                className={`px-2.5 py-1.5 rounded-lg transition cursor-pointer ${
                  modelVariant === "production_exp13"
                    ? "bg-[#D97706] text-white shadow-2xs"
                    : "text-[#ACA5BC] hover:text-[#FAF6F0]"
                }`}
              >
                2,731 Signs
              </button>
              <button
                type="button"
                onClick={() => setModelVariant("demo_50")}
                className={`px-2.5 py-1.5 rounded-lg transition cursor-pointer ${
                  modelVariant === "demo_50"
                    ? "bg-[#D97706] text-white shadow-2xs"
                    : "text-[#ACA5BC] hover:text-[#FAF6F0]"
                }`}
              >
                Demo 50
              </button>
            </div>
          )}

          <div className="relative flex-1 sm:flex-none shrink-0 min-w-0">
            <Camera className="w-3.5 h-3.5 text-[#B6AFC6] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <select
              aria-label="Select webcam device"
              value={selectedDeviceId}
              onChange={(e) => selectCamera(e.target.value)}
              disabled={devices.length === 0}
              className="w-full sm:w-auto sm:max-w-[200px] min-w-0 max-w-full appearance-none bg-[#231D32] hover:bg-[#251F35] border border-white/10 text-[#FAF6F0] text-xs font-semibold rounded-xl pl-8 pr-8 py-2 shadow-2xs transition cursor-pointer focus:outline-none focus:ring-2 focus:ring-amber-400 disabled:opacity-60 disabled:cursor-not-allowed truncate"
            >
              {devices.length === 0 ? (
                <option value="">No camera detected</option>
              ) : (
                devices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label}
                  </option>
                ))
              )}
            </select>
            <ChevronDown className="w-3.5 h-3.5 text-[#ACA5BC] absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          </div>
        </div>
      </div>

      {/* Real Webcam Preview + Overlay Mode Switcher */}
      <div
        ref={previewWrapperRef}
        className="relative w-full aspect-[4/3] sm:aspect-[16/9] sm:min-h-[300px] rounded-[20px] overflow-hidden bg-[#120F1B] border border-white/10 min-w-0"
      >
        {/* Real Mirrored Webcam Feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          aria-label="Live ASL webcam video preview"
          className={`w-full h-full object-cover scale-x-[-1] transition-opacity duration-200 ${
            isStreaming && !cameraError ? "opacity-100" : "opacity-0"
          }`}
        />

        {/* Real-Time Hand Landmark Skeleton Overlay (Mirrored with Preview) */}
        <canvas
          ref={canvasRef}
          aria-hidden="true"
          className={`pointer-events-none absolute inset-0 w-full h-full object-cover scale-x-[-1] transition-opacity duration-200 ${
            isStreaming && !cameraError ? "opacity-100" : "opacity-0"
          }`}
        />

        {/* Subtle Viewfinder Corner Brackets */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-5 sm:inset-x-16 sm:inset-y-6"
        >
          <span className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-white/75 rounded-tl-md" />
          <span className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-white/75 rounded-tr-md" />
          <span className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-white/75 rounded-bl-md" />
          <span className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-white/75 rounded-br-md" />
        </div>

        {/* Top-Left LIVE / Status Pill */}
        <div className="absolute top-3.5 left-3.5 z-20 flex items-center gap-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/55 backdrop-blur-md border border-white/15 text-white text-[11px] font-bold tracking-wider uppercase shadow-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                isStreaming
                  ? "bg-red-500 animate-pulse"
                  : isPaused
                    ? "bg-amber-400"
                    : "bg-zinc-400"
              }`}
            />
            <span>{isStreaming ? "LIVE" : isPaused ? "PAUSED" : "OFFLINE"}</span>
          </div>

          {isStreaming && isHandDetected && (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#D97706]/90 text-white text-[11px] font-bold shadow-sm">
              <Hand className="w-3 h-3" />
              <span>Hand tracked</span>
            </div>
          )}
        </div>

        {/* Top-Right Real FPS Pill (ONLY shown when real FPS is actively measured) */}
        {isStreaming && fps !== null && fps > 0 && (
          <div className="absolute top-3.5 right-3.5 z-20 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/55 backdrop-blur-md border border-white/15 text-white text-[11px] font-bold shadow-sm">
            <SignalHigh className="w-3.5 h-3.5 text-emerald-400" />
            <span>{fps} FPS</span>
          </div>
        )}

        {/* Right-Side Recognition Mode Switcher Pills (desktop/tablet inside preview) */}
        <div
          role="radiogroup"
          aria-label="Recognition mode"
          className="hidden sm:flex absolute right-3.5 top-1/2 -translate-y-1/2 z-20 flex-col gap-2"
        >
          {MODE_OPTIONS.map((opt) => {
            const active = mode === opt.id
            return (
              <motion.button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={active}
                whileHover={prefersReduced ? undefined : { scale: 1.03 }}
                whileTap={prefersReduced ? undefined : { scale: 0.97 }}
                transition={{ duration: 0.18 }}
                onClick={() => setMode(opt.id)}
                className={`relative inline-flex items-center gap-2 px-3.5 py-2 rounded-full text-xs font-bold backdrop-blur-md border transition cursor-pointer select-none ${
                  active
                    ? "text-white border-amber-400 bg-[#3B2314]/85 shadow-[0_0_20px_rgba(255,121,0,0.45)]"
                    : "text-white/85 border-white/20 bg-black/45 hover:bg-black/65 hover:border-white/35"
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="home-live-mode-ring"
                    transition={
                      prefersReduced
                        ? { duration: 0 }
                        : { type: "spring", stiffness: 450, damping: 32 }
                    }
                    className="pointer-events-none absolute inset-0 rounded-full ring-1 ring-amber-400"
                  />
                )}
                <span className={active ? "text-amber-300" : "text-white/80"}>
                  <ModeIcon id={opt.id} />
                </span>
                <span className="pr-1">{opt.label}</span>
              </motion.button>
            )
          })}
        </div>

        {/* Bottom-Center Capture Pill (Word 2.2s Isolated Capture Mode) */}
        {mode === "Word" && isStreaming && !cameraError && !isPaused && (
          <div className="absolute bottom-3.5 left-1/2 -translate-x-1/2 z-20 flex items-center justify-center">
            {captureState === "recording" ? (
              <div className="flex items-center gap-2.5 px-4 py-2 rounded-full bg-black/75 backdrop-blur-md border border-amber-400 text-white text-xs font-bold shadow-lg min-w-[200px]">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse shrink-0" />
                <span className="shrink-0">Signing...</span>
                <div className="flex-1 h-1.5 rounded-full bg-white/20 overflow-hidden">
                  <div
                    className="h-full bg-[#D97706] transition-all duration-100"
                    style={{ width: `${captureProgress}%` }}
                  />
                </div>
                <span className="tabular-nums text-[11px]">{captureProgress}%</span>
              </div>
            ) : captureState === "analyzing" ? (
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/75 backdrop-blur-md border border-amber-400/70 text-white text-xs font-bold shadow-lg">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-amber-300" />
                <span>Analyzing sign...</span>
              </div>
            ) : (
              <motion.button
                type="button"
                whileHover={prefersReduced ? undefined : { scale: 1.03 }}
                whileTap={prefersReduced ? undefined : { scale: 0.96 }}
                onClick={startWordCapture}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-[#FF8C1A] to-[#E85D04] text-white text-xs font-bold shadow-[0_10px_22px_-6px_rgba(255,121,0,0.65)] cursor-pointer"
              >
                <Hand className="w-3.5 h-3.5" />
                <span>Capture Sign (2.2s · Space)</span>
              </motion.button>
            )}
          </div>
        )}

        {/* Camera Permission / Error / Paused Clean Overlay */}
        <AnimatePresence>
          {(cameraError || isPaused || (!isStreaming && !cameraError)) && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: prefersReduced ? 0 : 0.2 }}
              className="absolute inset-0 z-10 flex items-center justify-center bg-[#1A130F]/82 backdrop-blur-xs p-5 sm:p-6 text-center"
            >
              <div className="max-w-xs sm:max-w-sm flex flex-col items-center gap-3 text-white">
                <div className="rounded-3xl bg-[#161221]/10 border border-white/15 px-3 py-2 flex items-center justify-center">
                  <SignBridgeMascot
                    size="sm"
                    variant="wave"
                    state={isPaused ? "idle" : "listening"}
                  />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white">
                    {cameraError
                      ? cameraError.code === "permission-denied"
                        ? "Camera Permission Required"
                        : cameraError.code === "not-found"
                          ? "Webcam Not Detected"
                          : "Camera Unavailable"
                      : isPaused
                        ? "Recognition Paused"
                        : "Initializing Webcam..."}
                  </h3>
                  <p className="text-xs text-white/75 mt-1 leading-relaxed">
                    {cameraError
                      ? cameraError.message
                      : isPaused
                        ? "Resume streaming whenever you are ready to sign."
                        : "Requesting webcam access for live gesture tracking."}
                  </p>
                </div>

                {cameraError ? (
                  <motion.button
                    type="button"
                    whileHover={prefersReduced ? undefined : { scale: 1.03 }}
                    whileTap={prefersReduced ? undefined : { scale: 0.96 }}
                    onClick={retryCamera}
                    className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-gradient-to-r from-[#FF8C1A] to-[#E85D04] text-white text-xs font-bold shadow-[0_8px_18px_-6px_rgba(255,121,0,0.6)] cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Retry Camera</span>
                  </motion.button>
                ) : isPaused ? (
                  <motion.button
                    type="button"
                    whileHover={prefersReduced ? undefined : { scale: 1.03 }}
                    whileTap={prefersReduced ? undefined : { scale: 0.96 }}
                    onClick={togglePause}
                    className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-gradient-to-r from-[#FF8C1A] to-[#E85D04] text-white text-xs font-bold shadow-[0_8px_18px_-6px_rgba(255,121,0,0.6)] cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Resume Recognition</span>
                  </motion.button>
                ) : null}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Mobile Mode Switcher Strip (below video preview on screens < sm) */}
      <div
        role="radiogroup"
        aria-label="Recognition mode"
        className="flex items-center gap-1.5 sm:hidden min-w-0 w-full"
      >
        {MODE_OPTIONS.map((opt) => {
          const active = mode === opt.id
          const flexGrow =
            opt.id === "Fingerspelling"
              ? "flex-[1.3]"
              : opt.id === "Continuous"
                ? "flex-[1.1]"
                : "flex-[0.8]"
          return (
            <button
              key={opt.id}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => setMode(opt.id)}
              className={`${flexGrow} min-w-0 flex items-center justify-center gap-1 px-2 py-2 rounded-xl text-[10.5px] font-bold border transition cursor-pointer ${
                active
                  ? "bg-[#3B2314] text-white border-amber-400 shadow-xs"
                  : "bg-[#231D32] text-[#B6AFC6] border-white/10"
              }`}
            >
              <span className={active ? "text-amber-300 shrink-0" : "text-[#ACA5BC] shrink-0"}>
                <ModeIcon id={opt.id} />
              </span>
              <span className="truncate">{opt.label}</span>
            </button>
          )
        })}
      </div>

      {/* Translation Output Bar */}
      <div className="flex flex-wrap sm:flex-nowrap items-center justify-between gap-2.5 sm:gap-3 rounded-2xl bg-[#251F35] border border-white/10 px-3 sm:px-3.5 py-2.5 min-w-0 w-full">
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0 basis-full sm:basis-auto sm:flex-1">
          <div className="w-9 h-9 rounded-xl bg-[#1D1829] border border-[#F1E0D2] text-[#FAF6F0] flex items-center justify-center shrink-0">
            <Languages className="w-4 h-4" />
          </div>

          <div className="min-w-0 flex-1 flex items-center gap-2.5">
            {recognizedText ? (
              <p
                className="text-sm sm:text-[15px] font-bold text-[#FAF6F0] truncate"
                aria-live="polite"
              >
                {recognizedText}
              </p>
            ) : (
              <p className="text-sm italic text-[#ACA5BC] truncate">
                Start signing...
              </p>
            )}

            {activePrediction && recognizedText && (
              <span className="hidden md:inline-flex items-center px-2 py-0.5 rounded-full bg-amber-400/18 text-amber-300 text-[10px] font-bold shrink-0">
                {activePrediction.confidence.toFixed(1)}%
              </span>
            )}
          </div>
        </div>

        {/* Output Action Buttons */}
        <div className="flex items-center gap-2 shrink-0 ml-auto">
          {recognizedText && (
            <motion.button
              type="button"
              whileHover={prefersReduced ? undefined : { scale: 1.06 }}
              whileTap={prefersReduced ? undefined : { scale: 0.93 }}
              onClick={clearOutput}
              title="Clear and save session"
              aria-label="Clear recognized text"
              className="w-9 h-9 rounded-full bg-[#1D1829] border border-white/10 flex items-center justify-center text-[#ACA5BC] hover:text-[#FAF6F0] hover:border-amber-400/40 transition cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
            </motion.button>
          )}

          <motion.button
            type="button"
            whileHover={prefersReduced ? undefined : { scale: 1.06 }}
            whileTap={prefersReduced ? undefined : { scale: 0.93 }}
            onClick={handleCopy}
            title="Copy recognized text"
            aria-label="Copy recognized text"
            className="w-9 h-9 rounded-full bg-[#1D1829] border border-white/10 flex items-center justify-center text-[#FAF6F0] hover:bg-amber-400/20 hover:border-amber-400/40 transition cursor-pointer"
          >
            {copied ? (
              <Check className="w-4 h-4 text-[#22B66B]" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </motion.button>

          <motion.button
            type="button"
            whileHover={prefersReduced ? undefined : { scale: 1.06 }}
            whileTap={prefersReduced ? undefined : { scale: 0.93 }}
            onClick={togglePause}
            title={isPaused ? "Resume recognition" : "Pause recognition"}
            aria-label={isPaused ? "Resume recognition" : "Pause recognition"}
            className="w-9 h-9 rounded-full bg-[#1D1829] border border-white/10 flex items-center justify-center text-[#FAF6F0] hover:bg-amber-400/20 hover:border-amber-400/40 transition cursor-pointer"
          >
            {isPaused || !isStreaming ? (
              <Play className="w-4 h-4 fill-current text-amber-400" />
            ) : (
              <Pause className="w-4 h-4" />
            )}
          </motion.button>

          <motion.button
            type="button"
            whileHover={prefersReduced ? undefined : { scale: 1.06 }}
            whileTap={prefersReduced ? undefined : { scale: 0.93 }}
            onClick={speakOutput}
            title="Speak recognized text with Kokoro TTS"
            aria-label="Speak recognized text"
            className={`w-9 h-9 rounded-full border flex items-center justify-center transition cursor-pointer ${
              isSpeaking
                ? "bg-[#D97706] border-amber-500 text-white"
                : "bg-[#1D1829] border-white/10 text-[#FAF6F0] hover:bg-amber-400/20 hover:border-amber-400/40"
            }`}
          >
            <Volume2 className="w-4 h-4" />
          </motion.button>

          {onExpandStudio && (
            <motion.button
              type="button"
              whileHover={prefersReduced ? undefined : { scale: 1.03 }}
              whileTap={prefersReduced ? undefined : { scale: 0.96 }}
              onClick={onExpandStudio}
              className="hidden sm:inline-flex items-center gap-1.5 px-3 py-2 rounded-full bg-[#251F35] hover:bg-amber-400/18 border border-amber-400/30 text-[11px] font-bold text-amber-300 transition cursor-pointer"
            >
              <span>Studio</span>
            </motion.button>
          )}

          <motion.button
            type="button"
            whileHover={prefersReduced ? undefined : { scale: 1.06 }}
            whileTap={prefersReduced ? undefined : { scale: 0.93 }}
            onClick={toggleFullscreen}
            title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen preview"}
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            className="w-9 h-9 rounded-full bg-[#1D1829] border border-white/10 flex items-center justify-center text-[#FAF6F0] hover:bg-amber-400/20 hover:border-amber-400/40 transition cursor-pointer"
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </motion.button>
        </div>
      </div>

      {/* Live Model Top-3 Candidates Strip (clickable to refine last word/char) */}
      {isStreaming && topCandidates.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 px-1 min-w-0">
          <span className="text-[11px] font-bold uppercase tracking-wider text-[#ACA5BC] mr-1">
            Top Predictions:
          </span>
          {topCandidates.map((cand, idx) => (
            <button
              key={`${cand.gloss}-${idx}`}
              type="button"
              onClick={() => selectCandidate(cand.gloss, cand.confidence)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold border transition cursor-pointer ${
                idx === 0
                  ? "bg-[#251F35] border-amber-400/50 text-[#D95300] hover:bg-amber-400/18"
                  : "bg-[#1D1829] border-white/10 text-[#B6AFC6] hover:border-amber-400/40 hover:text-[#FAF6F0]"
              }`}
            >
              <span>{cand.gloss}</span>
              <span className="opacity-75 tabular-nums">{cand.confidence.toFixed(1)}%</span>
            </button>
          ))}
        </div>
      )}
    </motion.section>
  )
}

