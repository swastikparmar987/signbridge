"use client"

import * as React from "react"
import {
  ArrowLeft,
  Camera,
  Check,
  Copy,
  Delete,
  FlipHorizontal2,
  HelpCircle,
  Pause,
  Play,
  RotateCcw,
  Settings,
  Volume2,
  VolumeX,
} from "lucide-react"
import { useSignBridgeStore, type RecognitionMode, type SignBridgePreferences } from "@/lib/signbridge-store"
import { FeedbackLayer } from "@/components/feedback/feedback-layer"
import { emitFeedback } from "@/lib/feedback/bus"
import { playFor, unlockSound } from "@/lib/feedback/sound"
import { SignBridgeLogoMark } from "@/components/brand/signbridge-bridge-logo"
import { useAslRecognition } from "@/lib/use-asl-recognition"
import { InteractiveHandGuide } from "./interactive-hand-guide"

interface CommunicationCanvasProps {
  initialMode: RecognitionMode
  onBackToMenu: () => void
  onOpenWindow: (win: import("@/lib/signbridge-store").StudioWindowId) => void
  tourOpen?: boolean
  forceSettingsPopoverOpen?: boolean | null
  onStartTour?: () => void
}

const MODE_TABS: Array<{ mode: RecognitionMode; label: string }> = [
  { mode: "Word", label: "Words" },
  { mode: "Continuous", label: "Continuous" },
  { mode: "Fingerspelling", label: "Fingerspelling" },
]

export function CommunicationCanvas({
  initialMode,
  onBackToMenu,
  onOpenWindow,
  tourOpen = false,
  forceSettingsPopoverOpen = null,
  onStartTour,
}: CommunicationCanvasProps) {
  const {
    videoRef,
    canvasRef,
    mode,
    setMode,
    devices,
    selectedDeviceId,
    selectCamera,
    isStreaming,
    isPaused,
    cameraError,
    fps,
    isHandDetected,
    isStrokeActive,
    recognizedText,
    glosses,
    activePrediction,
    topCandidates,
    captureState,
    captureProgress,
    startWordCapture,
    selectCandidate,
    appendFingerspellingSpace,
    backspaceFingerspelling,
    isSpeaking,
    retryCamera,
    togglePause,
    clearOutput,
    copyOutput,
    speakOutput,
  } = useAslRecognition(initialMode)

  const { preferences, updatePreferences } = useSignBridgeStore()
  const captionSizeClass =
    preferences.captionSize === "xl"
      ? "text-lg sm:text-2xl"
      : preferences.captionSize === "md"
        ? "text-sm sm:text-base"
        : "text-base sm:text-lg"
  const setPref = (patch: Partial<SignBridgePreferences>) => {
    updatePreferences(patch)
    emitFeedback({ type: "ui", kind: "toggle" })
  }

  const [mirrored, setMirrored] = React.useState(true)
  const [settingsOpen, setSettingsOpen] = React.useState(false)
  const effectiveSettingsOpen = forceSettingsPopoverOpen ?? settingsOpen
  const [copied, setCopied] = React.useState(false)
  const [autoSpeak, setAutoSpeak] = React.useState(true)
  const [backendHealth, setBackendHealth] = React.useState<{
    status: "checking" | "online" | "offline"
    hardwareDevice?: string
  }>({ status: "checking" })
  const settingsPanelRef = React.useRef<HTMLDivElement | null>(null)
  const lastAutoSpokenRef = React.useRef<string>("")

  // Poll /api/health lightly to confirm SignBridge backend connectivity
  React.useEffect(() => {
    let mounted = true
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/health", { cache: "no-store" })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (!mounted) return
        if (data?.status === "healthy" || data?.num_classes) {
          setBackendHealth({
            status: "online",
            hardwareDevice: typeof data.hardware_device === "string" ? data.hardware_device : undefined,
          })
        } else {
          setBackendHealth({ status: "offline" })
        }
      } catch {
        if (mounted) setBackendHealth({ status: "offline" })
      }
    }
    void checkHealth()
    const timer = window.setInterval(() => void checkHealth(), 15000)
    return () => {
      mounted = false
      window.clearInterval(timer)
    }
  }, [])

  // Sync mode chosen in the unified menu upon entering the live canvas
  React.useEffect(() => {
    setMode(initialMode)
  }, [initialMode, setMode])

  // Global keyboard shortcuts: Spacebar switches modes anytime; Enter triggers Word capture
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (tourOpen) return
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return

      if (e.code === "Space") {
        e.preventDefault()
        const idx = MODE_TABS.findIndex((t) => t.mode === mode)
        const next = MODE_TABS[(idx + 1) % MODE_TABS.length]
        setMode(next.mode)
      } else if (e.code === "Enter" && mode === "Word" && captureState === "idle") {
        e.preventDefault()
        startWordCapture()
      } else if (e.code === "Backspace" && mode === "Fingerspelling") {
        e.preventDefault()
        backspaceFingerspelling()
      } else if (e.code === "Escape" && effectiveSettingsOpen) {
        setSettingsOpen(false)
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [mode, setMode, captureState, startWordCapture, backspaceFingerspelling, effectiveSettingsOpen, tourOpen])

  // Close quick camera/model popover on outside click
  React.useEffect(() => {
    if (!effectiveSettingsOpen) return
    const onDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null
      if (target?.closest?.('[data-tour-overlay="true"]')) return
      if (settingsPanelRef.current && !settingsPanelRef.current.contains(target)) {
        setSettingsOpen(false)
      }
    }
    window.addEventListener("mousedown", onDown)
    return () => window.removeEventListener("mousedown", onDown)
  }, [effectiveSettingsOpen])

  const handleCopy = async () => {
    await copyOutput()
    setCopied(true)
    setTimeout(() => setCopied(false), 1400)
  }

  const placeholderText =
    mode === "Word"
      ? "Press Enter or Sign Word to recognize an isolated sign..."
      : mode === "Fingerspelling"
        ? "Sign letters naturally to spell words..."
        : "Sign continuously — your words translate here..."

  const displaySentence = recognizedText.trim()
  const recentGlosses = glosses.slice(-6)

  // Smart debounced Auto-Speak TTS readout for completed words / translated sentences
  React.useEffect(() => {
    if (!displaySentence) {
      lastAutoSpokenRef.current = ""
      return
    }
    if (!autoSpeak) return
    if (displaySentence === lastAutoSpokenRef.current) return

    const delayMs = mode === "Word" ? 250 : mode === "Fingerspelling" ? 1800 : 1250
    const timer = window.setTimeout(() => {
      const latest = recognizedText.trim()
      if (!latest || latest === lastAutoSpokenRef.current) return
      lastAutoSpokenRef.current = latest
      void speakOutput(latest)
    }, delayMs)

    return () => window.clearTimeout(timer)
  }, [displaySentence, recognizedText, autoSpeak, mode, speakOutput])

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#15121E] text-[#FAF6F0] select-none">
      {/* Main 85%+ Open Webcam Studio Viewport */}
      <div className="absolute inset-2.5 sm:inset-4 rounded-[26px] overflow-hidden bg-[#1C1829] ring-1 ring-white/10 shadow-[0_32px_90px_-16px_rgba(0,0,0,0.85)]">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={[
            "h-full w-full object-cover transition-opacity duration-300",
            mirrored ? "-scale-x-100" : "",
            isStreaming ? "opacity-100" : "opacity-0",
          ].join(" ")}
        />

        {/* Luminous amber hand-landmark canvas */}
        <canvas
          ref={canvasRef}
          className={[
            "pointer-events-none absolute inset-0 h-full w-full object-cover",
            mirrored ? "-scale-x-100" : "",
          ].join(" ")}
        />

        {/* Subtle dark edge scrim so floating chrome stays legible without boxing the signer */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "linear-gradient(to bottom, rgba(21,18,30,0.58) 0%, rgba(21,18,30,0) 22%, rgba(21,18,30,0) 68%, rgba(21,18,30,0.74) 100%)",
          }}
        />

        {/* Deaf-first visual feedback: edge alerts, recognition bursts, speaking indicator */}
        <FeedbackLayer
          mirrored={mirrored}
          isStreaming={isStreaming}
          isHandDetected={isHandDetected}
          isStrokeActive={isStrokeActive || captureState === "recording"}
          isSpeaking={isSpeaking}
          hasCameraError={Boolean(cameraError)}
        />

        {/* Camera initializing / permission overlay */}
        {!isStreaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-[#191525]/92">
            {cameraError ? (
              <div className="max-w-sm rounded-2xl border border-white/12 bg-[#231E33]/90 p-6 backdrop-blur-xl shadow-2xl">
                <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-amber-400/15 text-amber-300">
                  <Camera className="h-5 w-5" />
                </div>
                <h2 className="font-display text-lg font-semibold text-white">
                  {cameraError.code === "not-found" ? "No Camera Detected" : "Camera Permission Needed"}
                </h2>
                <p className="mt-1.5 text-xs leading-relaxed text-[#B3ACC3]">
                  {cameraError.message}
                </p>
                <div className="mt-5 flex items-center justify-center gap-2.5">
                  <button
                    type="button"
                    onClick={retryCamera}
                    className="rounded-full bg-amber-400 px-5 py-2 text-xs font-bold text-[#1A130D] hover:brightness-105"
                  >
                    Enable Camera
                  </button>
                  <button
                    type="button"
                    onClick={onBackToMenu}
                    className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-xs font-medium text-white/80 hover:bg-white/10"
                  >
                    Back to Menu
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="h-9 w-9 animate-spin rounded-full border-2 border-amber-400/30 border-t-amber-400" />
                <span className="text-xs font-medium tracking-wide text-[#BEB8CE]">
                  Preparing live signing canvas...
                </span>
              </div>
            )}
          </div>
        )}

        {/* Word capture countdown ring badge when recording */}
        {mode === "Word" && captureState !== "idle" && (
          <div className="pointer-events-none absolute top-20 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2.5 rounded-full border border-amber-400/40 bg-[#1B1626]/85 px-4 py-1.5 backdrop-blur-md">
            <span className="h-2 w-2 animate-ping rounded-full bg-amber-400" />
            <span className="text-xs font-semibold text-amber-200">
              {captureState === "recording"
                ? `Signing window · ${Math.round(captureProgress)}%`
                : "Recognizing sign..."}
            </span>
          </div>
        )}
      </div>

      {/* Top Floating Glass Chrome */}
      <header className="pointer-events-none fixed top-0 inset-x-0 z-30 flex items-center justify-between px-6 sm:px-9 pt-6 sm:pt-7">
        {/* Left: Back to Menu + Logo */}
        <div className="pointer-events-auto flex items-center gap-2.5">
          <button
            type="button"
            data-tour="studio-menu"
            onClick={onBackToMenu}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/12 bg-[#1B1726]/75 px-3.5 py-2 text-xs font-medium text-white/90 backdrop-blur-xl transition hover:border-white/25 hover:bg-[#252034]"
          >
            <ArrowLeft className="h-3.5 w-3.5 text-amber-300" />
            <span>Menu</span>
          </button>

          <div className="hidden md:flex items-center gap-2 pl-1.5 text-sm font-display font-bold text-white/95 drop-shadow">
            <SignBridgeLogoMark size="xs" priority />
            <span>SignBridge</span>
          </div>
        </div>

        {/* Center: Clean Pill Mode Switcher */}
        <nav
          data-tour="studio-mode-switcher"
          aria-label="Signing mode switcher"
          className="pointer-events-auto flex items-center rounded-full border border-white/12 bg-[#1A1625]/78 p-1 backdrop-blur-xl shadow-lg"
        >
          {MODE_TABS.map((tab) => {
            const active = mode === tab.mode
            return (
              <button
                key={tab.mode}
                type="button"
                aria-pressed={active}
                onClick={() => setMode(tab.mode)}
                className={[
                  "rounded-full px-3.5 sm:px-4 py-1.5 text-xs font-semibold transition-all duration-200",
                  active
                    ? "bg-gradient-to-b from-[#FF8A3D] to-[#FB6A18] text-[#17110B] shadow-[0_4px_16px_rgba(251,106,24,0.4)]"
                    : "text-[#BEB8CE] hover:text-white",
                ].join(" ")}
              >
                {tab.label}
              </button>
            )
          })}
        </nav>

        {/* Right: Quiet Circular Controls & Settings Popover */}
        <div className="pointer-events-auto relative flex items-center gap-2" ref={settingsPanelRef}>
          {/* Backend AI health badge */}
          <div
            title={
              backendHealth.status === "online"
                ? `SignBridge neural backend connected (${backendHealth.hardwareDevice || "ready"})`
                : backendHealth.status === "checking"
                  ? "Checking neural backend connection..."
                  : "Backend API unreachable on /api/health"
            }
            className={[
              "hidden xl:flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] font-medium backdrop-blur-xl transition-colors",
              backendHealth.status === "online"
                ? "border-emerald-400/25 bg-[#162922]/80 text-emerald-200"
                : backendHealth.status === "offline"
                  ? "border-rose-400/35 bg-rose-950/75 text-rose-200"
                  : "border-white/10 bg-[#1B1726]/70 text-white/65",
            ].join(" ")}
          >
            <span
              className={[
                "h-2 w-2 rounded-full",
                backendHealth.status === "online"
                  ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.9)]"
                  : backendHealth.status === "offline"
                    ? "bg-rose-400"
                    : "animate-pulse bg-amber-300",
              ].join(" ")}
            />
            <span>
              {backendHealth.status === "online"
                ? `AI Ready${backendHealth.hardwareDevice ? ` · ${backendHealth.hardwareDevice}` : ""}`
                : backendHealth.status === "offline"
                  ? "AI Offline"
                  : "Connecting..."}
            </span>
          </div>

          {/* Subtle hand tracking status dot pill */}
          <div
            data-tour="studio-tracking"
            className="hidden lg:flex items-center gap-1.5 rounded-full border border-white/10 bg-[#1B1726]/70 px-3 py-1.5 text-[11px] font-medium text-white/75 backdrop-blur-xl"
          >
            <span
              className={[
                "h-2 w-2 rounded-full transition-colors",
                isHandDetected ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.9)]" : "bg-white/30",
              ].join(" ")}
            />
            <span>
              {isHandDetected
                ? `Tracking${fps !== null && fps > 0 ? ` · ${fps}fps` : ""}`
                : "Hands ready"}
            </span>
          </div>

          <button
            type="button"
            data-tour="studio-auto-tts"
            aria-pressed={autoSpeak}
            aria-label={autoSpeak ? "Auto-speak enabled" : "Auto-speak muted"}
            title={autoSpeak ? "Auto-speak TTS: On" : "Auto-speak TTS: Off"}
            onClick={() => setAutoSpeak((prev) => !prev)}
            className={[
              "hidden sm:inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] font-semibold backdrop-blur-xl transition",
              autoSpeak
                ? "border-amber-400/40 bg-amber-400/15 text-amber-200 hover:bg-amber-400/25"
                : "border-white/12 bg-[#1B1726]/75 text-white/60 hover:border-white/25 hover:text-white",
            ].join(" ")}
          >
            {autoSpeak ? <Volume2 className="h-3.5 w-3.5 text-amber-300" /> : <VolumeX className="h-3.5 w-3.5" />}
            <span>Auto TTS</span>
          </button>

          <button
            type="button"
            data-tour="studio-pause"
            aria-label={isPaused ? "Resume camera" : "Pause camera"}
            title={isPaused ? "Resume camera" : "Pause camera"}
            onClick={togglePause}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-white/12 bg-[#1B1726]/75 text-white/85 backdrop-blur-xl transition hover:border-white/25 hover:bg-[#252034]"
          >
            {isPaused ? <Play className="h-4 w-4 text-amber-300" /> : <Pause className="h-4 w-4" />}
          </button>

          <button
            type="button"
            data-tour="studio-speak"
            aria-label="Speak translation aloud"
            title="Speak aloud"
            onClick={() => speakOutput()}
            className={[
              "flex h-9 w-9 items-center justify-center rounded-full border backdrop-blur-xl transition",
              isSpeaking
                ? "border-amber-400 bg-amber-400/20 text-amber-300"
                : "border-white/12 bg-[#1B1726]/75 text-white/85 hover:border-white/25 hover:bg-[#252034]",
            ].join(" ")}
          >
            <Volume2 className="h-4 w-4" />
          </button>

          {onStartTour && (
            <button
              type="button"
              data-tour="studio-tour"
              aria-label="Take studio tour"
              title="Take studio tour"
              onClick={onStartTour}
              className="flex h-9 w-9 items-center justify-center rounded-full border border-white/12 bg-[#1B1726]/75 text-amber-300 backdrop-blur-xl transition hover:border-amber-400/40 hover:bg-amber-400/15 cursor-pointer"
            >
              <HelpCircle className="h-4 w-4" />
            </button>
          )}

          <button
            type="button"
            data-tour="studio-settings"
            aria-label="Camera and recognition settings"
            onClick={() => setSettingsOpen((prev) => !prev)}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-white/12 bg-[#1B1726]/75 text-white/85 backdrop-blur-xl transition hover:border-white/25 hover:bg-[#252034]"
          >
            <Settings className="h-4 w-4" />
          </button>

          {effectiveSettingsOpen && (
            <div className="absolute right-0 top-12 w-72 max-h-[calc(100vh-7rem)] overflow-y-auto rounded-2xl border border-white/15 bg-[#1D182A]/95 p-3.5 text-xs shadow-2xl backdrop-blur-xl z-50 space-y-3">
              <div data-tour="studio-camera-select">
                <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-white/50">
                  Camera Source
                </div>
                <select
                  value={selectedDeviceId}
                  onChange={(e) => selectCamera(e.target.value)}
                  className="w-full rounded-xl border border-white/12 bg-white/5 px-2.5 py-2 text-xs text-white outline-none focus:border-amber-400"
                >
                  {devices.length === 0 ? (
                    <option value="">Default Webcam</option>
                  ) : (
                    devices.map((dev, i) => (
                      <option key={dev.deviceId || i} value={dev.deviceId} className="bg-[#1D182A]">
                        {dev.label || `Camera ${i + 1}`}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <button
                type="button"
                data-tour="studio-mirror"
                onClick={() => setMirrored((m) => !m)}
                className="flex w-full items-center justify-between rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-white/85 hover:bg-white/10"
              >
                <span className="inline-flex items-center gap-2">
                  <FlipHorizontal2 className="h-3.5 w-3.5 text-amber-300" />
                  Mirror Camera Preview
                </span>
                <span className="text-[11px] font-semibold text-amber-300">
                  {mirrored ? "On" : "Off"}
                </span>
              </button>

              <div className="space-y-1.5 border-t border-white/10 pt-3">
                <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-white/50">
                  Feedback
                </div>
                {(
                  [
                    { key: "visualAlerts", label: "Visual alerts", hint: "Glowing edges & bursts" },
                    { key: "haptics", label: "Vibration", hint: "On supported phones" },
                    { key: "soundEffects", label: "Sound effects", hint: "For hearing companions" },
                  ] as const
                ).map((opt) => {
                  const on = preferences[opt.key]
                  return (
                    <button
                      key={opt.key}
                      type="button"
                      role="switch"
                      aria-checked={on}
                      onClick={() => {
                        const next = !on
                        if (opt.key === "soundEffects" && next) {
                          unlockSound()
                          playFor({ type: "ui", kind: "start" })
                        }
                        if (opt.key === "haptics" && next && typeof navigator !== "undefined") {
                          try {
                            navigator.vibrate?.([30, 40, 30])
                          } catch {
                            // ignore
                          }
                        }
                        setPref({ [opt.key]: next } as Partial<SignBridgePreferences>)
                      }}
                      className="flex w-full items-center justify-between rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-left text-white/85 hover:bg-white/10"
                    >
                      <span>
                        <span className="block">{opt.label}</span>
                        <span className="block text-[10px] text-white/45">{opt.hint}</span>
                      </span>
                      <span
                        aria-hidden="true"
                        className={[
                          "relative h-5 w-9 shrink-0 rounded-full transition-colors",
                          on ? "bg-amber-500" : "bg-white/15",
                        ].join(" ")}
                      >
                        <span
                          className={[
                            "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all",
                            on ? "left-[18px]" : "left-0.5",
                          ].join(" ")}
                        />
                      </span>
                    </button>
                  )
                })}

                <div className="pt-1.5">
                  <div className="mb-1 text-[10px] text-white/50">Effects</div>
                  <div className="grid grid-cols-3 gap-1 rounded-xl bg-white/5 p-1" role="radiogroup" aria-label="Effects level">
                    {(["full", "subtle", "off"] as const).map((lv) => (
                      <button
                        key={lv}
                        type="button"
                        role="radio"
                        aria-checked={preferences.effectsLevel === lv}
                        onClick={() => setPref({ effectsLevel: lv })}
                        className={[
                          "rounded-lg py-1.5 text-[11px] font-semibold capitalize transition",
                          preferences.effectsLevel === lv ? "bg-amber-500 text-[#1B1726]" : "text-white/70 hover:bg-white/10",
                        ].join(" ")}
                      >
                        {lv}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="pt-1.5">
                  <div className="mb-1 text-[10px] text-white/50">Caption size</div>
                  <div className="grid grid-cols-3 gap-1 rounded-xl bg-white/5 p-1" role="radiogroup" aria-label="Caption size">
                    {(
                      [
                        ["md", "A", "text-[11px]"],
                        ["lg", "A", "text-[13px]"],
                        ["xl", "A", "text-[16px]"],
                      ] as const
                    ).map(([size, glyph, cls]) => (
                      <button
                        key={size}
                        type="button"
                        role="radio"
                        aria-checked={preferences.captionSize === size}
                        aria-label={`Caption size ${size === "md" ? "medium" : size === "lg" ? "large" : "extra large"}`}
                        onClick={() => setPref({ captionSize: size })}
                        className={[
                          "rounded-lg py-1 font-bold transition",
                          cls,
                          preferences.captionSize === size ? "bg-amber-500 text-[#1B1726]" : "text-white/70 hover:bg-white/10",
                        ].join(" ")}
                      >
                        {glyph}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <button
                type="button"
                data-tour="studio-train-custom"
                onClick={() => {
                  setSettingsOpen(false)
                  onOpenWindow("custom-learner")
                }}
                className="w-full rounded-xl border border-amber-400/25 bg-amber-400/10 px-3 py-2 text-left font-medium text-amber-200 hover:bg-amber-400/20"
              >
                Train Custom Signs →
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Bottom Floating Sentence Cartouche + Gloss Chips */}
      <footer className="pointer-events-none fixed inset-x-0 bottom-5 sm:bottom-7 z-30 flex flex-col items-center px-4">
        <div className="pointer-events-auto flex w-full max-w-[540px] lg:max-w-[660px] flex-col items-center gap-2.5">
          {/* Live streaming sign/letter hypothesis + gloss chips + top candidates */}
          {(activePrediction ||
            isStrokeActive ||
            recentGlosses.length > 0 ||
            (mode === "Word" && topCandidates.length > 0)) && (
            <div className="flex flex-wrap items-center justify-center gap-1.5">
              {activePrediction ? (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/50 bg-[#1D1628]/88 px-3 py-0.5 text-[11px] font-semibold text-amber-200 backdrop-blur-md shadow-[0_4px_18px_rgba(251,106,24,0.22)]">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
                  <span>
                    {mode === "Fingerspelling" ? "Letter" : "Seeing"}: {activePrediction.label} ·{" "}
                    {Math.round(activePrediction.confidence)}%
                  </span>
                </span>
              ) : (
                isStrokeActive &&
                mode === "Continuous" && (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/40 bg-[#1D1628]/85 px-3 py-0.5 text-[11px] font-medium text-amber-200/90 backdrop-blur-md">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
                    <span>Reading sign motion…</span>
                  </span>
                )
              )}

              {recentGlosses.map((g, idx) => (
                <span
                  key={`${g}-${idx}`}
                  className="rounded-lg border border-white/12 bg-[#1D1829]/75 px-2.5 py-0.5 font-mono text-[11px] font-semibold tracking-wider text-white/80 backdrop-blur-md uppercase"
                >
                  {g}
                </span>
              ))}

              {mode === "Word" &&
                topCandidates.slice(0, 3).map((cand) => (
                  <button
                    key={cand.gloss}
                    type="button"
                    onClick={() => selectCandidate(cand.gloss, cand.confidence)}
                    className="rounded-lg border border-amber-400/35 bg-amber-400/15 px-2.5 py-0.5 text-[11px] font-semibold text-amber-200 backdrop-blur-md transition hover:bg-amber-400/25"
                  >
                    {cand.gloss} · {Math.round(cand.confidence)}%
                  </button>
                ))}
            </div>
          )}

          {/* High-Contrast Cream Sentence Caption Card */}
          <div
            data-tour="studio-caption-bar"
            className="relative flex w-full items-center gap-3 rounded-2xl bg-[#FAF6F0] px-4 sm:px-5 py-3.5 text-[#18141C] shadow-[0_24px_65px_-12px_rgba(0,0,0,0.8)] ring-1 ring-white/50"
          >
            {/* Subtle recording progress bar across top of card */}
            {mode === "Word" && captureState === "recording" && (
              <div className="absolute top-0 left-3 right-3 h-[3px] overflow-hidden rounded-full bg-amber-900/10">
                <div
                  className="h-full bg-amber-500 transition-all duration-100"
                  style={{ width: `${Math.min(100, captureProgress)}%` }}
                />
              </div>
            )}

            <span
              className={[
                "h-2.5 w-2.5 shrink-0 rounded-full transition-all",
                captureState === "recording" || (mode === "Continuous" && isStrokeActive)
                  ? "animate-pulse bg-amber-500 ring-4 ring-amber-400/30"
                  : isHandDetected
                    ? "bg-amber-500"
                    : "bg-amber-400/60",
              ].join(" ")}
            />

            <p
              role="status"
              aria-live="polite"
              className={[
                "flex-1 font-medium leading-snug break-words transition-[font-size]",
                captionSizeClass,
                displaySentence
                  ? "text-[#16121A]"
                  : mode === "Continuous" && activePrediction
                    ? "text-[#5C4B3B]"
                    : "text-[#8A8294]",
              ].join(" ")}
            >
              {displaySentence ||
                (mode === "Continuous" && activePrediction
                  ? `Signing: ${activePrediction.label} · ${Math.round(activePrediction.confidence)}% (pause briefly to translate)`
                  : mode === "Continuous" && isStrokeActive
                    ? "Reading sign motion — pause ~1s to translate..."
                    : placeholderText)}
            </p>

            {/* Word-mode trigger pill */}
            {mode === "Word" && (
              <button
                type="button"
                disabled={captureState !== "idle"}
                onClick={startWordCapture}
                className={[
                  "shrink-0 rounded-full px-3.5 py-1.5 text-xs font-bold transition",
                  captureState === "idle"
                    ? "bg-[#1A141E] text-[#FAF6F0] hover:bg-[#2D2435]"
                    : "bg-amber-200 text-amber-900 opacity-80",
                ].join(" ")}
              >
                {captureState === "recording" ? "Signing..." : captureState === "analyzing" ? "Reading..." : "Sign Word · ↵"}
              </button>
            )}

            {/* Fingerspelling Space / Backspace quick actions */}
            {mode === "Fingerspelling" && (
              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  onClick={appendFingerspellingSpace}
                  title="Insert space between spelled words"
                  className="rounded-full border border-[#1A141E]/15 bg-[#EFE8DF] px-2.5 py-1 text-[11px] font-semibold text-[#2A2230] transition hover:bg-[#E4DACD]"
                >
                  Space
                </button>
                <button
                  type="button"
                  onClick={backspaceFingerspelling}
                  aria-label="Backspace letter"
                  title="Delete last letter (Backspace)"
                  className="flex h-7 w-7 items-center justify-center rounded-full border border-[#1A141E]/15 bg-[#EFE8DF] text-[#2A2230] transition hover:bg-[#E4DACD]"
                >
                  <Delete className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            {/* Utility icons: Clear, Copy, Speak */}
            <div className="flex items-center gap-1 shrink-0">
              {displaySentence && (
                <>
                  <button
                    type="button"
                    aria-label="Clear recognized text"
                    title="Clear"
                    onClick={clearOutput}
                    className="flex h-8 w-8 items-center justify-center rounded-full text-[#6B6475] transition hover:bg-black/5 hover:text-[#18141C]"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    aria-label="Copy recognized text"
                    title="Copy"
                    onClick={handleCopy}
                    className="flex h-8 w-8 items-center justify-center rounded-full text-[#6B6475] transition hover:bg-black/5 hover:text-[#18141C]"
                  >
                    {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </>
              )}

              <button
                type="button"
                aria-label="Read sentence aloud"
                title="Speak aloud"
                onClick={() => speakOutput()}
                className={[
                  "flex h-9 w-9 items-center justify-center rounded-full transition",
                  isSpeaking
                    ? "bg-amber-500 text-white shadow-md"
                    : "bg-amber-500/15 text-amber-700 hover:bg-amber-500/25",
                ].join(" ")}
              >
                <Volume2 className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </footer>

      {/* Corner Interactive 3D Hand Guide */}
      <div
        data-tour="studio-hand-guide"
        className="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 z-40 hidden md:block"
      >
        <InteractiveHandGuide mode={mode} activeLabel={activePrediction?.label ?? null} />
      </div>
    </div>
  )
}
