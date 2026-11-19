"use client"

import * as React from "react"
import { ArrowRight, BookOpen, Command, FolderOpen, HelpCircle, Settings, Sparkles } from "lucide-react"
import type { RecognitionMode, StudioWindowId } from "@/lib/signbridge-store"
import { SignBridgeLogoMark } from "@/components/brand/signbridge-bridge-logo"
import { emitFeedback } from "@/lib/feedback/bus"
import { InteractiveMenuHand, type ModePointerRef } from "./menu-3d-hands"

interface UnifiedModeMenuProps {
  selectedMode: RecognitionMode
  onSelectMode: (mode: RecognitionMode) => void
  onStartCommunicating: (mode: RecognitionMode) => void
  onOpenWindow: (win: StudioWindowId) => void
  onOpenCommandPalette: () => void
  tourOpen?: boolean
  forceToolsMenuOpen?: boolean | null
  onStartTour?: () => void
}

interface ModeCardMeta {
  mode: RecognitionMode
  title: string
  subtitle: string
}

const MODE_CARDS: ModeCardMeta[] = [
  {
    mode: "Word",
    title: "Words",
    subtitle: "Single ASL vocabulary signs",
  },
  {
    mode: "Continuous",
    title: "Continuous",
    subtitle: "Natural multi-sign sentences",
  },
  {
    mode: "Fingerspelling",
    title: "Fingerspelling",
    subtitle: "Spell words letter by letter",
  },
]

const MODE_ORDER: RecognitionMode[] = ["Word", "Continuous", "Fingerspelling"]

export function UnifiedModeMenu({
  selectedMode,
  onSelectMode,
  onStartCommunicating,
  onOpenWindow,
  onOpenCommandPalette,
  tourOpen = false,
  forceToolsMenuOpen = null,
  onStartTour,
}: UnifiedModeMenuProps) {
  const [hoveredMode, setHoveredMode] = React.useState<RecognitionMode | null>(null)
  const [cameraReadyLabel, setCameraReadyLabel] = React.useState<"Checking..." | "Camera Ready" | "Grant Camera">("Camera Ready")
  const [menuOpen, setMenuOpen] = React.useState(false)
  const effectiveMenuOpen = forceToolsMenuOpen ?? menuOpen

  const settingsBtnRef = React.useRef<HTMLDivElement | null>(null)

  const pointerRef = React.useRef<Record<RecognitionMode, ModePointerRef>>({
    Word: { x: 0, y: 0 },
    Continuous: { x: 0, y: 0 },
    Fingerspelling: { x: 0, y: 0 },
  })

  // Lightweight non-intrusive check for camera availability without activating webcam stream on menu screen
  React.useEffect(() => {
    let active = true
    const checkDevices = async () => {
      try {
        if (!navigator.mediaDevices?.enumerateDevices) return
        const devs = await navigator.mediaDevices.enumerateDevices()
        const hasCam = devs.some((d) => d.kind === "videoinput")
        if (!active) return
        setCameraReadyLabel(hasCam ? "Camera Ready" : "Grant Camera")
      } catch {
        if (active) setCameraReadyLabel("Camera Ready")
      }
    }
    void checkDevices()
    return () => {
      active = false
    }
  }, [])

  const startWithFeedback = React.useCallback(
    (mode: RecognitionMode) => {
      emitFeedback({ type: "ui", kind: "start" })
      onStartCommunicating(mode)
    },
    [onStartCommunicating]
  )

  // Feedback (vibration / optional sound) whenever the selected mode changes
  const firstModeRef = React.useRef(true)
  React.useEffect(() => {
    if (firstModeRef.current) {
      firstModeRef.current = false
      return
    }
    emitFeedback({ type: "modeChange", mode: selectedMode })
  }, [selectedMode])

  // Keyboard shortcuts: Spacebar cycles modes, Arrow keys switch, Enter starts communicating
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (tourOpen) return
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      ) {
        return
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return

      if (e.code === "Space") {
        e.preventDefault()
        const idx = MODE_ORDER.indexOf(selectedMode)
        const next = MODE_ORDER[(idx + 1) % MODE_ORDER.length]
        onSelectMode(next)
      } else if (e.code === "ArrowRight") {
        e.preventDefault()
        const idx = MODE_ORDER.indexOf(selectedMode)
        onSelectMode(MODE_ORDER[(idx + 1) % MODE_ORDER.length])
      } else if (e.code === "ArrowLeft") {
        e.preventDefault()
        const idx = MODE_ORDER.indexOf(selectedMode)
        onSelectMode(MODE_ORDER[(idx - 1 + MODE_ORDER.length) % MODE_ORDER.length])
      } else if (e.code === "Enter" && !effectiveMenuOpen) {
        e.preventDefault()
        startWithFeedback(selectedMode)
      } else if (e.code === "Escape" && effectiveMenuOpen) {
        setMenuOpen(false)
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [selectedMode, onSelectMode, startWithFeedback, effectiveMenuOpen, tourOpen])

  // Close settings dropdown on outside click
  React.useEffect(() => {
    if (!effectiveMenuOpen) return
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null
      if (target?.closest?.('[data-tour-overlay="true"]')) return
      if (settingsBtnRef.current && !settingsBtnRef.current.contains(target)) {
        setMenuOpen(false)
      }
    }
    window.addEventListener("mousedown", onClick)
    return () => window.removeEventListener("mousedown", onClick)
  }, [effectiveMenuOpen])

  const handleCardPointerMove = (mode: RecognitionMode, e: React.PointerEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const nx = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1
    const ny = ((e.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1
    pointerRef.current[mode] = { x: nx, y: ny }
  }

  const handleCardPointerLeave = (mode: RecognitionMode) => {
    pointerRef.current[mode] = { x: 0, y: 0 }
    setHoveredMode((prev) => (prev === mode ? null : prev))
  }

  const handleCardClick = (mode: RecognitionMode) => {
    if (selectedMode === mode) {
      startWithFeedback(mode)
      return
    }
    onSelectMode(mode)
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-[#15121E] text-[#FAF6F0] flex flex-col select-none">
      {/* Velvet studio ambient backdrop */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(circle at 50% 48%, rgba(251, 106, 24, 0.11) 0%, rgba(45, 36, 68, 0.24) 40%, rgba(16, 13, 25, 0.92) 82%)",
        }}
      />

      {/* Minimal Header */}
      <header className="relative z-30 flex items-center justify-between px-6 sm:px-10 pt-6 pb-2">
        <a href="/" aria-label="SignBridge home" className="flex items-center gap-3 transition-opacity hover:opacity-80">
          <SignBridgeLogoMark size="sm" priority />
          <span className="font-display text-lg font-bold tracking-tight text-white">
            SignBridge
          </span>
        </a>

        <div className="flex items-center gap-2.5" ref={settingsBtnRef}>
          <button
            type="button"
            data-tour="menu-dictionary"
            onClick={() => onOpenWindow("dictionary")}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-white/6 px-3.5 py-1.5 text-xs font-medium text-[#FAF6F0] transition hover:border-amber-400/40 hover:bg-amber-400/12 hover:text-amber-200 cursor-pointer"
          >
            <BookOpen className="h-3.5 w-3.5 text-amber-300" />
            <span>3D Dictionary · 2,731</span>
          </button>

          <button
            type="button"
            data-tour="menu-custom"
            onClick={() => onOpenWindow("custom-learner")}
            className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/25 bg-amber-400/10 px-3.5 py-1.5 text-xs font-medium text-amber-200 transition hover:bg-amber-400/20"
          >
            <Sparkles className="h-3.5 w-3.5 text-amber-300" />
            <span>Custom Learner</span>
          </button>

          <button
            type="button"
            data-tour="menu-camera"
            onClick={() => startWithFeedback(selectedMode)}
            className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-[#1B3A2C]/85 px-3.5 py-1.5 text-xs font-medium text-emerald-200 transition hover:bg-[#1E4534]"
          >
            <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.9)]" />
            <span>{cameraReadyLabel}</span>
          </button>

          {onStartTour && (
            <button
              type="button"
              data-tour="menu-tour"
              onClick={onStartTour}
              title="Take interactive feature tour"
              className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-white/6 px-3 py-1.5 text-xs font-medium text-[#E8E3F2] transition hover:border-amber-400/40 hover:bg-amber-400/12 hover:text-amber-200 cursor-pointer"
            >
              <HelpCircle className="h-3.5 w-3.5 text-amber-300" />
              <span className="hidden sm:inline">Tour</span>
            </button>
          )}

          <div className="relative">
            <button
              type="button"
              data-tour="menu-tools"
              aria-label="Open Studio tools menu"
              onClick={() => setMenuOpen((prev) => !prev)}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-[#C9C3D8] transition hover:border-white/20 hover:bg-white/10 hover:text-white"
            >
              <Settings className="h-4 w-4" />
            </button>

            {effectiveMenuOpen && (
              <div className="absolute right-0 mt-2 w-60 rounded-2xl border border-white/12 bg-[#1D192B]/95 p-1.5 shadow-2xl backdrop-blur-xl z-50">
                <button
                  type="button"
                  onClick={() => {
                    setMenuOpen(false)
                    onOpenWindow("dictionary")
                  }}
                  className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs font-medium text-[#E8E3F2] hover:bg-white/10"
                >
                  <BookOpen className="h-3.5 w-3.5 text-amber-400" />
                  3D Sign Dictionary (2,731)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMenuOpen(false)
                    onOpenWindow("custom-learner")
                  }}
                  className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs font-medium text-[#E8E3F2] hover:bg-white/10"
                >
                  <Sparkles className="h-3.5 w-3.5 text-amber-400" />
                  Custom Sign Learner
                </button>
                <button
                  type="button"
                  data-tour="menu-tool-video"
                  onClick={() => {
                    setMenuOpen(false)
                    onOpenWindow("video")
                  }}
                  className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs font-medium text-[#E8E3F2] hover:bg-white/10"
                >
                  <FolderOpen className="h-3.5 w-3.5 text-amber-400" />
                  Analyze Uploaded Video
                </button>
                <div className="my-1 h-px bg-white/10" />
                <button
                  type="button"
                  data-tour="menu-tool-command"
                  onClick={() => {
                    setMenuOpen(false)
                    onOpenCommandPalette()
                  }}
                  className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-xs font-medium text-[#C8C2D6] hover:bg-white/10"
                >
                  <span className="inline-flex items-center gap-2.5">
                    <Command className="h-3.5 w-3.5 text-white/60" />
                    Command Palette
                  </span>
                  <span className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-white/70">⌘K</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Centered Studio Menu */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 sm:px-8 pb-10 pt-2">
        <div className="text-center max-w-xl mb-8 sm:mb-10">
          <h1 className="font-display text-3xl sm:text-4xl md:text-[2.75rem] font-bold tracking-tight text-white">
            How would you like to sign?
          </h1>
          <p className="mt-2.5 text-sm sm:text-[15px] text-[#ACA5BC]">
            Pick a mode to start communicating — SignBridge handles the recognition.
          </p>
        </div>

        {/* Three Interactive 3D Hand Cards */}
        <div className="grid w-full max-w-[820px] grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-5">
          {MODE_CARDS.map((card) => {
            const isSelected = selectedMode === card.mode
            const isHovered = hoveredMode === card.mode

            return (
              <button
                key={card.mode}
                type="button"
                data-tour={`mode-${card.mode}`}
                aria-pressed={isSelected}
                onClick={() => handleCardClick(card.mode)}
                onDoubleClick={() => startWithFeedback(card.mode)}
                onPointerEnter={() => setHoveredMode(card.mode)}
                onPointerMove={(e) => handleCardPointerMove(card.mode, e)}
                onPointerLeave={() => handleCardPointerLeave(card.mode)}
                className={[
                  "group relative flex flex-col items-center justify-between rounded-2xl px-5 pt-4 pb-6 text-center transition-all duration-300 outline-none focus-visible:ring-2 focus-visible:ring-amber-400/80",
                  "h-[250px] sm:h-[268px]",
                  isSelected
                    ? "bg-[#272136]/85 border border-amber-400/80 shadow-[0_0_45px_-10px_rgba(251,106,24,0.45),inset_0_1px_0_rgba(255,255,255,0.12)] -translate-y-0.5"
                    : "bg-[#221D30]/70 border border-white/10 shadow-[0_14px_30px_-14px_rgba(0,0,0,0.6),inset_0_1px_0_rgba(255,255,255,0.06)] hover:border-amber-300/45 hover:bg-[#272136]/85 hover:-translate-y-1",
                ].join(" ")}
              >
                {/* Warm internal spotlight glow behind the 3D hand */}
                <div
                  aria-hidden="true"
                  className={[
                    "pointer-events-none absolute top-6 left-1/2 -translate-x-1/2 h-32 w-32 rounded-full blur-2xl transition-opacity duration-300",
                    isHovered || isSelected ? "opacity-90 bg-amber-400/20" : "opacity-35 bg-amber-400/10",
                  ].join(" ")}
                />

                {/* Sculpted 3D Clay Hand Interactive Stage */}
                <div className="relative z-10 h-[146px] sm:h-[158px] w-full flex items-center justify-center">
                  <InteractiveMenuHand
                    mode={card.mode}
                    isHovered={isHovered}
                    isSelected={isSelected}
                    pointerRef={pointerRef}
                  />
                </div>

                {/* Label & 1-line subtitle */}
                <div className="relative z-30 mt-1">
                  <div className="font-display text-lg font-bold tracking-tight text-white">
                    {card.title}
                  </div>
                  <div className="mt-0.5 text-xs text-[#A39CB3]">
                    {card.subtitle}
                  </div>
                </div>
              </button>
            )
          })}
        </div>

        {/* Bottom Primary CTA + Shortcut Hint */}
        <div className="mt-8 sm:mt-10 flex flex-col items-center gap-3">
          <button
            type="button"
            data-tour="menu-start"
            onClick={() => startWithFeedback(selectedMode)}
            className="inline-flex items-center gap-2 rounded-full bg-gradient-to-b from-[#FF8A3D] to-[#FB6A18] px-7 py-3 text-sm font-bold text-[#1A130D] shadow-[0_12px_36px_-6px_rgba(251,106,24,0.62)] transition duration-200 hover:brightness-105 hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>Start Communicating</span>
            <ArrowRight className="h-4 w-4" />
          </button>

          <p className="text-[11px] font-medium tracking-wide text-[#7E7890]">
            Spacebar switches modes anytime
          </p>
        </div>
      </main>
    </div>
  )
}
