"use client"

import * as React from "react"
import { ArrowLeft, ArrowRight, Command } from "lucide-react"
import { SignBridgeLogoMark } from "@/components/brand/signbridge-bridge-logo"
import { CommandPalette } from "@/components/navigation/command-palette"
import { CommunicationCanvas } from "@/components/studio/communication-canvas"
import { UnifiedModeMenu } from "@/components/studio/unified-mode-menu"
import { CustomSignsView } from "@/components/views/custom-signs-view"
import { DictionaryView } from "@/components/views/dictionary-view"
import { VideoAnalysisView } from "@/components/views/video-view"
import {
  RecognitionMode,
  SignBridgeStoreProvider,
  StudioWindowId
} from "@/lib/signbridge-store"

type StudioStage = "menu" | "communicate" | StudioWindowId

const STUDIO_WINDOW_META: Record<
  StudioWindowId,
  { title: string; subtitle: string; badge: string }
> = {
  "custom-learner": {
    title: "Custom Sign Learner",
    subtitle: "Record samples, train your personal signs, and verify live prediction",
    badge: "Personal Model",
  },
  video: {
    title: "Analyze Uploaded Video",
    subtitle: "Upload saved MP4, MOV, or WebM clips for landmark ASL inference",
    badge: "Video Inference",
  },
  dictionary: {
    title: "3D Sign Dictionary",
    subtitle: "Interactive 3D anatomical hand reference for all 2,731 ASL glosses",
    badge: "2,731 Glosses",
  },
}

function isStudioWindow(stage: StudioStage): stage is StudioWindowId {
  return (
    stage === "custom-learner" ||
    stage === "video" ||
    stage === "dictionary"
  )
}

const HASH_TO_STAGE: Record<string, StudioStage> = {
  "#live": "communicate",
  "#custom-learner": "custom-learner",
  "#video": "video",
  "#dictionary": "dictionary",
}

const STAGE_TO_HASH: Record<StudioStage, string> = {
  menu: "",
  communicate: "#live",
  "custom-learner": "#custom-learner",
  video: "#video",
  dictionary: "#dictionary",
}

function stageFromHash(hash: string): StudioStage {
  return HASH_TO_STAGE[hash] ?? "menu"
}

function buildStageUrl(stage: StudioStage): string {
  if (typeof window === "undefined") return "/"
  const base = `${window.location.pathname}${window.location.search}`
  return `${base}${STAGE_TO_HASH[stage]}`
}

export default function SignBridgeAppPage() {
  const [studioStage, setStudioStage] = React.useState<StudioStage>("menu")
  const [selectedMode, setSelectedMode] = React.useState<RecognitionMode>("Continuous")
  const [isCommandOpen, setIsCommandOpen] = React.useState(false)
  const [customGlossSeed, setCustomGlossSeed] = React.useState<string | null>(null)
  const [dictionaryGlossSeed, setDictionaryGlossSeed] = React.useState<string | undefined>(undefined)

  const stageRef = React.useRef<StudioStage>(studioStage)
  const modeRef = React.useRef<RecognitionMode>(selectedMode)
  React.useEffect(() => {
    stageRef.current = studioStage
  }, [studioStage])
  React.useEffect(() => {
    modeRef.current = selectedMode
  }, [selectedMode])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const syncTimer = window.setTimeout(() => {
      const initialStage = stageFromHash(window.location.hash)
      if (initialStage !== "menu") {
        setStudioStage(initialStage)
      }
      window.history.replaceState(
        { studioStage: initialStage, selectedMode: modeRef.current },
        "",
        buildStageUrl(initialStage)
      )
    }, 0)

    const onPopState = (event: PopStateEvent) => {
      const state = event.state as {
        studioStage?: StudioStage
        selectedMode?: RecognitionMode
      } | null
      const targetStage = state?.studioStage ?? stageFromHash(window.location.hash)
      if (state?.selectedMode) {
        setSelectedMode(state.selectedMode)
      }
      setStudioStage(targetStage)
    }

    window.addEventListener("popstate", onPopState)
    return () => {
      window.clearTimeout(syncTimer)
      window.removeEventListener("popstate", onPopState)
    }
  }, [])

  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setIsCommandOpen((prev) => !prev)
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [])

  const navigateToStage = React.useCallback(
    (nextStage: StudioStage, nextMode?: RecognitionMode) => {
      const resolvedMode = nextMode ?? modeRef.current
      if (nextMode) {
        setSelectedMode(nextMode)
      }
      if (typeof window !== "undefined") {
        const sameStage = stageRef.current === nextStage
        const sameMode = modeRef.current === resolvedMode
        if (!sameStage || (nextStage === "communicate" && !sameMode)) {
          window.history.pushState(
            { studioStage: nextStage, selectedMode: resolvedMode },
            "",
            buildStageUrl(nextStage)
          )
        }
      }
      setStudioStage(nextStage)
    },
    []
  )

  const openMode = React.useCallback(
    (mode: RecognitionMode) => {
      navigateToStage("communicate", mode)
    },
    [navigateToStage]
  )

  const openWindow = React.useCallback(
    (win: StudioWindowId) => {
      navigateToStage(win)
    },
    [navigateToStage]
  )

  const goBackToMenu = React.useCallback(() => {
    navigateToStage("menu")
  }, [navigateToStage])

  const renderStudioWindow = (win: StudioWindowId) => {
    switch (win) {
      case "custom-learner":
        return <CustomSignsView initialGloss={customGlossSeed} />
      case "video":
        return <VideoAnalysisView />
      case "dictionary":
        return (
          <DictionaryView
            initialGloss={dictionaryGlossSeed}
            onTrySigningLive={() => openMode("Word")}
          />
        )
    }
  }

  const activeWindowMeta = isStudioWindow(studioStage)
    ? STUDIO_WINDOW_META[studioStage]
    : null

  return (
    <SignBridgeStoreProvider>
      {studioStage === "menu" ? (
        <UnifiedModeMenu
          selectedMode={selectedMode}
          onSelectMode={setSelectedMode}
          onStartCommunicating={openMode}
          onOpenWindow={openWindow}
          onOpenCommandPalette={() => setIsCommandOpen(true)}
        />
      ) : studioStage === "communicate" ? (
        <CommunicationCanvas
          initialMode={selectedMode}
          onBackToMenu={goBackToMenu}
          onOpenWindow={openWindow}
        />
      ) : (
        <div className="relative min-h-screen w-full overflow-x-hidden bg-[#15121E] text-[#FAF6F0]">
          <div
            aria-hidden="true"
            className="pointer-events-none fixed inset-0"
            style={{
              background:
                "radial-gradient(circle at 50% 18%, rgba(245, 158, 11, 0.08) 0%, rgba(45, 36, 68, 0.22) 45%, rgba(16, 13, 25, 0.96) 85%)",
            }}
          />

          <header className="sticky top-0 z-30 border-b border-white/10 bg-[#15121E]/88 backdrop-blur-xl">
            <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-3 px-4 sm:px-8 py-3.5">
              <div className="flex items-center gap-3 min-w-0">
                <button
                  type="button"
                  onClick={goBackToMenu}
                  className="inline-flex items-center gap-1.5 rounded-full border border-white/12 bg-white/5 px-3.5 py-1.5 text-xs font-semibold text-[#E8E3F2] transition hover:border-white/25 hover:bg-white/10 hover:text-white shrink-0 cursor-pointer"
                >
                  <ArrowLeft className="h-3.5 w-3.5" />
                  <span>Back to Menu</span>
                </button>

                {activeWindowMeta && (
                  <div className="hidden sm:flex items-center gap-2.5 min-w-0">
                    <span className="h-4 w-px bg-white/15 shrink-0" />
                    <SignBridgeLogoMark size="xs" priority />
                    <span className="text-sm font-bold text-white truncate">
                      {activeWindowMeta.title}
                    </span>
                    <span className="rounded-full border border-amber-400/25 bg-amber-400/10 px-2.5 py-0.5 text-[10px] font-semibold text-amber-200 shrink-0">
                      {activeWindowMeta.badge}
                    </span>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => setIsCommandOpen(true)}
                  className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-[#ACA5BC] transition hover:bg-white/10 hover:text-white cursor-pointer"
                >
                  <Command className="h-3.5 w-3.5" />
                  <span>⌘K</span>
                </button>

                <button
                  type="button"
                  onClick={() => openMode(modeRef.current)}
                  className="inline-flex items-center gap-1.5 rounded-full bg-gradient-to-b from-[#FBBF24] to-[#F59E0B] px-4 py-1.5 text-xs font-bold text-[#1A130D] shadow-[0_8px_24px_-6px_rgba(245,158,11,0.55)] transition hover:brightness-105 cursor-pointer"
                >
                  <span>Open Live Studio</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </header>

          <main
            className={`relative z-10 mx-auto w-full ${
              studioStage === "dictionary"
                ? "max-w-[1440px] px-4 sm:px-6 pt-4 pb-6"
                : "max-w-6xl px-4 sm:px-8 pt-6 pb-12"
            }`}
          >
            {isStudioWindow(studioStage) && renderStudioWindow(studioStage)}
          </main>
        </div>
      )}

      <CommandPalette
        open={isCommandOpen}
        onClose={() => setIsCommandOpen(false)}
        onOpenMode={openMode}
        onOpenWindow={openWindow}
        onSelectCustomGloss={(gloss) => setCustomGlossSeed(gloss)}
        onSelectDictionaryGloss={(gloss) => setDictionaryGlossSeed(gloss)}
      />
    </SignBridgeStoreProvider>
  )
}
