"use client"

import * as React from "react"
import { ArrowLeft, ArrowRight, Command, HelpCircle } from "lucide-react"
import { SignBridgeLogoMark } from "@/components/brand/signbridge-bridge-logo"
import { CommandPalette } from "@/components/navigation/command-palette"
import { CommunicationCanvas } from "@/components/studio/communication-canvas"
import {
  GuidedTour,
  TourAskPrompt,
  type TourStep,
} from "@/components/studio/guided-tour"
import { UnifiedModeMenu } from "@/components/studio/unified-mode-menu"
import { CustomSignsView } from "@/components/views/custom-signs-view"
import { DictionaryView } from "@/components/views/dictionary-view"
import { FeedbackDispatcher } from "@/components/feedback/feedback-dispatcher"
import { VideoAnalysisView } from "@/components/views/video-view"
import {
  RecognitionMode,
  SignBridgeStoreProvider,
  StudioWindowId
} from "@/lib/signbridge-store"

type StudioStage = "menu" | "communicate" | StudioWindowId
type TourScope = "none" | "menu" | "studio" | "custom-learner"
type TourPromptScope = "none" | "menu" | "custom-learner"

const TOUR_STORAGE_KEY = "signbridge.tour.seen.v1"
const CUSTOM_TOUR_STORAGE_KEY = "signbridge.custom-tour.seen.v1"
const TOUR_OPT_OUT_KEY = "signbridge.tour.optout.v1"

function readStorageFlag(key: string): string | null {
  try {
    return typeof window !== "undefined" ? window.localStorage.getItem(key) : null
  } catch {
    return null
  }
}

function writeStorageFlag(key: string, value: string) {
  try {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(key, value)
    }
  } catch {
    // Ignore storage quota or private browsing errors
  }
}

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
  const [tourScope, setTourScope] = React.useState<TourScope>("none")
  const [tourPromptScope, setTourPromptScope] = React.useState<TourPromptScope>("none")
  const [tourRunId, setTourRunId] = React.useState(0)
  const [forceToolsMenuOpen, setForceToolsMenuOpen] = React.useState<boolean | null>(null)
  const [forceStudioPopoverOpen, setForceStudioPopoverOpen] = React.useState<boolean | null>(null)

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
    let autoTourTimer: number | undefined
    const syncTimer = window.setTimeout(() => {
      const initialStage = stageFromHash(window.location.hash)
      if (initialStage !== "menu") {
        setStudioStage(initialStage)
      } else if (!readStorageFlag(TOUR_OPT_OUT_KEY) && !readStorageFlag(TOUR_STORAGE_KEY)) {
        autoTourTimer = window.setTimeout(() => {
          writeStorageFlag(TOUR_STORAGE_KEY, "1")
          setTourPromptScope("menu")
        }, 550)
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
      setTourScope("none")
      setTourPromptScope("none")
      setForceToolsMenuOpen(null)
      setForceStudioPopoverOpen(null)
      setStudioStage(targetStage)
    }

    window.addEventListener("popstate", onPopState)
    return () => {
      window.clearTimeout(syncTimer)
      if (autoTourTimer !== undefined) window.clearTimeout(autoTourTimer)
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

  React.useEffect(() => {
    if (studioStage !== "custom-learner" || typeof window === "undefined") return
    if (readStorageFlag(TOUR_OPT_OUT_KEY) || readStorageFlag(CUSTOM_TOUR_STORAGE_KEY)) {
      return
    }
    const timer = window.setTimeout(() => {
      writeStorageFlag(CUSTOM_TOUR_STORAGE_KEY, "1")
      setTourPromptScope("custom-learner")
    }, 450)
    return () => window.clearTimeout(timer)
  }, [studioStage])

  const navigateToStage = React.useCallback(
    (nextStage: StudioStage, nextMode?: RecognitionMode) => {
      const resolvedMode = nextMode ?? modeRef.current
      if (nextMode) {
        setSelectedMode(nextMode)
      }
      setTourScope("none")
      setTourPromptScope("none")
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
    setTourScope("none")
    setTourPromptScope("none")
    setForceStudioPopoverOpen(null)
    navigateToStage("menu")
  }, [navigateToStage])

  const markTourSeen = React.useCallback(() => {
    writeStorageFlag(TOUR_STORAGE_KEY, "1")
  }, [])

  const closeTour = React.useCallback(() => {
    markTourSeen()
    setTourScope("none")
    setForceToolsMenuOpen(null)
    setForceStudioPopoverOpen(null)
  }, [markTourSeen])

  const startMenuTour = React.useCallback(() => {
    markTourSeen()
    setTourPromptScope("none")
    setForceToolsMenuOpen(null)
    setTourRunId((k) => k + 1)
    setTourScope("menu")
  }, [markTourSeen])

  const startStudioTour = React.useCallback(() => {
    setTourPromptScope("none")
    setForceStudioPopoverOpen(null)
    setTourRunId((k) => k + 1)
    setTourScope("studio")
  }, [])

  const startStudioTourFromMenu = React.useCallback(() => {
    markTourSeen()
    setTourPromptScope("none")
    setForceToolsMenuOpen(null)
    setForceStudioPopoverOpen(null)
    navigateToStage("communicate", modeRef.current)
    setTourRunId((k) => k + 1)
    setTourScope("studio")
  }, [markTourSeen, navigateToStage])

  const markCustomTourSeen = React.useCallback(() => {
    writeStorageFlag(CUSTOM_TOUR_STORAGE_KEY, "1")
  }, [])

  const closeCustomTour = React.useCallback(() => {
    markCustomTourSeen()
    setTourScope("none")
  }, [markCustomTourSeen])

  const startCustomTour = React.useCallback(() => {
    markCustomTourSeen()
    setTourPromptScope("none")
    setTourRunId((k) => k + 1)
    setTourScope("custom-learner")
  }, [markCustomTourSeen])

  const startStudioTourFromCustom = React.useCallback(() => {
    markCustomTourSeen()
    setTourPromptScope("none")
    setForceStudioPopoverOpen(null)
    navigateToStage("communicate", modeRef.current)
    setTourRunId((k) => k + 1)
    setTourScope("studio")
  }, [markCustomTourSeen, navigateToStage])

  const dismissTourPrompt = React.useCallback(() => {
    if (tourPromptScope === "menu") markTourSeen()
    if (tourPromptScope === "custom-learner") markCustomTourSeen()
    setTourPromptScope("none")
  }, [tourPromptScope, markTourSeen, markCustomTourSeen])

  const dontShowToursAgain = React.useCallback(() => {
    writeStorageFlag(TOUR_OPT_OUT_KEY, "1")
    markTourSeen()
    markCustomTourSeen()
    setTourPromptScope("none")
    setTourScope("none")
  }, [markTourSeen, markCustomTourSeen])

  const acceptTourPrompt = React.useCallback(() => {
    if (tourPromptScope === "custom-learner") {
      startCustomTour()
      return
    }
    startMenuTour()
  }, [tourPromptScope, startCustomTour, startMenuTour])

  const menuTourSteps = React.useMemo<TourStep[]>(
    () => [
      {
        id: "menu-dictionary",
        target: '[data-tour="menu-dictionary"]',
        badge: "Reference",
        title: "3D Sign Dictionary (2,731 Words)",
        description:
          "Browse all 2,731 ASL signs as animated 3D hand poses. Search any word, scrub frames, and orbit between Front, Side, Top, or Signer POV.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "menu-custom",
        target: '[data-tour="menu-custom"]',
        badge: "Personalize",
        title: "Custom Sign Learner",
        description:
          "Teach SignBridge your own gestures or regional signs by recording 3 quick clips and training your private custom profile.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "menu-camera",
        target: '[data-tour="menu-camera"]',
        badge: "Quick Launch",
        title: "Camera Readiness & Instant Start",
        description:
          "Shows when your webcam is ready. Click this pill anytime to jump straight into Live Studio with your active signing mode.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "menu-tools",
        target: '[data-tour="menu-tools"]',
        badge: "Menu",
        title: "Studio Tools Menu",
        description:
          "Opens extra tools including Video Analysis, Custom Sign Learner, 3D Dictionary, and the quick Command Palette.",
        placement: "bottom",
        radius: 14,
        onBeforeEnter: () => setForceToolsMenuOpen(null),
      },
      {
        id: "menu-tool-video",
        target: '[data-tour="menu-tool-video"]',
        badge: "Offline Clips",
        title: "Analyze Uploaded Video",
        description:
          "Upload a saved MP4, MOV, or WebM recording to recognize ASL signs directly from video files without needing a live webcam.",
        placement: "left",
        radius: 12,
        onBeforeEnter: () => setForceToolsMenuOpen(true),
      },
      {
        id: "menu-tool-command",
        target: '[data-tour="menu-tool-command"]',
        badge: "Shortcut",
        title: "Command Palette (⌘K)",
        description:
          "Press ⌘K (or Ctrl+K) from anywhere to search the 2,731-sign dictionary or jump straight to any signing mode or tool.",
        placement: "left",
        radius: 12,
        onBeforeEnter: () => setForceToolsMenuOpen(true),
      },
      {
        id: "mode-Word",
        target: '[data-tour="mode-Word"]',
        badge: "Mode 1 of 3",
        title: "Words Mode",
        description:
          "Recognize one distinct ASL vocabulary sign at a time with top-3 quick match candidates so you can confirm or practice signs.",
        placement: "bottom",
        radius: 18,
        onBeforeEnter: () => setForceToolsMenuOpen(false),
      },
      {
        id: "mode-Continuous",
        target: '[data-tour="mode-Continuous"]',
        badge: "Mode 2 of 3",
        title: "Continuous Mode",
        description:
          "Sign naturally phrase after phrase. Pause briefly (~1 second) and SignBridge turns your recognized glosses into smooth English sentences.",
        placement: "bottom",
        radius: 18,
        onBeforeEnter: () => setForceToolsMenuOpen(null),
      },
      {
        id: "mode-Fingerspelling",
        target: '[data-tour="mode-Fingerspelling"]',
        badge: "Mode 3 of 3",
        title: "Fingerspelling Mode",
        description:
          "Spell names, acronyms, and custom words letter-by-letter from A to Z using static ASL handshapes.",
        placement: "bottom",
        radius: 18,
      },
      {
        id: "menu-tour",
        target: '[data-tour="menu-tour"]',
        badge: "Help",
        title: "Replay Tour Anytime",
        description:
          "Click Tour whenever you want a quick refresher on every SignBridge button or shortcut.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "menu-start",
        target: '[data-tour="menu-start"]',
        badge: "Go Live",
        title: "Start Communicating",
        description:
          "Opens the Live Studio webcam view in your selected mode — or click Tour Live Studio to step through every button inside the camera workspace.",
        placement: "top",
        radius: 999,
      },
    ],
    []
  )

  const studioTourSteps = React.useMemo<TourStep[]>(
    () => [
      {
        id: "studio-menu",
        target: '[data-tour="studio-menu"]',
        badge: "Navigation",
        title: "Back to Mode Menu",
        description:
          "Return to the home mode menu, 3D Dictionary, Custom Sign Learner, or Video Analysis anytime.",
        placement: "bottom",
        radius: 999,
        onBeforeEnter: () => setForceStudioPopoverOpen(null),
      },
      {
        id: "studio-mode-switcher",
        target: '[data-tour="studio-mode-switcher"]',
        badge: "Modes",
        title: "Live Mode Switcher",
        description:
          "Switch seamlessly between Words, Continuous, and Fingerspelling (or press Spacebar) without interrupting your camera.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "studio-tracking",
        target: '[data-tour="studio-tracking"]',
        badge: "Vision",
        title: "Hand Tracking Status",
        description:
          "Turns green as soon as your hands are tracked in frame so you know SignBridge sees your gestures clearly.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "studio-auto-tts",
        target: '[data-tour="studio-auto-tts"]',
        badge: "Voice",
        title: "Auto TTS",
        description:
          "Automatically speak completed English sentences and signed words out loud as you finish signing.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "studio-pause",
        target: '[data-tour="studio-pause"]',
        badge: "Camera",
        title: "Pause / Resume Camera",
        description:
          "Freeze or resume live webcam recognition whenever you want to take a break.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "studio-speak",
        target: '[data-tour="studio-speak"]',
        badge: "Audio",
        title: "Speak Translation Aloud",
        description:
          "Play back the current translated text out loud anytime with one click.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "studio-settings",
        target: '[data-tour="studio-settings"]',
        badge: "Controls",
        title: "Camera & Custom Shortcuts",
        description:
          "Switch webcam devices, flip Mirror Preview, or jump directly into Custom Sign Training.",
        placement: "bottom",
        radius: 999,
        onBeforeEnter: () => setForceStudioPopoverOpen(null),
      },
      {
        id: "studio-camera-select",
        target: '[data-tour="studio-camera-select"]',
        badge: "Hardware",
        title: "Camera Source",
        description:
          "Choose which built-in or external webcam SignBridge reads from.",
        placement: "left",
        radius: 12,
        onBeforeEnter: () => setForceStudioPopoverOpen(true),
      },
      {
        id: "studio-mirror",
        target: '[data-tour="studio-mirror"]',
        badge: "Preview",
        title: "Mirror Camera Preview",
        description:
          "Flip your webcam horizontally like a mirror while keeping ASL hand orientation accurate for recognition.",
        placement: "left",
        radius: 12,
        onBeforeEnter: () => setForceStudioPopoverOpen(true),
      },
      {
        id: "studio-train-custom",
        target: '[data-tour="studio-train-custom"]',
        badge: "Personalize",
        title: "Train Custom Signs",
        description:
          "Jump directly to Custom Sign Learner to record and train your own custom signs.",
        placement: "left",
        radius: 12,
        onBeforeEnter: () => setForceStudioPopoverOpen(true),
      },
      {
        id: "studio-caption-bar",
        target: '[data-tour="studio-caption-bar"]',
        badge: "Output",
        title: "Live Caption & Output Bar",
        description:
          "Watch recognized words, letters, and smoothed English sentences appear live — plus quick Sign Word, Space, Backspace, Clear, Copy, and Speak buttons.",
        placement: "top",
        radius: 18,
        onBeforeEnter: () => setForceStudioPopoverOpen(false),
      },
      {
        id: "studio-hand-guide",
        target: '[data-tour="studio-hand-guide"]',
        badge: "3D Guide",
        title: "Interactive 3D Sign Guide",
        description:
          "Preview the current sign or handshape in 3D — drag to rotate or click to expand.",
        placement: "left",
        radius: 18,
        onBeforeEnter: () => setForceStudioPopoverOpen(null),
      },
    ],
    []
  )

  const customTourSteps = React.useMemo<TourStep[]>(
    () => [
      {
        id: "custom-guide",
        target: '[data-tour="custom-guide"]',
        badge: "Overview",
        title: "How Few-Shot Custom Learning Works",
        description:
          "SignBridge encodes just 3 short 2.2s clips per sign into personal 512-D Conformer prototypes saved strictly to your profile — ready in seconds with zero cloud GPU wait.",
        placement: "bottom",
        radius: 22,
      },
      {
        id: "custom-gloss-input",
        target: '[data-tour="custom-gloss-input"]',
        badge: "Step 1 · Name",
        title: "Name or Search a Sign",
        description:
          "Type any brand-new sign label (like COFFEE or MY_NAME) or search all 2,731 ASL glosses. Press Enter or click Use to set your recording target.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "custom-gloss-pills",
        target: '[data-tour="custom-gloss-pills"]',
        badge: "Quick Select",
        title: "Starter & Saved Sign Chips",
        description:
          "Click any pill to switch targets fast. Custom signs you record automatically appear at the front of this row.",
        placement: "bottom",
        radius: 16,
      },
      {
        id: "custom-camera-stage",
        target: '[data-tour="custom-camera-stage"]',
        badge: "Step 2 · Preview",
        title: "Live Hand-Tracked Camera",
        description:
          "Watch live MediaPipe hand keypoints over your webcam feed. Wait until the top-left badge shows Hands Detected in green before capturing.",
        placement: "right",
        radius: 20,
      },
      {
        id: "custom-record-sample",
        target: '[data-tour="custom-record-sample"]',
        badge: "Capture",
        title: "Record Sample (2.2s)",
        description:
          "Captures a 2.2-second clip (32 resampled frames) for your target sign. Motion quality is checked automatically and 3 valid clips unlock prototype readiness.",
        placement: "top",
        radius: 999,
      },
      {
        id: "custom-test-sign",
        target: '[data-tour="custom-test-sign"]',
        badge: "Verify",
        title: "Live Test Sign (2.2s)",
        description:
          "Sign once without saving to check confidence, top-K matches, and confirm whether your Custom Model or Base Model recognized the gesture.",
        placement: "top",
        radius: 999,
      },
      {
        id: "custom-pause-camera",
        target: '[data-tour="custom-pause-camera"]',
        badge: "Camera",
        title: "Pause / Resume Camera",
        description:
          "Temporarily turns off the webcam and MediaPipe tracker anytime you need to step away or switch setups.",
        placement: "top",
        radius: 999,
      },
      {
        id: "custom-train-model",
        target: '[data-tour="custom-train-model"]',
        badge: "Step 3 · Compile",
        title: "Train Custom Model",
        description:
          "Unlocks once you record samples. Click here to compile personal few-shot prototypes across all your saved custom signs and activate them.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "custom-prediction-toggle",
        target: '[data-tour="custom-prediction-toggle"]',
        badge: "Live Routing",
        title: "Prediction: Active in Live / Standby",
        description:
          "Controls whether your personal prototypes override base vocabulary inside Live Studio (glows orange when Active in Live).",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "custom-saved-panel",
        target: '[data-tour="custom-saved-panel"]',
        badge: "Library",
        title: "Saved Personalized Signs",
        description:
          "Track each sign from 0/3 to Model Ready, click any card to record more clips, remove a sign with its trash icon, sync via Refresh, or Reset All.",
        placement: "left",
        radius: 24,
      },
      {
        id: "custom-tour",
        target: '[data-tour="custom-tour"]',
        badge: "Help",
        title: "Replay This Tour Anytime",
        description:
          "Click Tour in the top bar (or Take Button Tour above) whenever you want to review Custom Sign Learner controls.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "custom-back",
        target: '[data-tour="custom-back"]',
        badge: "Navigation",
        title: "Back to Menu",
        description:
          "Return to the home mode selector, 3D Sign Dictionary, or Video Analysis anytime.",
        placement: "bottom",
        radius: 999,
      },
      {
        id: "custom-open-live",
        target: '[data-tour="custom-open-live"]',
        badge: "Step 4 · Go Live",
        title: "Open Live Studio",
        description:
          "Jump straight into real-time webcam translation using your trained custom signs — or click Tour Live Studio to tour the live canvas.",
        placement: "bottom",
        radius: 999,
      },
    ],
    []
  )

  const renderStudioWindow = (win: StudioWindowId) => {
    switch (win) {
      case "custom-learner":
        return (
          <CustomSignsView
            initialGloss={customGlossSeed}
            onStartTour={startCustomTour}
          />
        )
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
      <FeedbackDispatcher />
      {studioStage === "menu" ? (
        <UnifiedModeMenu
          selectedMode={selectedMode}
          onSelectMode={setSelectedMode}
          onStartCommunicating={openMode}
          onOpenWindow={openWindow}
          onOpenCommandPalette={() => setIsCommandOpen(true)}
          tourOpen={tourScope === "menu"}
          forceToolsMenuOpen={forceToolsMenuOpen}
          onStartTour={startMenuTour}
        />
      ) : studioStage === "communicate" ? (
        <CommunicationCanvas
          initialMode={selectedMode}
          onBackToMenu={goBackToMenu}
          onOpenWindow={openWindow}
          tourOpen={tourScope === "studio"}
          forceSettingsPopoverOpen={forceStudioPopoverOpen}
          onStartTour={startStudioTour}
        />
      ) : (
        <div className="relative min-h-screen w-full overflow-x-hidden bg-[#15121E] text-[#FAF6F0]">
          <div
            aria-hidden="true"
            className="pointer-events-none fixed inset-0"
            style={{
              background:
                "radial-gradient(circle at 50% 18%, rgba(251, 106, 24, 0.08) 0%, rgba(45, 36, 68, 0.22) 45%, rgba(16, 13, 25, 0.96) 85%)",
            }}
          />

          <header className="sticky top-0 z-30 border-b border-white/10 bg-[#15121E]/88 backdrop-blur-xl">
            <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-3 px-4 sm:px-8 py-3.5">
              <div className="flex items-center gap-3 min-w-0">
                <button
                  type="button"
                  data-tour={studioStage === "custom-learner" ? "custom-back" : undefined}
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
                {studioStage === "custom-learner" && (
                  <button
                    type="button"
                    data-tour="custom-tour"
                    onClick={startCustomTour}
                    title="Take Custom Sign Learner guided tour"
                    aria-label="Take Custom Sign Learner guided tour"
                    className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/35 bg-amber-400/10 px-3 py-1.5 text-xs font-semibold text-amber-200 transition hover:bg-amber-400/20 cursor-pointer"
                  >
                    <HelpCircle className="h-3.5 w-3.5" />
                    <span>Tour</span>
                  </button>
                )}

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
                  data-tour={studioStage === "custom-learner" ? "custom-open-live" : undefined}
                  onClick={() => openMode(modeRef.current)}
                  className="inline-flex items-center gap-1.5 rounded-full bg-gradient-to-b from-[#FF8A3D] to-[#FB6A18] px-4 py-1.5 text-xs font-bold text-[#1A130D] shadow-[0_8px_24px_-6px_rgba(251,106,24,0.55)] transition hover:brightness-105 cursor-pointer"
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

      <GuidedTour
        key={`menu-${tourRunId}`}
        open={studioStage === "menu" && tourScope === "menu"}
        steps={menuTourSteps}
        finishLabel="Finish"
        secondaryFinishAction={{
          label: "Tour Live Studio",
          onClick: startStudioTourFromMenu,
        }}
        onClose={closeTour}
      />

      <GuidedTour
        key={`studio-${tourRunId}`}
        open={studioStage === "communicate" && tourScope === "studio"}
        steps={studioTourSteps}
        finishLabel="Start Signing"
        onClose={closeTour}
      />

      <GuidedTour
        key={`custom-${tourRunId}`}
        open={studioStage === "custom-learner" && tourScope === "custom-learner"}
        steps={customTourSteps}
        finishLabel="Start Training"
        secondaryFinishAction={{
          label: "Tour Live Studio",
          onClick: startStudioTourFromCustom,
        }}
        onClose={closeCustomTour}
      />

      <TourAskPrompt
        open={
          tourScope === "none" &&
          ((studioStage === "menu" && tourPromptScope === "menu") ||
            (studioStage === "custom-learner" &&
              tourPromptScope === "custom-learner"))
        }
        badge={
          tourPromptScope === "custom-learner"
            ? "Custom Learner Guide"
            : "Welcome Tour"
        }
        title={
          tourPromptScope === "custom-learner"
            ? "Want a quick tour of Custom Sign Learner?"
            : "New to SignBridge? Take a quick tour"
        }
        description={
          tourPromptScope === "custom-learner"
            ? "See what each button does and learn how 3 short clips turn into a custom sign. You can replay anytime via Tour."
            : "Walk button-by-button through every feature in floating bubbles. You can replay anytime from the Tour button."
        }
        stepCount={
          tourPromptScope === "custom-learner"
            ? customTourSteps.length
            : menuTourSteps.length
        }
        onStartTour={acceptTourPrompt}
        onDismiss={dismissTourPrompt}
        onDontShowAgain={dontShowToursAgain}
      />
    </SignBridgeStoreProvider>
  )
}
