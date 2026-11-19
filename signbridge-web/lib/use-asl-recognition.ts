"use client"

import * as React from "react"
import { toast } from "sonner"
import {
  RecognitionMode,
  useSignBridgeStore
} from "@/lib/signbridge-store"

import {
  clearMediaPipeHandsListener,
  getSharedMediaPipeHands,
  MediaPipeHandsResults as SharedHandsResults,
  sendVideoFrameToSharedHands,
  setMediaPipeHandsListener
} from "@/lib/mediapipe-hands"
import { emitFeedback, type FeedbackPoint } from "@/lib/feedback/bus"

export interface TopCandidateItem {
  gloss: string
  rawGloss: string
  confidence: number
}

export type RecognitionModelVariant = "production_exp13" | "demo_50"
export type WordCaptureState = "idle" | "recording" | "analyzing"

export interface CameraDeviceOption {
  deviceId: string
  label: string
}

export interface CameraErrorState {
  code: "permission-denied" | "not-found" | "device-busy" | "unsupported" | "unknown"
  message: string
}

export interface LivePredictionSnapshot {
  label: string
  confidence: number
}

interface LandmarkPoint {
  x: number
  y: number
  z?: number
}

function cleanConfidencePercentage(rawConf: unknown): number {
  const val = typeof rawConf === "number" ? rawConf : parseFloat(String(rawConf ?? 0)) || 0
  if (!Number.isFinite(val) || val <= 0) return 0
  return Math.min(99.9, Math.max(0.1, Number(val.toFixed(1))))
}

function normalizeContinuousConfPct(rawConf: unknown): number {
  const val = typeof rawConf === "number" ? rawConf : parseFloat(String(rawConf ?? 0)) || 0
  if (!Number.isFinite(val) || val <= 0) return 0
  return cleanConfidencePercentage(val <= 1.0 ? val * 100 : val)
}

function formatGlossLabel(gloss: string): string {
  const cleaned = gloss.trim().replace(/\d+$/g, "")
  return cleaned
    .split(/[\s_]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ")
}

export function useAslRecognition(initialMode: RecognitionMode = "Continuous") {
  const { reportInferenceTelemetry, recordSession, preferences, personalizationEnabled } =
    useSignBridgeStore()
  const preferencesRef = React.useRef(preferences)
  React.useEffect(() => {
    preferencesRef.current = preferences
  }, [preferences])
  const effectivePersonalization = Boolean(
    personalizationEnabled || preferences.personalizationEnabled
  )
  const personalizationEnabledRef = React.useRef(effectivePersonalization)
  React.useEffect(() => {
    personalizationEnabledRef.current = effectivePersonalization
  }, [effectivePersonalization])

  const [mode, setMode] = React.useState<RecognitionMode>(initialMode)
  const modeRef = React.useRef<RecognitionMode>(initialMode)

  const [devices, setDevices] = React.useState<CameraDeviceOption[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = React.useState<string>("")
  const [isStreaming, setIsStreaming] = React.useState(false)
  const [isPaused, setIsPaused] = React.useState(false)
  const [cameraError, setCameraError] = React.useState<CameraErrorState | null>(null)
  const [fps, setFps] = React.useState<number | null>(null)
  const [isHandDetected, setIsHandDetected] = React.useState(false)

  // Real recognition outputs ONLY (never seeded with fake phrases)
  const [glosses, setGlosses] = React.useState<string[]>([])
  const [smoothedText, setSmoothedText] = React.useState<string>("")
  const [fingerspellingWord, setFingerspellingWord] = React.useState<string>("")
  const [activePrediction, setActivePrediction] =
    React.useState<LivePredictionSnapshot | null>(null)
  const [isStrokeActive, setIsStrokeActive] = React.useState(false)
  const [isSpeaking, setIsSpeaking] = React.useState(false)

  const videoRef = React.useRef<HTMLVideoElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const mediaStreamRef = React.useRef<MediaStream | null>(null)
  const handsDetectorRef = React.useRef<boolean>(false)
  const animFrameRef = React.useRef<number | null>(null)
  const isMountedRef = React.useRef(true)
  // Feedback: last palm position (normalized video coords) + hand presence hysteresis
  const handPointRef = React.useRef<FeedbackPoint | null>(null)
  const handShownRef = React.useRef(false)
  const lastHandSeenMsRef = React.useRef(0)
  const continuousInStrokeRef = React.useRef(false)
  const isStreamingRef = React.useRef(false)
  const isPausedRef = React.useRef(false)
  const isProcessingFrameRef = React.useRef(false)
  const isInferencingRef = React.useRef(false)
  const prevHandWristsRef = React.useRef<{
    slot0: [number, number] | null
    slot1: [number, number] | null
    lastSeenMs: number
  }>({ slot0: null, slot1: null, lastSeenMs: 0 })

  const [topCandidates, setTopCandidates] = React.useState<TopCandidateItem[]>([])
  const [modelVariant, setModelVariantState] = React.useState<RecognitionModelVariant>("production_exp13")
  const modelVariantRef = React.useRef<RecognitionModelVariant>("production_exp13")
  const [captureState, setCaptureState] = React.useState<WordCaptureState>("idle")
  const captureStateRef = React.useRef<WordCaptureState>("idle")
  const [captureProgress, setCaptureProgress] = React.useState(0)

  const rollingBufferRef = React.useRef<
    Array<{ t: number; landmarks: number[][]; hasHand: boolean; disp?: number }>
  >([])
  const wordCaptureFramesRef = React.useRef<
    Array<{ t: number; landmarks: number[][]; hasHand: boolean; disp?: number }>
  >([])
  const wordCaptureStartRef = React.useRef<number>(0)
  const wordAutoStateRef = React.useRef<{
    inStroke: boolean
    frames: Array<{ t: number; landmarks: number[][]; hasHand: boolean; disp?: number }>
    startMs: number
    lastMotionMs: number
    lastHandMs: number
    cooldownUntilMs: number
  }>({
    inStroke: false,
    frames: [],
    startMs: 0,
    lastMotionMs: 0,
    lastHandMs: 0,
    cooldownUntilMs: 0
  })
  const prevWordFramePtsRef = React.useRef<{ slot0: number[] | null; slot1: number[] | null }>({
    slot0: null,
    slot1: null
  })
  const lastInferenceTimeRef = React.useRef(0)
  const lastPredictedGlossRef = React.useRef<string>("")
  const lastCommitTimeRef = React.useRef(0)
  const consecutiveTop1Ref = React.useRef<{ gloss: string; count: number }>({ gloss: "", count: 0 })
  const lastEmittedCharRef = React.useRef<string | null>(null)
  const charHoldCountRef = React.useRef(0)
  const fpsFrameCountRef = React.useRef(0)
  const fpsWindowStartRef = React.useRef<number>(0)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)
  const continuousSessionIdRef = React.useRef<string>("web-session")
  React.useEffect(() => {
    const id =
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID().slice(0, 8)
        : Date.now().toString(36)
    continuousSessionIdRef.current = `web-${id}`
  }, [])
  const continuousPendingFramesRef = React.useRef<Array<{ t: number; landmarks: number[][] }>>([])
  const lastHandActiveMsRef = React.useRef<number>(0)
  const showingFinalizedSentenceRef = React.useRef<boolean>(false)
  const hasUnfinalizedContinuousRef = React.useRef<boolean>(false)
  const continuousFinalizeSentRef = React.useRef<boolean>(false)
  const lastNlpSmoothedKeyRef = React.useRef<string>("")
  const lastAutoSpokenRef = React.useRef<{ text: string; at: number }>({ text: "", at: 0 })
  const speakOutputRef = React.useRef<(overrideText?: unknown, isAuto?: boolean) => Promise<void>>(
    async () => {}
  )

  const resetWordAutoState = React.useCallback((cooldownUntilMs = 0) => {
    wordAutoStateRef.current = {
      inStroke: false,
      frames: [],
      startMs: 0,
      lastMotionMs: 0,
      lastHandMs: 0,
      cooldownUntilMs
    }
    prevWordFramePtsRef.current = { slot0: null, slot1: null }
  }, [])

  const resetContinuousBackendSession = React.useCallback(() => {
    continuousPendingFramesRef.current = []
    lastHandActiveMsRef.current = 0
    showingFinalizedSentenceRef.current = false
    hasUnfinalizedContinuousRef.current = false
    continuousFinalizeSentRef.current = false
    lastNlpSmoothedKeyRef.current = ""
    resetWordAutoState()
    setIsStrokeActive(false)
    fetch("/api/continuous/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: continuousSessionIdRef.current })
    }).catch(() => {})
  }, [resetWordAutoState])

  const handleModeChange = React.useCallback((nextMode: RecognitionMode) => {
    modeRef.current = nextMode
    setMode(nextMode)
    rollingBufferRef.current = []
    wordCaptureFramesRef.current = []
    captureStateRef.current = "idle"
    setCaptureState("idle")
    setCaptureProgress(0)
    lastPredictedGlossRef.current = ""
    consecutiveTop1Ref.current = { gloss: "", count: 0 }
    lastEmittedCharRef.current = null
    charHoldCountRef.current = 0
    setTopCandidates([])
    setActivePrediction(null)
    resetContinuousBackendSession()
  }, [resetContinuousBackendSession])

  const setModelVariant = React.useCallback((nextVariant: RecognitionModelVariant) => {
    modelVariantRef.current = nextVariant
    setModelVariantState(nextVariant)
    consecutiveTop1Ref.current = { gloss: "", count: 0 }
    lastPredictedGlossRef.current = ""
    setTopCandidates([])
    setActivePrediction(null)
    resetContinuousBackendSession()
  }, [resetContinuousBackendSession])

  const recognizedText = React.useMemo(() => {
    if (mode === "Fingerspelling") {
      return fingerspellingWord
    }
    if (smoothedText.trim().length > 0) {
      return smoothedText.trim()
    }
    return glosses.join(" ").trim()
  }, [mode, fingerspellingWord, smoothedText, glosses])

  const recognizedRef = React.useRef({
    mode: initialMode as RecognitionMode,
    text: "",
    glosses: [] as string[],
    confidence: undefined as number | undefined
  })

  React.useEffect(() => {
    recognizedRef.current = {
      mode,
      text: recognizedText,
      glosses:
        mode === "Fingerspelling"
          ? fingerspellingWord.trim().split(/\s+/).filter(Boolean)
          : glosses,
      confidence: activePrediction?.confidence
    }
  }, [mode, recognizedText, fingerspellingWord, glosses, activePrediction])

  const enumerateCameras = React.useCallback(async () => {
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      !navigator.mediaDevices.enumerateDevices
    ) {
      return
    }
    try {
      const list = await navigator.mediaDevices.enumerateDevices()
      const videoInputs = list
        .filter((d) => d.kind === "videoinput")
        .map((d, idx) => ({
          deviceId: d.deviceId || `camera-${idx}`,
          label: d.label?.trim() || `Webcam ${idx + 1}`
        }))
      if (isMountedRef.current) {
        setDevices(videoInputs)
        if (videoInputs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoInputs[0].deviceId)
        }
      }
    } catch {
      // Ignore enumeration rejection before permission grant
    }
  }, [selectedDeviceId])

  const runNlpSmooth = React.useCallback(
    async (
      tokens: string[],
      hypotheses?: Array<Array<{ gloss?: string; confidence?: number }>>
    ) => {
      if (tokens.length < 2) return
      try {
        const res = await fetch("/api/nlp/smooth", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            glosses: tokens,
            ...(hypotheses && hypotheses.length === tokens.length ? { hypotheses } : {})
          })
        })
        if (!res.ok) return
        const data = await res.json()
        if (isMountedRef.current && typeof data.english === "string" && data.english.trim()) {
          setSmoothedText(data.english.trim())
        }
      } catch {
        // Fall back naturally to joined glosses
      }
    },
    []
  )

  const applyFinalizedUtterance = React.useCallback(
    (fin: { glosses?: unknown[]; translation?: { english?: string } }) => {
      const finFormatted = (Array.isArray(fin.glosses) ? fin.glosses : [])
        .map((g) => formatGlossLabel(String(g || "")))
        .filter(Boolean)
      if (finFormatted.length === 0) return

      showingFinalizedSentenceRef.current = true
      hasUnfinalizedContinuousRef.current = false
      lastNlpSmoothedKeyRef.current = ""
      lastPredictedGlossRef.current = finFormatted[finFormatted.length - 1]
      lastCommitTimeRef.current = performance.now()
      setGlosses(finFormatted)
      setIsStrokeActive(false)
      const translated =
        typeof fin.translation?.english === "string" ? fin.translation.english.trim() : ""
      const finalDisplay = translated || finFormatted.join(" ")
      setSmoothedText(finalDisplay)
      emitFeedback({ type: "sentence", text: finalDisplay })
      if (preferencesRef.current.autoSpeakTTS && finalDisplay) {
        void speakOutputRef.current(finalDisplay, true)
      }
    },
    []
  )

  const runBackendInference = React.useCallback(async () => {
    if (isInferencingRef.current || isPausedRef.current || !isStreamingRef.current) return

    const currentMode = modeRef.current
    const buffer = rollingBufferRef.current
    const nowMs = performance.now()
    const windowFrames = buffer.filter((item) => nowMs - item.t <= 2300)
    const activeFrames = windowFrames.filter((item) => item.hasHand)

    let continuousBatch: Array<{ t: number; landmarks: number[][] }> = []
    const shouldFallbackFinalize =
      currentMode === "Continuous" &&
      continuousPendingFramesRef.current.length === 0 &&
      hasUnfinalizedContinuousRef.current &&
      !continuousFinalizeSentRef.current &&
      !showingFinalizedSentenceRef.current &&
      lastHandActiveMsRef.current > 0 &&
      nowMs - lastHandActiveMsRef.current >= 1300

    if (currentMode === "Fingerspelling") {
      if (activeFrames.length < 4) return
    } else if (currentMode === "Word") {
      // Word strokes are dispatched directly on completion by onHandsResults / finishWordCapture
      return
    } else if (!shouldFallbackFinalize) {
      if (continuousPendingFramesRef.current.length === 0) return
      continuousBatch = continuousPendingFramesRef.current.splice(0, 60)
    }

    isInferencingRef.current = true
    const reqStart = performance.now()

    try {
      if (currentMode === "Fingerspelling") {
        const res = await fetch("/api/fingerspelling/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sequence: buffer.slice(-24).map((item) => item.landmarks),
            min_char_prob: 0.2,
            target_len: 24
          })
        })
        if (!res.ok) return
        const data = await res.json()
        const roundTripMs = performance.now() - reqStart
        const latency =
          typeof data.latency_ms === "number" ? data.latency_ms : roundTripMs

        if (!isMountedRef.current || modeRef.current !== "Fingerspelling") return

        const rawTopK = Array.isArray(data.top_k) ? data.top_k : []
        setTopCandidates(
          rawTopK.slice(0, 3).map((item: { char?: string; confidence?: number }) => ({
            gloss: String(item.char || ""),
            rawGloss: String(item.char || ""),
            confidence: cleanConfidencePercentage(item.confidence)
          }))
        )

        if (typeof data.char === "string" && data.char.trim().length > 0) {
          const ch = data.char.trim().toUpperCase()
          const conf = cleanConfidencePercentage(data.confidence)
          reportInferenceTelemetry(latency, conf)
          setActivePrediction({ label: ch, confidence: conf })

          if (ch === lastEmittedCharRef.current) {
            charHoldCountRef.current += 1
            if (charHoldCountRef.current === 2) {
              setFingerspellingWord((prev) => `${prev}${ch}`.slice(-64))
              emitFeedback({ type: "letter", label: ch, confidence: conf, at: handPointRef.current ?? undefined })
            }
          } else {
            lastEmittedCharRef.current = ch
            charHoldCountRef.current = 1
          }
        } else {
          setActivePrediction(null)
          lastEmittedCharRef.current = null
          charHoldCountRef.current = 0
          reportInferenceTelemetry(latency, null)
        }
        return
      }

      if (shouldFallbackFinalize) {
        continuousFinalizeSentRef.current = true
        const finRes = await fetch("/api/continuous/finalize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: continuousSessionIdRef.current,
            model_variant: modelVariantRef.current,
            user_id: preferencesRef.current.userId || "guest",
            personalization_enabled:
              Boolean(preferencesRef.current.userId) && personalizationEnabledRef.current,
            translate: true,
            clear_after: true
          })
        })
        if (finRes.ok && isMountedRef.current && modeRef.current === "Continuous") {
          const finData = await finRes.json()
          applyFinalizedUtterance(finData)
          if (!Array.isArray(finData.glosses) || finData.glosses.length === 0) {
            hasUnfinalizedContinuousRef.current = false
            setIsStrokeActive(false)
          }
        }
        return
      }

      // Continuous mode -> stream incremental (N, 42, 3) frames through kinematic CSLR segmenter + decoder
      const res = await fetch("/api/continuous/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: continuousSessionIdRef.current,
          frames: continuousBatch.map((item) => item.landmarks),
          timestamps_ms: continuousBatch.map((item) => item.t),
          model_variant: modelVariantRef.current,
          user_id: preferencesRef.current.userId || "guest",
          personalization_enabled:
            Boolean(preferencesRef.current.userId) && personalizationEnabledRef.current,
          auto_finalize_on_pause: true,
          auto_translate_on_pause: true,
          top_k: 5
        })
      })
      if (!res.ok) return
      const data = await res.json()
      const latency = performance.now() - reqStart

      if (!isMountedRef.current || modeRef.current !== "Continuous") return

      const inStroke = Boolean(data.in_stroke)
      setIsStrokeActive(inStroke)
      if (inStroke && !continuousInStrokeRef.current) {
        emitFeedback({ type: "strokeStart", at: handPointRef.current ?? undefined })
      }
      continuousInStrokeRef.current = inStroke
      if (inStroke) {
        hasUnfinalizedContinuousRef.current = true
      }

      const committedList = Array.isArray(data.newly_committed) ? data.newly_committed : []
      committedList.forEach((item: { gloss?: string; confidence?: number }) => {
        if (typeof item?.gloss === "string" && item.gloss.trim()) {
          emitFeedback({
            type: "gloss",
            label: formatGlossLabel(item.gloss),
            confidence: normalizeContinuousConfPct(item.confidence),
            at: handPointRef.current ?? undefined
          })
        }
      })
      const liveSource =
        data.live_candidate ||
        (committedList.length > 0 ? committedList[committedList.length - 1] : null)

      if (liveSource && typeof liveSource.gloss === "string" && liveSource.gloss.trim()) {
        const liveGloss = formatGlossLabel(liveSource.gloss)
        const liveConf = normalizeContinuousConfPct(liveSource.confidence)
        const rawTopK = Array.isArray(liveSource.top_k) ? liveSource.top_k : []
        const candidates: TopCandidateItem[] = rawTopK
          .slice(0, 3)
          .map((item: { gloss?: string; confidence?: number }) => ({
            gloss: formatGlossLabel(String(item.gloss || "")),
            rawGloss: String(item.gloss || ""),
            confidence: normalizeContinuousConfPct(item.confidence)
          }))
          .filter((item: TopCandidateItem) => item.gloss.length > 0)

        setTopCandidates(candidates)
        setActivePrediction({ label: liveGloss, confidence: liveConf })
        reportInferenceTelemetry(latency, liveConf)
      } else {
        setActivePrediction(null)
        reportInferenceTelemetry(latency, null)
      }

      if (data.pause_finalized) {
        continuousFinalizeSentRef.current = true
        hasUnfinalizedContinuousRef.current = false
        setIsStrokeActive(false)
        if (data.finalized_utterance) {
          applyFinalizedUtterance(
            data.finalized_utterance as { glosses?: unknown[]; translation?: { english?: string } }
          )
        }
      } else if (Array.isArray(data.active_glosses) && data.active_glosses.length > 0) {
        showingFinalizedSentenceRef.current = false
        hasUnfinalizedContinuousRef.current = true
        const activeFormatted = data.active_glosses
          .map((g: unknown) => formatGlossLabel(String(g || "")))
          .filter(Boolean)

        if (activeFormatted.length > 0) {
          lastPredictedGlossRef.current = activeFormatted[activeFormatted.length - 1]
          setGlosses(activeFormatted)
          const activeKey = activeFormatted.join("|")
          if (activeFormatted.length === 1) {
            lastNlpSmoothedKeyRef.current = activeKey
            setSmoothedText(activeFormatted[0])
          } else if (activeKey !== lastNlpSmoothedKeyRef.current) {
            lastNlpSmoothedKeyRef.current = activeKey
            setSmoothedText(activeFormatted.join(" "))
            const activeHyps = Array.isArray(data.active_hypotheses)
              ? data.active_hypotheses
              : undefined
            void runNlpSmooth(activeFormatted, activeHyps)
          }
        }
      }
    } catch {
      // Backend might be offline or mid-restart; UI stays clean without fake output
    } finally {
      isInferencingRef.current = false
    }
  }, [applyFinalizedUtterance, reportInferenceTelemetry, runNlpSmooth])

  const captureTimeoutRef = React.useRef<number | null>(null)

  const trimActiveWordFrames = React.useCallback(
    (captured: Array<{ landmarks: number[][]; hasHand: boolean }>): number[][][] => {
      const firstHand = captured.findIndex((f) => f.hasHand)
      if (firstHand < 0) return captured.map((f) => f.landmarks)
      let lastHand = firstHand
      for (let i = captured.length - 1; i >= firstHand; i--) {
        if (captured[i].hasHand) {
          lastHand = i
          break
        }
      }
      const start = Math.max(0, firstHand - 1)
      const end = Math.min(captured.length - 1, lastHand + 1)
      const sliced = captured.slice(start, end + 1)
      if (sliced.length <= 40) {
        return sliced.map((f) => f.landmarks)
      }
      const sampled32: number[][][] = []
      for (let i = 0; i < 32; i++) {
        const idx = Math.min(sliced.length - 1, Math.round((i * (sliced.length - 1)) / 31))
        sampled32.push(sliced[idx].landmarks)
      }
      return sampled32
    },
    []
  )

  const predictWordSequence = React.useCallback(
    async (
      captured: Array<{ t: number; landmarks: number[][]; hasHand: boolean; disp?: number }>,
      source: "manual" | "auto"
    ) => {
      if (isInferencingRef.current) return
      const activeCount = captured.filter((f) => f.hasHand).length

      if (source === "manual") {
        if (captured.length < 10 || activeCount < 3) {
          toast.info("No complete hand sign captured. Keep hand inside frame and try again.")
          return
        }
      } else if (activeCount < 5 || captured.length < 5) {
        return
      }

      isInferencingRef.current = true
      try {
        const sequence = trimActiveWordFrames(captured)
        const reqStart = performance.now()
        const res = await fetch("/api/predict/sequence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sequence,
            model_variant: modelVariantRef.current,
            top_k: 5,
            mode: "word",
            user_id: preferencesRef.current.userId || "guest",
            personalization_enabled:
              Boolean(preferencesRef.current.userId) && personalizationEnabledRef.current
          })
        })
        if (!res.ok) return
        const data = await res.json()
        const latency =
          typeof data.latency_ms === "number" && data.latency_ms > 0
            ? data.latency_ms
            : performance.now() - reqStart

        if (!isMountedRef.current || modeRef.current !== "Word") return
        const rawGloss = String(data.predicted_gloss || data.gloss || "").trim()
        if (!rawGloss) return

        const upperGloss = rawGloss.toUpperCase()
        if (
          upperGloss === "NO_SIGN" ||
          upperGloss === "_NO_SIGN" ||
          upperGloss === "<BLANK>" ||
          Boolean(data.is_no_sign)
        ) {
          reportInferenceTelemetry(latency, null)
          if (source === "manual") {
            setActivePrediction(null)
            setTopCandidates([])
            toast.info("No distinct ASL sign detected in clip. Perform a clear sign and retry.")
            emitFeedback({ type: "noSign" })
          }
          return
        }

        const conf = cleanConfidencePercentage(data.confidence)
        const minAcceptConf = source === "auto" ? 10 : 5
        if (source === "auto" && conf < minAcceptConf) {
          reportInferenceTelemetry(latency, null)
          return
        }

        const formatted = formatGlossLabel(rawGloss)
        const rawTopK = Array.isArray(data.top_k) ? data.top_k : []
        const candidates: TopCandidateItem[] = rawTopK
          .slice(0, 3)
          .map((item: { gloss?: string; confidence?: number }) => ({
            gloss: formatGlossLabel(String(item.gloss || "")),
            rawGloss: String(item.gloss || ""),
            confidence: cleanConfidencePercentage(item.confidence)
          }))
          .filter((item: TopCandidateItem) => item.gloss.length > 0)

        setTopCandidates(candidates)
        reportInferenceTelemetry(latency, conf)
        setActivePrediction({ label: formatted, confidence: conf })

        if (conf >= minAcceptConf) {
          lastPredictedGlossRef.current = formatted
          lastCommitTimeRef.current = performance.now()
          emitFeedback({ type: "gloss", label: formatted, confidence: conf, at: handPointRef.current ?? undefined })
          setGlosses((prev) => {
            const next = [...prev, formatted]
            if (next.length >= 2) {
              void runNlpSmooth(next)
            } else {
              setSmoothedText(formatted)
            }
            return next
          })
          if (preferencesRef.current.autoSpeakTTS) {
            void speakOutputRef.current(formatted, true)
          }
          if (source === "manual") {
            toast.success(`Recognized "${formatted}" (${conf.toFixed(1)}%)`)
          }
        }
      } catch {
        if (source === "manual") {
          toast.error("Backend unreachable while analyzing sign")
          emitFeedback({ type: "error", kind: "backend" })
        }
      } finally {
        isInferencingRef.current = false
      }
    },
    [reportInferenceTelemetry, runNlpSmooth, trimActiveWordFrames]
  )

  const finishWordCapture = React.useCallback(async () => {
    if (captureStateRef.current !== "recording") return
    if (captureTimeoutRef.current !== null) {
      window.clearTimeout(captureTimeoutRef.current)
      captureTimeoutRef.current = null
    }
    captureStateRef.current = "analyzing"
    setCaptureState("analyzing")
    setCaptureProgress(100)

    const captured = [...wordCaptureFramesRef.current]
    try {
      await predictWordSequence(captured, "manual")
    } finally {
      if (isMountedRef.current) {
        captureStateRef.current = "idle"
        setCaptureState("idle")
        setCaptureProgress(0)
        resetWordAutoState(performance.now() + 500)
      }
    }
  }, [predictWordSequence, resetWordAutoState])

  const startWordCapture = React.useCallback(() => {
    if (!isStreamingRef.current || isPausedRef.current) {
      toast.info("Camera must be streaming to capture a sign")
      return
    }
    if (captureStateRef.current !== "idle") return
    resetWordAutoState()
    setIsStrokeActive(false)
    wordCaptureFramesRef.current = []
    wordCaptureStartRef.current = performance.now()
    captureStateRef.current = "recording"
    setCaptureState("recording")
    setCaptureProgress(4)
    if (captureTimeoutRef.current !== null) {
      window.clearTimeout(captureTimeoutRef.current)
    }
    captureTimeoutRef.current = window.setTimeout(() => {
      void finishWordCapture()
    }, 2250)
  }, [finishWordCapture, resetWordAutoState])

  const selectCandidate = React.useCallback(
    (gloss: string, confidence: number) => {
      if (!gloss) return
      if (modeRef.current === "Fingerspelling") {
        const ch = gloss.trim().toUpperCase()
        setFingerspellingWord((prev) => `${prev.slice(0, -1)}${ch}`.slice(-64))
        setActivePrediction({ label: ch, confidence })
        return
      }
      setActivePrediction({ label: gloss, confidence })
      lastPredictedGlossRef.current = gloss
      setGlosses((prev) => {
        const next = prev.length > 0 ? [...prev.slice(0, -1), gloss] : [gloss]
        if (next.length >= 2) {
          void runNlpSmooth(next)
        } else {
          setSmoothedText(next[0] || "")
        }
        return next
      })
    },
    [runNlpSmooth]
  )

  const onHandsResults = React.useCallback(
    (results: SharedHandsResults) => {
      const hasHands = Boolean(
        results?.multiHandLandmarks && results.multiHandLandmarks.length > 0
      )
      setIsHandDetected(hasHands)

      {
        const nowMs = performance.now()
        const firstHand = hasHands ? results.multiHandLandmarks?.[0] : undefined
        if (firstHand && firstHand.length >= 18) {
          const palm = [0, 5, 9, 13, 17]
          handPointRef.current = {
            x: palm.reduce((a, i) => a + (firstHand[i]?.x ?? 0), 0) / palm.length,
            y: palm.reduce((a, i) => a + (firstHand[i]?.y ?? 0), 0) / palm.length
          }
          lastHandSeenMsRef.current = nowMs
          if (!handShownRef.current) {
            handShownRef.current = true
            emitFeedback({ type: "handFound", at: handPointRef.current })
          }
        } else if (handShownRef.current && nowMs - lastHandSeenMsRef.current > 450) {
          handShownRef.current = false
          emitFeedback({ type: "handLost" })
        }
      }

      const frameLandmarks: number[][] = Array.from({ length: 42 }, () => [0, 0, 0])
      const nowTs = performance.now()

      if (hasHands && results.multiHandLandmarks) {
        const detectedHands = results.multiHandLandmarks.map((lms: LandmarkPoint[]) => ({
          wristX: lms[0]?.x ?? 0,
          wristY: lms[0]?.y ?? 0,
          landmarks: lms
        }))

        if (detectedHands.length >= 2) {
          detectedHands.sort((a, b) => a.wristX - b.wristX)
          detectedHands[0].landmarks.forEach((pt: LandmarkPoint, i: number) => {
            if (i < 21) frameLandmarks[i] = [pt.x, pt.y, pt.z || 0]
          })
          detectedHands[1].landmarks.forEach((pt: LandmarkPoint, i: number) => {
            if (i < 21) frameLandmarks[21 + i] = [pt.x, pt.y, pt.z || 0]
          })
          prevHandWristsRef.current = {
            slot0: [detectedHands[0].wristX, detectedHands[0].wristY],
            slot1: [detectedHands[1].wristX, detectedHands[1].wristY],
            lastSeenMs: nowTs
          }
        } else {
          const single = detectedHands[0]
          const prev = prevHandWristsRef.current
          let useSlot1 = false
          if (prev.slot0 && prev.slot1 && nowTs - prev.lastSeenMs < 480) {
            const d0 = Math.hypot(single.wristX - prev.slot0[0], single.wristY - prev.slot0[1])
            const d1 = Math.hypot(single.wristX - prev.slot1[0], single.wristY - prev.slot1[1])
            if (d1 < d0 * 0.72) {
              useSlot1 = true
            }
          }
          const offset = useSlot1 ? 21 : 0
          single.landmarks.forEach((pt: LandmarkPoint, i: number) => {
            if (i < 21) frameLandmarks[offset + i] = [pt.x, pt.y, pt.z || 0]
          })
          prevHandWristsRef.current = {
            slot0: useSlot1 ? prev.slot0 : [single.wristX, single.wristY],
            slot1: useSlot1 ? [single.wristX, single.wristY] : null,
            lastSeenMs: nowTs
          }
        }
      }

      let disp = 0
      const extractPts = (offset: number): number[] | null => {
        const w = frameLandmarks[offset]
        const idx = frameLandmarks[offset + 8]
        const mid = frameLandmarks[offset + 12]
        if (!w || (w[0] === 0 && w[1] === 0 && w[2] === 0)) return null
        return [w[0], w[1], idx[0], idx[1], mid[0], mid[1]]
      }
      const p0 = extractPts(0)
      const p1 = extractPts(21)
      const prevPts = prevWordFramePtsRef.current
      if (p0 && prevPts.slot0) {
        for (let k = 0; k < 6; k += 2) {
          disp = Math.max(disp, Math.hypot(p0[k] - prevPts.slot0[k], p0[k + 1] - prevPts.slot0[k + 1]))
        }
      } else if (p0 && !prevPts.slot0) {
        disp = Math.max(disp, 0.025)
      }
      if (p1 && prevPts.slot1) {
        for (let k = 0; k < 6; k += 2) {
          disp = Math.max(disp, Math.hypot(p1[k] - prevPts.slot1[k], p1[k + 1] - prevPts.slot1[k + 1]))
        }
      } else if (p1 && !prevPts.slot1) {
        disp = Math.max(disp, 0.025)
      }
      prevWordFramePtsRef.current = { slot0: p0, slot1: p1 }

      const entry = { t: nowTs, landmarks: frameLandmarks, hasHand: hasHands, disp }
      rollingBufferRef.current.push(entry)
      if (rollingBufferRef.current.length > 80) {
        rollingBufferRef.current.shift()
      }

      // Draw luminous amber hand-skeleton glow onto overlay canvas
      const canvas = canvasRef.current
      const video = videoRef.current
      if (canvas) {
        if (video && video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }
        const ctx = canvas.getContext("2d")
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          if (hasHands && results.multiHandLandmarks) {
            const CONNECTIONS = [
              [0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8],
              [5, 9], [9, 10], [10, 11], [11, 12],
              [9, 13], [13, 14], [14, 15], [15, 16],
              [13, 17], [17, 18], [18, 19], [19, 20],
              [0, 17]
            ]
            const energized =
              wordAutoStateRef.current.inStroke ||
              captureStateRef.current === "recording" ||
              continuousInStrokeRef.current
            const TIPS = new Set([4, 8, 12, 16, 20])
            ctx.save()
            ctx.shadowColor = energized ? "rgba(255, 166, 109, 0.95)" : "rgba(255, 138, 61, 0.75)"
            ctx.shadowBlur = energized ? 18 : 10
            ctx.strokeStyle = energized ? "rgba(255, 166, 109, 1)" : "rgba(255, 138, 61, 0.92)"
            ctx.lineWidth = energized ? 3.4 : 2.6
            ctx.lineCap = "round"
            ctx.lineJoin = "round"
            results.multiHandLandmarks.forEach((hand) => {
              CONNECTIONS.forEach(([s, e]) => {
                const p1 = hand[s]
                const p2 = hand[e]
                if (p1 && p2) {
                  ctx.beginPath()
                  ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height)
                  ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height)
                  ctx.stroke()
                }
              })
              hand.forEach((pt, idx) => {
                const tip = TIPS.has(idx)
                ctx.beginPath()
                ctx.arc(pt.x * canvas.width, pt.y * canvas.height, idx === 0 ? 5.2 : tip ? 4.8 : 3.4, 0, 2 * Math.PI)
                ctx.fillStyle = idx === 0 ? "#FB6A18" : tip ? (energized ? "#FFFFFF" : "#FFE7B3") : "#FFF8EB"
                ctx.fill()
                if (tip) {
                  ctx.beginPath()
                  ctx.arc(pt.x * canvas.width, pt.y * canvas.height, energized ? 9 : 7, 0, 2 * Math.PI)
                  ctx.strokeStyle = energized ? "rgba(255, 166, 109, 0.6)" : "rgba(255, 138, 61, 0.35)"
                  ctx.lineWidth = 1.4
                  ctx.stroke()
                  ctx.strokeStyle = energized ? "rgba(255, 166, 109, 1)" : "rgba(255, 138, 61, 0.92)"
                  ctx.lineWidth = energized ? 3.4 : 2.6
                }
              })
            })
            ctx.restore()
          }
        }
      }

      // If in Word isolated capture mode, record frames cleanly over 2200ms
      if (captureStateRef.current === "recording") {
        wordCaptureFramesRef.current.push(entry)
        const elapsed = nowTs - wordCaptureStartRef.current
        const pct = Math.min(100, Math.round((elapsed / 2200) * 100))
        setCaptureProgress(pct)
        emitFeedback({ type: "captureProgress", progress: pct })
        if (elapsed >= 2200) {
          void finishWordCapture()
        }
        return
      }

      if (hasHands) {
        lastHandActiveMsRef.current = nowTs
        continuousFinalizeSentRef.current = false
      }
      if (
        modeRef.current === "Continuous" &&
        (hasHands || (lastHandActiveMsRef.current > 0 && nowTs - lastHandActiveMsRef.current <= 1150))
      ) {
        continuousPendingFramesRef.current.push({ t: nowTs, landmarks: frameLandmarks })
        if (continuousPendingFramesRef.current.length > 120) {
          continuousPendingFramesRef.current = continuousPendingFramesRef.current.slice(-120)
        }
      }

      if (modeRef.current === "Word" && captureStateRef.current === "idle") {
        const w = wordAutoStateRef.current
        if (!w.inStroke) {
          if (hasHands && disp >= 0.014 && nowTs >= w.cooldownUntilMs && !isInferencingRef.current) {
            const leadIn = rollingBufferRef.current.slice(-4, -1).filter((f) => f.hasHand)
            w.inStroke = true
            w.frames = [...leadIn, entry]
            w.startMs = nowTs
            w.lastMotionMs = nowTs
            w.lastHandMs = nowTs
            setIsStrokeActive(true)
            emitFeedback({ type: "strokeStart", at: handPointRef.current ?? undefined })
          }
        } else {
          w.frames.push(entry)
          if (hasHands) {
            w.lastHandMs = nowTs
            if (disp >= 0.011) {
              w.lastMotionMs = nowTs
            }
          }
          const activeCount = w.frames.reduce((acc, f) => acc + (f.hasHand ? 1 : 0), 0)
          const handDroppedPause = !hasHands && nowTs - w.lastHandMs >= 380 && activeCount >= 5
          const holdPause =
            hasHands &&
            nowTs - w.lastMotionMs >= 460 &&
            nowTs - w.startMs >= 520 &&
            activeCount >= 6
          const maxLenReached = w.frames.length >= 64 || nowTs - w.startMs >= 1800

          if (!isInferencingRef.current && (handDroppedPause || holdPause || maxLenReached)) {
            const cutoffMs = Math.max(w.lastMotionMs, w.lastHandMs) + 120
            const strokeFrames = w.frames.filter((f) => f.t <= cutoffMs)
            const finalFrames = strokeFrames.length >= 5 ? strokeFrames : [...w.frames]
            w.inStroke = false
            w.frames = []
            w.cooldownUntilMs = nowTs + 650
            setIsStrokeActive(false)
            void predictWordSequence(finalFrames, "auto")
          } else if (w.frames.length > 72) {
            w.inStroke = false
            w.frames = []
            setIsStrokeActive(false)
          }
        }
      }

      const now = performance.now()
      const intervalMs =
        modeRef.current === "Fingerspelling"
          ? 260
          : modeRef.current === "Continuous"
            ? 210
            : 480
      if (now - lastInferenceTimeRef.current >= intervalMs && !isInferencingRef.current) {
        lastInferenceTimeRef.current = now
        void runBackendInference()
      }
    },
    [finishWordCapture, predictWordSequence, runBackendInference]
  )

  const initMediaPipe = React.useCallback(async () => {
    if (typeof window === "undefined") return

    try {
      setMediaPipeHandsListener(onHandsResults as unknown as (r: SharedHandsResults) => void)
      await getSharedMediaPipeHands({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      })
      handsDetectorRef.current = true
    } catch {
      // MediaPipe WASM initialization unavailable
    }
  }, [onHandsResults])

  const stopTracks = React.useCallback(() => {
    isStreamingRef.current = false
    setIsStrokeActive(false)
    continuousPendingFramesRef.current = []
    lastHandActiveMsRef.current = 0
    resetWordAutoState()
    if (captureTimeoutRef.current !== null) {
      window.clearTimeout(captureTimeoutRef.current)
      captureTimeoutRef.current = null
    }
    clearMediaPipeHandsListener(onHandsResults as unknown as (r: SharedHandsResults) => void)
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      mediaStreamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    setIsHandDetected(false)
    setFps(null)
  }, [onHandsResults, resetWordAutoState])

  const startLoop = React.useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
    }
    fpsFrameCountRef.current = 0
    fpsWindowStartRef.current = performance.now()

    const tick = async () => {
      if (!isMountedRef.current || !isStreamingRef.current || isPausedRef.current) {
        return
      }

      const video = videoRef.current
      if (video && video.readyState >= 2 && video.videoWidth > 0) {
        fpsFrameCountRef.current += 1
        const now = performance.now()
        if (now - fpsWindowStartRef.current >= 1000) {
          const measured = Math.round(
            (fpsFrameCountRef.current * 1000) / (now - fpsWindowStartRef.current)
          )
          setFps(measured > 0 ? measured : null)
          fpsFrameCountRef.current = 0
          fpsWindowStartRef.current = now
        }

        if (handsDetectorRef.current && !isProcessingFrameRef.current) {
          isProcessingFrameRef.current = true
          try {
            await sendVideoFrameToSharedHands(video)
          } catch {
            // Skip frame on transient error
          } finally {
            isProcessingFrameRef.current = false
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(tick)
    }

    animFrameRef.current = requestAnimationFrame(tick)
  }, [])

  const startCamera = React.useCallback(
    async (deviceIdOverride?: string) => {
      if (
        typeof navigator === "undefined" ||
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {
        setCameraError({
          code: "unsupported",
          message: "Camera access requires a supported browser on localhost or HTTPS."
        })
        setIsStreaming(false)
        setFps(null)
        return
      }

      try {
        setCameraError(null)
        stopTracks()

        let hasVideoInput = false
        if (navigator.mediaDevices.enumerateDevices) {
          try {
            const existingDevices = await Promise.race([
              navigator.mediaDevices.enumerateDevices(),
              new Promise<MediaDeviceInfo[]>((resolve) => {
                window.setTimeout(() => resolve([]), 600)
              })
            ])
            hasVideoInput = existingDevices.some((d) => d.kind === "videoinput")
          } catch {
            hasVideoInput = false
          }
        }

        const devId = deviceIdOverride ?? selectedDeviceId
        const constraints: MediaStreamConstraints = {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
            ...(devId ? { deviceId: { ideal: devId } } : {})
          },
          audio: false
        }

        let didRaceTimeout = false
        let timeoutId: number | null = null
        const streamPromise = navigator.mediaDevices.getUserMedia(constraints)
        streamPromise
          .then((lateStream) => {
            if (didRaceTimeout || !isMountedRef.current) {
              lateStream.getTracks().forEach((t) => t.stop())
            }
          })
          .catch(() => {})

        let stream: MediaStream
        try {
          stream = await Promise.race([
            streamPromise,
            new Promise<MediaStream>((_, reject) => {
              timeoutId = window.setTimeout(
                () => {
                  didRaceTimeout = true
                  reject(
                    Object.assign(
                      new Error(
                        hasVideoInput
                          ? "Camera request timed out. Grant permission and retry."
                          : "No webcam device detected on this machine."
                      ),
                      { name: hasVideoInput ? "PermissionDeniedError" : "NotFoundError" }
                    )
                  )
                },
                hasVideoInput ? 12000 : 1200
              )
            })
          ])
        } finally {
          if (timeoutId !== null) {
            window.clearTimeout(timeoutId)
          }
        }
        if (!isMountedRef.current) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }

        mediaStreamRef.current = stream
        isStreamingRef.current = true
        isPausedRef.current = false
        setIsStreaming(true)
        setIsPaused(false)

        // Refresh camera labels now that permission was granted
        const trackSettings = stream.getVideoTracks()[0]?.getSettings()
        if (trackSettings?.deviceId) {
          setSelectedDeviceId(trackSettings.deviceId)
        }
        void enumerateCameras()

        if (videoRef.current) {
          const videoEl = videoRef.current
          const triggerPlay = () => {
            videoEl.play().catch(() => {})
          }
          videoEl.onloadedmetadata = triggerPlay
          videoEl.onloadeddata = triggerPlay
          videoEl.srcObject = stream
          triggerPlay()
        }

        void initMediaPipe()
        startLoop()
      } catch (err: unknown) {
        if (!isMountedRef.current) return
        stopTracks()
        const errObj = err as { name?: string; message?: string } | null
        const name = errObj?.name || ""
        let parsed: CameraErrorState
        if (name === "NotAllowedError" || name === "PermissionDeniedError") {
          parsed = {
            code: "permission-denied",
            message:
              "Camera permission is blocked. Enable webcam access in browser settings and retry."
          }
        } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
          parsed = {
            code: "not-found",
            message: "No webcam device detected on this machine."
          }
        } else if (name === "NotReadableError" || name === "TrackStartError") {
          parsed = {
            code: "device-busy",
            message: "Camera is currently in use by another application."
          }
        } else {
          parsed = {
            code: "unknown",
            message: errObj?.message || "Unable to start webcam stream."
          }
        }
        setCameraError(parsed)
      }
    },
    [selectedDeviceId, stopTracks, enumerateCameras, initMediaPipe, startLoop]
  )

  React.useEffect(() => {
    isMountedRef.current = true
    const bootTimer = window.setTimeout(() => {
      void enumerateCameras()
      void startCamera()
    }, 0)

    return () => {
      window.clearTimeout(bootTimer)
      isMountedRef.current = false
      const snap = recognizedRef.current
      if (snap.text.trim().length > 0) {
        recordSession({
          mode: snap.mode,
          text: snap.text,
          glosses: snap.glosses,
          confidence: snap.confidence
        })
      }
      stopTracks()
    }
    // Mount/unmount lifecycle only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const togglePause = React.useCallback(() => {
    if (!isStreamingRef.current && !isPausedRef.current) {
      void startCamera()
      return
    }
    if (isPausedRef.current) {
      isPausedRef.current = false
      setIsPaused(false)
      if (!mediaStreamRef.current) {
        void startCamera()
      } else {
        isStreamingRef.current = true
        setIsStreaming(true)
        videoRef.current?.play().catch(() => {})
        startLoop()
      }
      toast.info("Recognition resumed")
    } else {
      isPausedRef.current = true
      isStreamingRef.current = false
      setIsPaused(true)
      setIsStreaming(false)
      setFps(null)
      videoRef.current?.pause()
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
        animFrameRef.current = null
      }
      const snap = recognizedRef.current
      if (snap.text.trim().length > 0) {
        recordSession({
          mode: snap.mode,
          text: snap.text,
          glosses: snap.glosses,
          confidence: snap.confidence
        })
      }
      toast.info("Recognition paused")
    }
  }, [startCamera, startLoop, recordSession])

  const selectCamera = React.useCallback(
    (newDeviceId: string) => {
      setSelectedDeviceId(newDeviceId)
      void startCamera(newDeviceId)
    },
    [startCamera]
  )

  const clearOutput = React.useCallback(() => {
    const snap = recognizedRef.current
    if (snap.text.trim().length > 0) {
      recordSession({
        mode: snap.mode,
        text: snap.text,
        glosses: snap.glosses,
        confidence: snap.confidence
      })
    }
    setGlosses([])
    setSmoothedText("")
    setFingerspellingWord("")
    setActivePrediction(null)
    setTopCandidates([])
    setIsStrokeActive(false)
    hasUnfinalizedContinuousRef.current = false
    lastPredictedGlossRef.current = ""
    consecutiveTop1Ref.current = { gloss: "", count: 0 }
    lastEmittedCharRef.current = null
    charHoldCountRef.current = 0
    lastAutoSpokenRef.current = { text: "", at: 0 }
    resetContinuousBackendSession()
    toast.info("Cleared recognized text")
  }, [recordSession, resetContinuousBackendSession])

  const copyOutput = React.useCallback(async () => {
    if (!recognizedText) {
      toast.info("Start signing first to copy text")
      return
    }
    try {
      await navigator.clipboard.writeText(recognizedText)
      toast.success("Copied recognized text")
    } catch {
      toast.error("Unable to access clipboard")
    }
  }, [recognizedText])

  const appendFingerspellingSpace = React.useCallback(() => {
    setFingerspellingWord((prev) => (prev.endsWith(" ") || prev.length === 0 ? prev : `${prev} `.slice(-64)))
    lastEmittedCharRef.current = null
    charHoldCountRef.current = 0
  }, [])

  const backspaceFingerspelling = React.useCallback(() => {
    setFingerspellingWord((prev) => prev.slice(0, -1))
    lastEmittedCharRef.current = null
    charHoldCountRef.current = 0
  }, [])

  const speakOutput = React.useCallback(async (overrideText?: unknown, isAuto?: boolean) => {
    const targetText = (typeof overrideText === "string" ? overrideText : recognizedText).trim()
    if (!targetText) {
      toast.info("Start signing first to hear speech output")
      return
    }
    const autoTriggered = Boolean(isAuto) || typeof overrideText === "string"
    const nowAt = Date.now()
    if (
      autoTriggered &&
      lastAutoSpokenRef.current.text === targetText &&
      nowAt - lastAutoSpokenRef.current.at < 4500
    ) {
      return
    }
    if (autoTriggered) {
      lastAutoSpokenRef.current = { text: targetText, at: nowAt }
    }
    if (isSpeaking && typeof overrideText !== "string") return
    setIsSpeaking(true)

    const fallbackWebSpeech = (text: string) => {
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        try {
          window.speechSynthesis.cancel()
          const u = new SpeechSynthesisUtterance(text)
          u.onend = () => setIsSpeaking(false)
          u.onerror = () => setIsSpeaking(false)
          window.speechSynthesis.speak(u)
        } catch {
          setIsSpeaking(false)
        }
      } else {
        setIsSpeaking(false)
      }
    }

    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: targetText,
          voice: preferencesRef.current.ttsVoice || "af_sarah"
        })
      })
      if (res.ok) {
        const data = await res.json()
        const src = data.audio_url || data.audio_data
        if (src) {
          audioRef.current?.pause()
          const audio = new Audio(src)
          audioRef.current = audio
          audio.onended = () => setIsSpeaking(false)
          audio.onerror = () => fallbackWebSpeech(targetText)
          await audio.play()
          return
        }
      }
      fallbackWebSpeech(targetText)
    } catch {
      fallbackWebSpeech(targetText)
    }
  }, [recognizedText, isSpeaking])

  React.useEffect(() => {
    speakOutputRef.current = speakOutput
  }, [speakOutput])

  return {
    videoRef,
    canvasRef,
    mode,
    setMode: handleModeChange,
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
    retryCamera: () => startCamera(),
    togglePause,
    clearOutput,
    copyOutput,
    speakOutput
  }
}
