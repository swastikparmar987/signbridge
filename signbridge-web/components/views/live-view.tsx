"use client"

import * as React from "react"
import { motion } from "framer-motion"
import {
  Camera,
  CameraOff,
  Settings,
  Volume2,
  X,
  Copy,
  Check,
  Hand,
  Menu,
  Type,
  Loader2,
  Timer,
  Sparkles
} from "lucide-react"
import { toast } from "sonner"
import { useSignBridgeStore } from "@/lib/signbridge-store"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"
import {
  clearMediaPipeHandsListener,
  getSharedMediaPipeHands,
  MediaPipeHandsResults,
  MediaPipeLandmark,
  sendVideoFrameToSharedHands,
  setMediaPipeHandsListener
} from "@/lib/mediapipe-hands"

interface TopKItem {
  gloss: string
  rawGloss: string
  confidence: number // 0.0 to 100.0 percentage
}

interface PredictionState {
  gloss: string
  confidence: number
  topK: TopKItem[]
}

interface TimestampedFrame {
  landmarks: number[][]
  t: number
  hasHand: boolean
}

type ModelVariant = "production_exp13" | "demo_50"
type CaptureState = "idle" | "recording" | "analyzing"

interface LiveRecognitionViewProps {
  onOpenMobileMenu?: () => void
  onSwitchToFingerspelling?: () => void
}

// Well-known ASL compound phrases mapped to clean Title Case
const KNOWN_GLOSS_DICTIONARY: Record<string, string> = {
  ILOVEYOU: "I Love You",
  THANKYOU: "Thank You",
  GOODMORNING: "Good Morning",
  GOODNIGHT: "Good Night",
  HOWAREYOU: "How Are You",
  SEEYOULATER: "See You Later",
  NICETOMEETYOU: "Nice To Meet You",
  DONTKNOW: "Don't Know",
  DONTWANT: "Don't Want",
  PLEASE: "Please",
  HELLO: "Hello",
  HELP: "Help",
  GOOD: "Good",
  YES: "Yes",
  NO: "No",
  FRIEND: "Friend",
  DEAF: "Deaf",
  HEARING: "Hearing",
  NAME: "Name",
  SIGN: "Sign",
  LANGUAGE: "Language",
  WELCOME: "Welcome",
}

function cleanConfidencePercentage(rawConf: unknown): number {
  const val = typeof rawConf === "number" ? rawConf : parseFloat(String(rawConf ?? 0)) || 0
  if (!Number.isFinite(val) || val <= 0) return 0
  return Math.min(99.9, Math.max(0.1, Number(val.toFixed(1))))
}

function formatGlossTitleCase(gloss: string): string {
  if (!gloss) return ""
  const upper = gloss.replace(/[^A-Za-z0-9]/g, "").toUpperCase()
  if (KNOWN_GLOSS_DICTIONARY[upper]) {
    return KNOWN_GLOSS_DICTIONARY[upper]
  }

  // Strip trailing dataset numbers (e.g. GLOVES1 -> GLOVES, SHAVE4 -> SHAVE)
  const cleaned = gloss.replace(/\d+$/, "")
  const cleanedUpper = cleaned.replace(/[^A-Za-z0-9]/g, "").toUpperCase()
  if (KNOWN_GLOSS_DICTIONARY[cleanedUpper]) {
    return KNOWN_GLOSS_DICTIONARY[cleanedUpper]
  }

  return cleaned
    .toLowerCase()
    .split(/[\s_-]+/)
    .filter((w) => w.length > 0)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
}

function resampleTimestampedSequence(frames: TimestampedFrame[], targetLen = 32): number[][][] {
  if (frames.length === 0) return []
  if (frames.length === 1) {
    return Array.from({ length: targetLen }, () => frames[0].landmarks)
  }
  const sampled: number[][][] = []
  for (let i = 0; i < targetLen; i++) {
    const idx = Math.min(
      frames.length - 1,
      Math.round((i * (frames.length - 1)) / (targetLen - 1))
    )
    sampled.push(frames[idx].landmarks)
  }
  return sampled
}

function drawHandSkeletons(
  handsLandmarks: MediaPipeLandmark[][],
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number
) {
  const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
  ]

  handsLandmarks.forEach((landmarks) => {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)"
    ctx.lineWidth = 2.8
    ctx.shadowBlur = 4
    ctx.shadowColor = "#FF7900"
    ctx.lineCap = "round"
    ctx.lineJoin = "round"

    CONNECTIONS.forEach(([start, end]) => {
      const p1 = landmarks[start]
      const p2 = landmarks[end]
      if (p1 && p2) {
        ctx.beginPath()
        ctx.moveTo(p1.x * w, p1.y * h)
        ctx.lineTo(p2.x * w, p2.y * h)
        ctx.stroke()
      }
    })

    landmarks.forEach((pt, idx) => {
      const px = pt.x * w
      const py = pt.y * h
      const isWrist = idx === 0
      const isFingertip = [4, 8, 12, 16, 20].includes(idx)

      ctx.beginPath()
      ctx.arc(px, py, isWrist ? 6 : isFingertip ? 5 : 4, 0, 2 * Math.PI)
      ctx.fillStyle = "#FFFFFF"
      ctx.shadowColor = "#FF7900"
      ctx.shadowBlur = 8
      ctx.fill()

      ctx.beginPath()
      ctx.arc(px, py, isWrist ? 3.5 : isFingertip ? 2.8 : 2.2, 0, 2 * Math.PI)
      ctx.fillStyle = "#FFFFFF"
      ctx.shadowBlur = 0
      ctx.fill()

      if (isFingertip) {
        ctx.beginPath()
        ctx.arc(px, py, 4.6, 0, 2 * Math.PI)
        ctx.strokeStyle = "#FF7900"
        ctx.lineWidth = 1.2
        ctx.stroke()
      }
    })
  })
}

export function LiveRecognitionView({ onOpenMobileMenu, onSwitchToFingerspelling }: LiveRecognitionViewProps = {}) {
  const { preferences } = useSignBridgeStore()
  const preferencesRef = React.useRef(preferences)
  React.useEffect(() => {
    preferencesRef.current = preferences
  }, [preferences])

  // WebRTC Camera State
  const [isCameraOn, setIsCameraOn] = React.useState(true)
  const isCameraRunningRef = React.useRef(true)
  const [videoDevices, setVideoDevices] = React.useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = React.useState<string>("")
  const [showCameraMenu, setShowCameraMenu] = React.useState(false)

  const videoRef = React.useRef<HTMLVideoElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const mediaStreamRef = React.useRef<MediaStream | null>(null)
  const animFrameRef = React.useRef<number | null>(null)
  const isMountedRef = React.useRef(true)

  // Shared MediaPipe Hands & Processing Locks
  const handsDetectorRef = React.useRef<boolean>(false)
  const isProcessingRef = React.useRef(false)
  const lastFrameTimeRef = React.useRef(0)
  const rollingBufferRef = React.useRef<TimestampedFrame[]>([]) // timestamped [T, 42, 3] preserving zeros

  // Tracking state
  const [isHandTracked, setIsHandTracked] = React.useState(false)
  const lastHandSeenTimeRef = React.useRef(0)

  // Model & Server Readiness
  const [isModelReady, setIsModelReady] = React.useState(false)
  const [modelVariant, setModelVariant] = React.useState<ModelVariant>("production_exp13")
  const modelVariantRef = React.useRef<ModelVariant>("production_exp13")

  // Real Prediction State (starts honest idle until real inference arrives)
  const [prediction, setPrediction] = React.useState<PredictionState>({
    gloss: "Ready to Sign",
    confidence: 0,
    topK: []
  })

  // Isolated 2.2s Word Capture State
  const [captureState, setCaptureState] = React.useState<CaptureState>("idle")
  const captureStateRef = React.useRef<CaptureState>("idle")
  const captureFramesRef = React.useRef<TimestampedFrame[]>([])
  const captureTimeoutRef = React.useRef<number | null>(null)

  // Inference pacing & consensus stabilization
  const lastInferenceTimeRef = React.useRef(0)
  const isInferencingRef = React.useRef(false)
  const lastPredictedSignRef = React.useRef("")
  const stableCandidateRef = React.useRef("")
  const stableStreakRef = React.useRef(0)

  // Sentence Builder State (chips)
  const [sentenceWords, setSentenceWords] = React.useState<string[]>([])
  const [hasCopied, setHasCopied] = React.useState(false)
  const [isPlayingTTS, setIsPlayingTTS] = React.useState(false)
  const audioPlayerRef = React.useRef<HTMLAudioElement | null>(null)

  React.useEffect(() => {
    modelVariantRef.current = modelVariant
    stableCandidateRef.current = ""
    stableStreakRef.current = 0
  }, [modelVariant])

  React.useEffect(() => {
    captureStateRef.current = captureState
  }, [captureState])

  // 1. Health check & Device enumeration on mount
  React.useEffect(() => {
    isMountedRef.current = true
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => {
        if (isMountedRef.current && (data.status === "healthy" || data.model_loaded)) {
          setIsModelReady(true)
        }
      })
      .catch(() => {
        if (isMountedRef.current) setIsModelReady(false)
      })

    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then((devices) => {
        const videoInputs = devices.filter((d) => d.kind === "videoinput")
        setVideoDevices(videoInputs)
        if (videoInputs.length > 0) {
          setSelectedDeviceId((prev) => prev || videoInputs[0].deviceId)
        }
      }).catch(console.warn)
    }
  }, [])

  // 2. Shared MediaPipe Hands Singleton Initialization & Frame Loop
  const onMediaPipeResultsRef = React.useRef<(results: MediaPipeHandsResults) => void>(() => {})
  const runContinuousInferenceRef = React.useRef<() => Promise<void>>(async () => {})
  const finishWordCaptureRef = React.useRef<() => Promise<void>>(async () => {})
  const stableHandsListener = React.useCallback((results: MediaPipeHandsResults) => {
    onMediaPipeResultsRef.current(results)
  }, [])

  const initMediaPipe = async () => {
    if (typeof window === "undefined") return

    try {
      setMediaPipeHandsListener(stableHandsListener)
      await getSharedMediaPipeHands({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      })
      handsDetectorRef.current = true
      if (isMountedRef.current) setIsModelReady(true)
    } catch (err) {
      console.error("MediaPipe initialization error:", err)
      return
    }

    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
    }

    const processFrame = async () => {
      if (!isCameraRunningRef.current || !isMountedRef.current) return

      const now = performance.now()
      // Pace at ~30 FPS (33ms) and never overlap concurrent WASM sends
      if (now - lastFrameTimeRef.current >= 32) {
        const video = videoRef.current
        if (video && video.readyState >= 2 && handsDetectorRef.current) {
          if (!isProcessingRef.current) {
            isProcessingRef.current = true
            lastFrameTimeRef.current = now
            try {
              await sendVideoFrameToSharedHands(video, {
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
              })
            } catch {
              // Frame skip when switching tabs/cameras
            } finally {
              isProcessingRef.current = false
            }
          }
        }
      }

      if (now - lastHandSeenTimeRef.current > 800) {
        setIsHandTracked(false)
      }

      if (isCameraRunningRef.current && isMountedRef.current) {
        animFrameRef.current = requestAnimationFrame(processFrame)
      }
    }

    animFrameRef.current = requestAnimationFrame(processFrame)
  }

  // 3. Draw Hand Tracking Overlay & Process Landmarks
  const onMediaPipeResults = (results: MediaPipeHandsResults) => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas) return

    if (video && video.videoWidth > 0 && video.videoHeight > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }
    }

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const frameLandmarks: number[][] = Array(42).fill(0).map(() => [0, 0, 0])
    const hasHands = Boolean(results.multiHandLandmarks && results.multiHandLandmarks.length > 0)

    if (hasHands && results.multiHandLandmarks) {
      setIsHandTracked(true)
      lastHandSeenTimeRef.current = performance.now()

      const detectedHands = results.multiHandLandmarks.map((lms) => ({
        wristX: lms[0]?.x ?? 0,
        landmarks: lms
      }))

      // ASL Citizen left-to-right spatial wrist sorting
      if (detectedHands.length === 2) {
        detectedHands.sort((a, b) => a.wristX - b.wristX)
        detectedHands[0].landmarks.forEach((pt, i) => {
          if (i < 21) frameLandmarks[i] = [pt.x, pt.y, pt.z || 0.0]
        })
        detectedHands[1].landmarks.forEach((pt, i) => {
          if (i < 21) frameLandmarks[21 + i] = [pt.x, pt.y, pt.z || 0.0]
        })
      } else if (detectedHands.length === 1) {
        detectedHands[0].landmarks.forEach((pt, i) => {
          if (i < 21) frameLandmarks[i] = [pt.x, pt.y, pt.z || 0.0]
        })
      }

      drawHandSkeletons(results.multiHandLandmarks, ctx, canvas.width, canvas.height)
    }

    const now = performance.now()
    const item: TimestampedFrame = {
      landmarks: frameLandmarks,
      t: now,
      hasHand: hasHands
    }

    if (captureStateRef.current === "recording") {
      captureFramesRef.current.push(item)
      if (captureFramesRef.current.length >= 64) {
        void finishWordCaptureRef.current()
        return
      }
    }

    // Preserve rest [0,0,0] frames in a timestamped 4s rolling buffer
    rollingBufferRef.current.push(item)
    while (
      rollingBufferRef.current.length > 150 ||
      (rollingBufferRef.current.length > 0 && now - rollingBufferRef.current[0].t > 4000)
    ) {
      rollingBufferRef.current.shift()
    }

    // Continuous Real-Time Inference Trigger (2.3s temporal window matching ASL Citizen)
    if (
      captureStateRef.current === "idle" &&
      !isInferencingRef.current &&
      now - lastInferenceTimeRef.current >= 280
    ) {
      const windowFrames = rollingBufferRef.current.filter((f) => now - f.t <= 2300)
      const activeFrames = windowFrames.filter((f) => f.hasHand).length
      const spanMs =
        windowFrames.length > 1 ? windowFrames[windowFrames.length - 1].t - windowFrames[0].t : 0

      if (windowFrames.length >= 10 && spanMs >= 1300 && activeFrames >= 3) {
        lastInferenceTimeRef.current = now
        void runContinuousInferenceRef.current()
      }
    }
  }
  React.useEffect(() => {
    onMediaPipeResultsRef.current = onMediaPipeResults
  })

  // 4. Sentence Append Helper & Sequence Prediction
  const appendWordToSentence = (wordString: string) => {
    const tokens = wordString.split(" ").filter((w) => w.trim().length > 0)
    setSentenceWords((prev) => {
      const lastToken = prev[prev.length - 1]?.toLowerCase()
      if (lastToken === tokens[0]?.toLowerCase()) return prev
      return [...prev, ...tokens]
    })
  }

  const predictSequence = async (sampled32: number[][][], forceCommit: boolean) => {
    if (sampled32.length === 0) return

    const res = await fetch("/api/predict/sequence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sequence: sampled32,
        top_k: 3,
        model_variant: modelVariantRef.current,
        user_id: preferencesRef.current.userId || "guest",
        personalization_enabled:
          Boolean(preferencesRef.current.userId) &&
          Boolean(preferencesRef.current.personalizationEnabled)
      })
    })

    if (!res.ok) return
    const data = await res.json()
    if (!isMountedRef.current) return

    const rawGloss = (data.predicted_gloss || data.gloss || "").trim()
    const confNum = cleanConfidencePercentage(data.confidence)
    if (!rawGloss) return

    const formattedGloss = formatGlossTitleCase(rawGloss)
    const rawTopK = Array.isArray(data.top_k) ? data.top_k : []
    const newTopK: TopKItem[] = rawTopK.slice(0, 3).map((item: { gloss?: string; confidence?: number }) => {
      const itemRaw = String(item.gloss || rawGloss)
      return {
        gloss: formatGlossTitleCase(itemRaw),
        rawGloss: itemRaw,
        confidence: cleanConfidencePercentage(item.confidence)
      }
    })

    setPrediction({
      gloss: formattedGloss,
      confidence: confNum,
      topK: newTopK
    })

    if (rawGloss === stableCandidateRef.current) {
      stableStreakRef.current += 1
    } else {
      stableCandidateRef.current = rawGloss
      stableStreakRef.current = 1
    }

    const shouldCommit = forceCommit
      ? confNum >= 10
      : confNum >= 26 || (confNum >= 14 && stableStreakRef.current >= 2)

    if (shouldCommit && formattedGloss !== lastPredictedSignRef.current) {
      lastPredictedSignRef.current = formattedGloss
      appendWordToSentence(formattedGloss)
    }
  }

  const finishWordCapture = async () => {
    if (captureStateRef.current !== "recording") return
    if (captureTimeoutRef.current !== null) {
      window.clearTimeout(captureTimeoutRef.current)
      captureTimeoutRef.current = null
    }

    const captured = [...captureFramesRef.current]
    captureFramesRef.current = []
    captureStateRef.current = "analyzing"
    setCaptureState("analyzing")

    const activeCount = captured.filter((f) => f.hasHand).length
    if (captured.length < 6 || activeCount < 2) {
      toast.info("Keep your signing hand in frame for ~2s during Word capture")
      captureStateRef.current = "idle"
      setCaptureState("idle")
      return
    }

    isInferencingRef.current = true
    try {
      const sampled32 = resampleTimestampedSequence(captured, 32)
      await predictSequence(sampled32, true)
    } catch (err) {
      console.warn("Capture inference warning:", err)
    } finally {
      isInferencingRef.current = false
      if (isMountedRef.current) {
        captureStateRef.current = "idle"
        setCaptureState("idle")
      }
    }
  }

  const startWordCapture = React.useCallback(() => {
    if (!isCameraRunningRef.current) {
      toast.info("Start the camera first to capture a sign")
      return
    }
    if (captureStateRef.current !== "idle") return

    if (captureTimeoutRef.current !== null) {
      window.clearTimeout(captureTimeoutRef.current)
    }
    captureFramesRef.current = []
    captureStateRef.current = "recording"
    setCaptureState("recording")

    captureTimeoutRef.current = window.setTimeout(() => {
      void finishWordCaptureRef.current()
    }, 2200)
  }, [])

  const runContinuousInference = async () => {
    if (isInferencingRef.current) return
    isInferencingRef.current = true

    try {
      const now = performance.now()
      const windowFrames = rollingBufferRef.current.filter((f) => now - f.t <= 2300)
      const activeFrames = windowFrames.filter((f) => f.hasHand).length
      const spanMs =
        windowFrames.length > 1 ? windowFrames[windowFrames.length - 1].t - windowFrames[0].t : 0
      if (windowFrames.length < 10 || spanMs < 1300 || activeFrames < 3) return

      const sampled32 = resampleTimestampedSequence(windowFrames, 32)
      await predictSequence(sampled32, false)
    } catch (err) {
      console.warn("Inference warning:", err)
    } finally {
      isInferencingRef.current = false
    }
  }

  React.useEffect(() => {
    finishWordCaptureRef.current = finishWordCapture
    runContinuousInferenceRef.current = runContinuousInference
  })

  const handleSelectCandidate = (candidate: TopKItem) => {
    setPrediction((prev) => ({
      ...prev,
      gloss: candidate.gloss,
      confidence: candidate.confidence
    }))
    lastPredictedSignRef.current = candidate.gloss
    appendWordToSentence(candidate.gloss)
  }

  // 5. Camera Lifecycle
  const initMediaPipeRef = React.useRef(initMediaPipe)
  React.useEffect(() => {
    initMediaPipeRef.current = initMediaPipe
  })

  const startCamera = React.useCallback(async (deviceId?: string) => {
    try {
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
          ...(deviceId ? { deviceId: { exact: deviceId } } : {})
        },
        audio: false
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      if (!isMountedRef.current) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      mediaStreamRef.current = stream
      isCameraRunningRef.current = true
      setIsCameraOn(true)

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(console.warn)
          if (canvasRef.current && videoRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth || 1280
            canvasRef.current.height = videoRef.current.videoHeight || 720
          }
        }
      }

      void initMediaPipeRef.current()
    } catch (err: unknown) {
      console.error("Camera access error:", err)
      isCameraRunningRef.current = false
      setIsCameraOn(false)
      toast.error("Unable to access camera. Please allow webcam permissions.")
    }
  }, [])

  const stopCamera = React.useCallback(() => {
    isCameraRunningRef.current = false
    if (captureTimeoutRef.current !== null) {
      window.clearTimeout(captureTimeoutRef.current)
      captureTimeoutRef.current = null
    }
    captureStateRef.current = "idle"
    setCaptureState("idle")
    clearMediaPipeHandsListener(stableHandsListener)
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      mediaStreamRef.current = null
    }
    setIsCameraOn(false)
    setIsHandTracked(false)
  }, [stableHandsListener])

  const toggleCamera = () => {
    if (isCameraOn) {
      stopCamera()
    } else {
      startCamera(selectedDeviceId || undefined)
    }
  }

  React.useEffect(() => {
    isMountedRef.current = true
    const timer = window.setTimeout(() => {
      if (isMountedRef.current) {
        void startCamera()
      }
    }, 0)
    return () => {
      window.clearTimeout(timer)
      isMountedRef.current = false
      stopCamera()
    }
  }, [startCamera, stopCamera])

  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      ) {
        return
      }
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault()
        startWordCapture()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [startWordCapture])

  // 7. Sentence Builder Actions
  const handleClearSentence = () => {
    setSentenceWords([])
    toast.info("Sentence cleared")
  }

  const handleCopySentence = () => {
    const fullText = sentenceWords.join(" ")
    if (!fullText) {
      toast.info("Sentence is empty")
      return
    }
    navigator.clipboard.writeText(fullText)
    setHasCopied(true)
    toast.success("Sentence copied to clipboard")
    setTimeout(() => setHasCopied(false), 2000)
  }

  const handleSpeakSentence = async () => {
    const fullText = sentenceWords.join(" ")
    if (!fullText) {
      toast.info("Sentence is empty")
      return
    }
    if (isPlayingTTS) return
    setIsPlayingTTS(true)

    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: fullText,
          voice: preferencesRef.current.ttsVoice || "af_sarah"
        })
      })

      if (res.ok) {
        const data = await res.json()
        const audioSrc = data.audio_url || data.audio_data
        if (audioSrc) {
          if (audioPlayerRef.current) {
            audioPlayerRef.current.pause()
          }
          const audio = new Audio(audioSrc)
          audioPlayerRef.current = audio
          audio.onended = () => setIsPlayingTTS(false)
          audio.onerror = () => fallbackSpeak(fullText)
          await audio.play()
          toast.success("Voicing sentence with Kokoro TTS")
          return
        }
      }
      fallbackSpeak(fullText)
    } catch {
      fallbackSpeak(fullText)
    }
  }

  const fallbackSpeak = (text: string) => {
    if ("speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(text)
      u.onend = () => setIsPlayingTTS(false)
      u.onerror = () => setIsPlayingTTS(false)
      window.speechSynthesis.speak(u)
    } else {
      setIsPlayingTTS(false)
    }
  }

  return (
    <div className="w-full flex flex-col space-y-6 select-none pb-6">
      {/* =========================================================================
          STUDIO HEADER BAR
         ========================================================================= */}
      <div className="w-full sb-glass-card rounded-[26px] px-5 sm:px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {onOpenMobileMenu && (
            <button
              onClick={onOpenMobileMenu}
              className="p-2 rounded-xl text-[#FAF6F0] hover:bg-[#251F35] lg:hidden transition cursor-pointer"
              aria-label="Open navigation menu"
            >
              <Menu className="w-5 h-5" />
            </button>
          )}
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#FAF6F0]">
              Live Recognition Studio
            </h1>
            <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Real-time 32-frame MediaPipe landmark recognition across 2,731 ASL signs
            </p>
          </div>
        </div>

        {/* Right studio controls */}
        <div className="flex flex-wrap items-center gap-2.5 self-start sm:self-auto">
          {/* Model Variant Switcher: Full 2,731 Signs vs Demo 50 */}
          <div className="inline-flex items-center rounded-full bg-[#161221] p-1 border border-white/10">
            {(
              [
                { id: "production_exp13", label: "2,731 Signs" },
                { id: "demo_50", label: "Demo 50" }
              ] as const
            ).map((opt) => {
              const active = modelVariant === opt.id
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setModelVariant(opt.id)}
                  className={`px-3 py-1 rounded-full text-[11px] font-bold transition cursor-pointer ${
                    active
                      ? "sb-orange-pill text-white"
                      : "text-[#ACA5BC] hover:text-[#FAF6F0]"
                  }`}
                >
                  {opt.label}
                </button>
              )
            })}
          </div>

          {preferences.personalizationEnabled && (
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#251F35] border border-amber-400/30 text-[11px] font-bold text-amber-300">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Custom Signs On</span>
            </div>
          )}

          {/* Quick Toggle: Full Signs vs Fingerspelling Mode */}
          {onSwitchToFingerspelling && (
            <button
              onClick={onSwitchToFingerspelling}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-[#251F35] border border-amber-400/30 text-xs font-bold text-amber-300 shadow-2xs hover:bg-amber-400/18 transition cursor-pointer"
              title="Switch to Fingerspelling alphabet recognition"
            >
              <Type className="w-3.5 h-3.5" />
              <span>Fingerspelling A–Z</span>
            </button>
          )}

          {/* Status Badge: Model Ready */}
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/12 border border-emerald-400/30/70 text-xs font-bold text-emerald-300 shadow-2xs">
            <span className="w-2 h-2 rounded-full bg-[#1FA968] animate-pulse" />
            <span>{isModelReady ? "EXP13 Ready" : "Connecting..."}</span>
          </div>
        </div>
      </div>

      {/* =========================================================================
          MAIN RECOGNITION WORKSPACE (Two-Column Layout: Left 65% / Right 35%)
         ========================================================================= */}
      <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-5 xl:gap-6 items-start">
        {/* =========================================================================
            LEFT COLUMN: Large Camera Preview Card (65% width)
           ========================================================================= */}
        <div className="lg:col-span-8 flex flex-col">
          <div className="relative w-full rounded-[24px] overflow-hidden aspect-video bg-[#171B26] shadow-md border border-white/10 flex items-center justify-center">
            {/* Always mount video + overlay canvas to avoid React StrictMode ref races */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover scale-x-[-1] ${isCameraOn ? "" : "hidden"}`}
            />

            <canvas
              ref={canvasRef}
              className={`absolute inset-0 w-full h-full object-cover pointer-events-none scale-x-[-1] ${
                isCameraOn ? "" : "hidden"
              }`}
            />

            {!isCameraOn && (
              <div className="flex flex-col items-center justify-center text-center p-8 text-zinc-400 gap-3 z-10">
                <div className="w-14 h-14 rounded-full bg-white/10 flex items-center justify-center text-zinc-300">
                  <CameraOff className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white">Camera is currently paused</h3>
                  <p className="text-xs text-zinc-400 mt-1">Click below to activate live sign recognition</p>
                </div>
                <button
                  onClick={toggleCamera}
                  className="mt-2 px-5 py-2 rounded-xl bg-[#D97706] hover:bg-[#B45309] text-white font-bold text-xs shadow-md transition cursor-pointer"
                >
                  Start Camera
                </button>
              </div>
            )}

            {/* Top-Left HUD: Live Status Pill with Hand Tracking Indicator */}
            <div className="absolute top-4 left-4 z-20 pointer-events-none">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md text-white text-xs font-bold border border-white/10 shadow-xs">
                <span className={`w-2 h-2 rounded-full ${isHandTracked ? "bg-[#22B66B] animate-pulse" : "bg-emerald-400"}`} />
                <span>Live</span>
                {isHandTracked && (
                  <>
                    <span className="w-1 h-1 rounded-full bg-white/40" />
                    <span className="text-[#FFE2CA] flex items-center gap-1 font-semibold">
                      <Hand className="w-3 h-3 text-amber-400" /> Hand Tracked
                    </span>
                  </>
                )}
              </div>
            </div>

            {/* Top-Right HUD: Camera Controls & Settings */}
            <div className="absolute top-4 right-4 z-20 flex items-center gap-2">
              {/* Snapshot / Camera Toggle button */}
              <button
                onClick={toggleCamera}
                className="w-9 h-9 rounded-xl bg-black/60 backdrop-blur-md text-white border border-white/10 flex items-center justify-center hover:bg-black/80 transition cursor-pointer"
                title={isCameraOn ? "Pause Camera" : "Resume Camera"}
              >
                {isCameraOn ? <Camera className="w-4.5 h-4.5" /> : <CameraOff className="w-4.5 h-4.5 text-red-400" />}
              </button>

              {/* Camera Settings button */}
              <div className="relative">
                <button
                  onClick={() => setShowCameraMenu(!showCameraMenu)}
                  className="w-9 h-9 rounded-xl bg-black/60 backdrop-blur-md text-white border border-white/10 flex items-center justify-center hover:bg-black/80 transition cursor-pointer"
                  title="Camera Settings"
                >
                  <Settings className="w-4.5 h-4.5" />
                </button>

                {showCameraMenu && (
                  <div className="absolute right-0 mt-2 w-64 rounded-2xl bg-[#1D1829] border border-white/10 shadow-xl p-3 z-50 text-left text-xs font-bold text-[#FAF6F0]">
                    <div className="pb-2 border-b border-white/10 text-[11px] uppercase tracking-wider text-[#ACA5BC]">
                      Select Camera Device
                    </div>
                    <div className="mt-2 space-y-1">
                      {videoDevices.length > 0 ? (
                        videoDevices.map((dev) => (
                          <button
                            key={dev.deviceId}
                            onClick={() => {
                              setSelectedDeviceId(dev.deviceId)
                              setShowCameraMenu(false)
                              startCamera(dev.deviceId)
                            }}
                            className={`w-full text-left px-2.5 py-1.5 rounded-lg truncate transition ${
                              selectedDeviceId === dev.deviceId
                                ? "bg-[#251F35] text-amber-400"
                                : "hover:bg-[#2A233C]"
                            }`}
                          >
                            {dev.label || `Camera ${dev.deviceId.slice(0, 5)}`}
                          </button>
                        ))
                      ) : (
                        <p className="text-[#ACA5BC] font-normal py-1">Default webcam active</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Bottom HUD: Isolated 2.2s Word Capture Trigger */}
            {isCameraOn && (
              <div className="absolute bottom-4 left-4 right-4 z-20 flex items-center justify-between pointer-events-none">
                <button
                  type="button"
                  onClick={startWordCapture}
                  disabled={captureState !== "idle"}
                  className={`pointer-events-auto inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold shadow-md backdrop-blur-md border transition cursor-pointer ${
                    captureState === "recording"
                      ? "bg-[#D97706] text-white border-[#FF9E44]"
                      : captureState === "analyzing"
                        ? "bg-black/75 text-white border-white/15"
                        : "bg-black/65 hover:bg-[#D97706] text-white border-white/15"
                  }`}
                  title="Record an isolated 2.2-second sign (or press Space)"
                >
                  {captureState === "analyzing" ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Timer className={`w-3.5 h-3.5 ${captureState === "recording" ? "animate-pulse" : ""}`} />
                  )}
                  <span>
                    {captureState === "recording"
                      ? "Recording Sign (2.2s)..."
                      : captureState === "analyzing"
                        ? "Analyzing Sign..."
                        : "Capture Sign (2.2s · Space)"}
                  </span>
                </button>
              </div>
            )}

            {/* Four Subtle White Corner Framing Indicators */}
            <div className="absolute top-4 left-4 w-7 h-7 border-t-[2.5px] border-l-[2.5px] border-white/80 rounded-tl-lg pointer-events-none z-10" />
            <div className="absolute top-4 right-4 w-7 h-7 border-t-[2.5px] border-r-[2.5px] border-white/80 rounded-tr-lg pointer-events-none z-10" />
            <div className="absolute bottom-4 left-4 w-7 h-7 border-b-[2.5px] border-l-[2.5px] border-white/80 rounded-bl-lg pointer-events-none z-10" />
            <div className="absolute bottom-4 right-4 w-7 h-7 border-b-[2.5px] border-r-[2.5px] border-white/80 rounded-br-lg pointer-events-none z-10" />
          </div>
        </div>

        {/* =========================================================================
            RIGHT COLUMN: Recognition Results (35% width, 2 Vertically Stacked Cards)
           ========================================================================= */}
        <div className="lg:col-span-4 flex flex-col space-y-5">
          {/* -----------------------------------------------------------------------
              CARD 1: Detected Sign with Expressive Owl Companion
             ----------------------------------------------------------------------- */}
          <div className="sb-surface-card w-full rounded-[26px] p-6 relative flex flex-col justify-between min-h-[195px]">
            {/* Top row: Label + Owl Mascot Companion */}
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#ACA5BC] uppercase tracking-wider">
                  Detected Sign
                </span>

                <SignBridgeMascot
                  variant="heart"
                  size="sm"
                  state={
                    isPlayingTTS
                      ? "speaking"
                      : prediction.confidence >= 40
                        ? "detected"
                        : isHandTracked
                          ? "listening"
                          : "idle"
                  }
                />
              </div>

              {/* Main Prediction Name */}
              <h2 className="text-3xl sm:text-4xl font-black text-[#FAF6F0] tracking-tight mt-1 mb-4">
                {prediction.gloss}
              </h2>
            </div>

            {/* Confidence Label, Value & Progress Bar */}
            <div>
              <div className="flex items-center justify-between text-xs font-semibold mb-2">
                <span className="text-[#ACA5BC]">Confidence</span>
                <span className="text-sm font-bold text-amber-400 font-mono tabular-nums">
                  {prediction.confidence.toFixed(1)}%
                </span>
              </div>

              {/* Horizontal orange progress bar */}
              <div className="w-full h-2.5 rounded-full bg-[#2C2440] overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, Math.max(0, prediction.confidence))}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="h-full rounded-full bg-gradient-to-r from-[#FF9436] to-[#E85D04]"
                />
              </div>
            </div>
          </div>

          {/* -----------------------------------------------------------------------
              CARD 2: Top Predictions (3 Distinct Rows)
             ----------------------------------------------------------------------- */}
          <div className="sb-surface-card w-full rounded-[26px] p-6 flex flex-col">
            <h3 className="text-base sm:text-lg font-extrabold text-[#FAF6F0] mb-3.5">
              Top Predictions
            </h3>

            <div className="space-y-2.5">
              {prediction.topK.length > 0 ? (
                prediction.topK.map((item, idx) => {
                  const isTop = idx === 0

                  return (
                    <button
                      key={item.rawGloss + idx}
                      type="button"
                      onClick={() => handleSelectCandidate(item)}
                      className={`w-full rounded-2xl p-3 flex items-center justify-between transition-colors cursor-pointer text-left ${
                        isTop
                          ? "bg-[#251F35] border border-amber-400/35 hover:bg-[#2F2744]"
                          : "bg-[#1A1527] border border-white/10 hover:bg-[#251F35]"
                      }`}
                      title={`Select ${item.gloss} and add to Sentence Builder`}
                    >
                      {/* Left: Hand icon + Prediction text */}
                      <div className="flex items-center gap-3 min-w-0">
                        <div
                          className={`w-8 h-8 rounded-xl flex items-center justify-center text-sm shrink-0 ${
                            isTop
                              ? "bg-amber-400/18 text-amber-300"
                              : "bg-white/10 text-[#ACA5BC]"
                          }`}
                        >
                          {isTop ? "🤟" : "✋"}
                        </div>
                        <span
                          className={`text-sm truncate ${
                            isTop
                              ? "font-bold text-[#FAF6F0]"
                              : "font-semibold text-[#FAF6F0]"
                          }`}
                        >
                          {item.gloss}
                        </span>
                      </div>

                      {/* Right: Confidence percentage */}
                      <span
                        className={`text-sm font-mono shrink-0 ml-2 ${
                          isTop
                            ? "font-bold text-amber-400"
                            : "font-medium text-[#ACA5BC]"
                        }`}
                      >
                        {item.confidence.toFixed(1)}%
                      </span>
                    </button>
                  )
                })
              ) : (
                <p className="text-xs text-[#ACA5BC] py-2">
                  Show a sign in front of the camera or press Capture Sign to see live Top-3 candidates.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* =========================================================================
          SENTENCE BUILDER (Full-Width Card Below Camera & Predictions)
         ========================================================================= */}
      <div className="sb-surface-card w-full rounded-[26px] p-5 sm:p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        {/* Left side: Heading + Chips */}
        <div className="flex-1 min-w-0">
          <span className="text-xs font-bold text-[#ACA5BC] uppercase tracking-wider block mb-2.5">
            Sentence Builder
          </span>

          {/* Word chips container */}
          <div className="flex flex-wrap items-center gap-2 min-h-[36px]">
            {sentenceWords.length > 0 ? (
              sentenceWords.map((word, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#231D32] border border-white/10 text-sm font-bold text-[#FAF6F0] shadow-2xs"
                >
                  <span>{word}</span>
                  <button
                    onClick={() => {
                      setSentenceWords((prev) => prev.filter((_, idx) => idx !== i))
                    }}
                    className="hover:text-red-500 transition text-[#ACA5BC]"
                    title="Remove word"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))
            ) : (
              <span className="text-xs text-[#ACA5BC] italic">
                Signs recognized will assemble into a sentence here...
              </span>
            )}

            {/* Blinking text cursor */}
            <span className="w-0.5 h-5 bg-[#D97706] animate-pulse inline-block align-middle ml-0.5" />
          </div>
        </div>

        {/* Right side: 3 Action Controls (Clear, Copy, Speak) */}
        <div className="flex items-center gap-2.5 shrink-0 self-end sm:self-center">
          {/* Clear Button */}
          <button
            onClick={handleClearSentence}
            disabled={sentenceWords.length === 0}
            className="w-10 h-10 rounded-xl border border-white/10 bg-[#1D1829] hover:bg-[#251F35] text-[#ACA5BC] hover:text-[#FAF6F0] flex items-center justify-center transition cursor-pointer disabled:opacity-40"
            title="Clear Sentence"
            aria-label="Clear sentence"
          >
            <X className="w-4.5 h-4.5" />
          </button>

          {/* Copy Button */}
          <button
            onClick={handleCopySentence}
            disabled={sentenceWords.length === 0}
            className="w-10 h-10 rounded-xl border border-white/10 bg-[#1D1829] hover:bg-[#251F35] text-[#ACA5BC] hover:text-[#FAF6F0] flex items-center justify-center transition cursor-pointer disabled:opacity-40"
            title="Copy to Clipboard"
            aria-label="Copy sentence"
          >
            {hasCopied ? <Check className="w-4.5 h-4.5 text-[#22B66B]" /> : <Copy className="w-4.5 h-4.5" />}
          </button>

          {/* Speak Button (Vivid Orange) */}
          <button
            onClick={handleSpeakSentence}
            disabled={sentenceWords.length === 0 || isPlayingTTS}
            className="px-5 py-2.5 rounded-xl bg-[#D97706] hover:bg-[#B45309] text-white font-bold text-sm shadow-xs flex items-center gap-2 transition cursor-pointer disabled:opacity-50"
            title="Speak Sentence with Kokoro TTS"
          >
            <Volume2 className="w-4.5 h-4.5" />
            <span>{isPlayingTTS ? "Speaking..." : "Speak"}</span>
          </button>
        </div>
      </div>
    </div>
  )
}
