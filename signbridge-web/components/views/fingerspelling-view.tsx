"use client"

import * as React from "react"
import {
  Camera,
  CameraOff,
  Volume2,
  Copy,
  Check,
  RotateCcw,
  Delete,
  Sparkles,
  Cpu,
  Layers,
  Menu,
  Hand
} from "lucide-react"
import { toast } from "sonner"
import {
  clearMediaPipeHandsListener,
  getSharedMediaPipeHands,
  MediaPipeHandsResults,
  MediaPipeLandmark,
  sendVideoFrameToSharedHands,
  setMediaPipeHandsListener
} from "@/lib/mediapipe-hands"
import { useSignBridgeStore } from "@/lib/signbridge-store"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"

interface FingerspellingViewProps {
  onOpenMobileMenu?: () => void
  onSwitchToIsolated?: () => void
}

function cleanConfidencePercentage(rawConf: unknown): number {
  const val = typeof rawConf === "number" ? rawConf : parseFloat(String(rawConf ?? 0)) || 0
  if (!Number.isFinite(val) || val <= 0) return 0
  return Math.min(99.9, Math.max(0.1, Number(val.toFixed(1))))
}

function drawHandSkeletons(hands: MediaPipeLandmark[][], ctx: CanvasRenderingContext2D, w: number, h: number) {
  const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
  ]

  hands.forEach((landmarks) => {
    ctx.strokeStyle = "rgba(255, 121, 0, 0.9)"
    ctx.lineWidth = 3
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
      const isTip = [4, 8, 12, 16, 20].includes(idx)

      ctx.beginPath()
      ctx.arc(px, py, isTip ? 6 : 4, 0, 2 * Math.PI)
      ctx.fillStyle = isTip ? "#FF7900" : "#FFFFFF"
      ctx.shadowColor = "#FF7900"
      ctx.shadowBlur = 4
      ctx.fill()
      ctx.shadowBlur = 0
      ctx.lineWidth = 1.5
      ctx.strokeStyle = "#FF7900"
      ctx.stroke()
    })
  })
}

export function FingerspellingView({ onOpenMobileMenu, onSwitchToIsolated }: FingerspellingViewProps) {
  const { modelHealth, preferences } = useSignBridgeStore()

  // WebRTC Camera State
  const [isCameraOn, setIsCameraOn] = React.useState(true)
  const isCameraRunningRef = React.useRef(true)
  const [cameraError, setCameraError] = React.useState<string | null>(null)

  const videoRef = React.useRef<HTMLVideoElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const mediaStreamRef = React.useRef<MediaStream | null>(null)
  const animFrameRef = React.useRef<number | null>(null)

  // MediaPipe Hands & Processing
  const handsDetectorRef = React.useRef<boolean>(false)
  const isProcessingRef = React.useRef(false)
  const rollingBufferRef = React.useRef<number[][][]>([]) // shape [T, 42, 3]

  // Model & Server Readiness
  const [fps, setFps] = React.useState<number | null>(null)
  const [latencyMs, setLatencyMs] = React.useState<number | null>(null)
  const [isHandDetected, setIsHandDetected] = React.useState(false)

  // Live Real Prediction State - STRICTLY NO FAKE OR MOCK PREDICTIONS
  const [currentChar, setCurrentChar] = React.useState<string | null>(null)
  const [charConfidence, setCharConfidence] = React.useState<number | null>(null)
  const [charStability, setCharStability] = React.useState<number | null>(null)
  const [charHistory, setCharHistory] = React.useState<string[]>([])
  
  // Real Assembled Word & Sentence State
  const [currentWord, setCurrentWord] = React.useState<string>("")
  const [sentenceWords, setSentenceWords] = React.useState<string[]>([])
  const [hasCopied, setHasCopied] = React.useState(false)
  const [isPlayingTTS, setIsPlayingTTS] = React.useState(false)
  const audioPlayerRef = React.useRef<HTMLAudioElement | null>(null)

  // Inference pacing & tracking
  const lastInferenceTimeRef = React.useRef(0)
  const isInferencingRef = React.useRef(false)
  const fpsCountRef = React.useRef(0)
  const lastFpsTimeRef = React.useRef(0)
  const lastEmittedCharRef = React.useRef<string | null>(null)
  const charHoldCountRef = React.useRef(0)

  // 1. Shared MediaPipe Hands Singleton Initialization
  const onHandsResultsRef = React.useRef<(results: MediaPipeHandsResults) => void>(() => {})
  const runFingerspellingInferenceRef = React.useRef<(sequence: number[][][]) => Promise<void>>(async () => {})
  const stableHandsListener = React.useCallback((results: MediaPipeHandsResults) => {
    onHandsResultsRef.current(results)
  }, [])

  const initHandsDetector = React.useCallback(async () => {
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
    } catch (err) {
      console.error("Fingerspelling hands detector error:", err)
    }
  }, [stableHandsListener])

  // 2. Setup Webcam Video Stream with Safe Error Handling
  React.useEffect(() => {
    let isMounted = true

    const startCamera = async () => {
      try {
        setCameraError(null)
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((t) => t.stop())
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user"
          },
          audio: false
        })

        if (!isMounted) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }

        mediaStreamRef.current = stream
        isCameraRunningRef.current = true

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => {
            if (!isMounted) return
            videoRef.current?.play().catch((err) => {
              if (err.name !== "AbortError") {
                console.warn("Video play error:", err)
              }
            })
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth || 1280
              canvasRef.current.height = videoRef.current.videoHeight || 720
            }
          }
        }

        void initHandsDetector()
      } catch (err: unknown) {
        if (!isMounted) return
        const e = err instanceof Error ? err : null
        if (e?.name !== "AbortError") {
          console.error("Camera access failed:", err)
          setCameraError(e?.message || "Failed to access webcam")
        }
      }
    }

    if (isCameraOn) {
      startCamera()
    } else {
      isCameraRunningRef.current = false
      clearMediaPipeHandsListener(stableHandsListener)
    }

    return () => {
      isMounted = false
      isCameraRunningRef.current = false
      clearMediaPipeHandsListener(stableHandsListener)
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop())
        mediaStreamRef.current = null
      }
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [isCameraOn, initHandsDetector, stableHandsListener])

  // 3. Animation Processing Loop
  React.useEffect(() => {
    let active = true

    const processFrame = async () => {
      if (!active) return

      fpsCountRef.current += 1
      const now = Date.now()
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(fpsCountRef.current)
        fpsCountRef.current = 0
        lastFpsTimeRef.current = now
      }

      if (
        isCameraRunningRef.current &&
        videoRef.current &&
        videoRef.current.readyState >= 2 &&
        handsDetectorRef.current &&
        !isProcessingRef.current
      ) {
        isProcessingRef.current = true
        try {
          await sendVideoFrameToSharedHands(videoRef.current, {
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
          })
        } catch {
          // Frame skip
        } finally {
          isProcessingRef.current = false
        }
      }

      if (active) {
        animFrameRef.current = requestAnimationFrame(processFrame)
      }
    }

    animFrameRef.current = requestAnimationFrame(processFrame)

    return () => {
      active = false
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [])

  // 4. Landmark Drawing & Frame Buffer
  const onHandsResults = (results: MediaPipeHandsResults) => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const frameLandmarks: number[][] = Array(42).fill(0).map(() => [0, 0, 0])
    const hasHands = Boolean(results.multiHandLandmarks && results.multiHandLandmarks.length > 0)
    setIsHandDetected(hasHands)

    if (hasHands && results.multiHandLandmarks) {
      const detected = results.multiHandLandmarks.map((lms) => ({
        wristX: lms[0]?.x ?? 0,
        landmarks: lms
      }))

      if (detected.length === 2) {
        detected.sort((a, b) => a.wristX - b.wristX)
        detected[0].landmarks.forEach((pt, i) => {
          if (i < 21) frameLandmarks[i] = [pt.x, pt.y, pt.z || 0.0]
        })
        detected[1].landmarks.forEach((pt, i) => {
          if (i < 21) frameLandmarks[21 + i] = [pt.x, pt.y, pt.z || 0.0]
        })
      } else if (detected.length === 1) {
        detected[0].landmarks.forEach((pt, i) => {
          if (i < 21) frameLandmarks[i] = [pt.x, pt.y, pt.z || 0.0]
        })
      }

      drawHandSkeletons(results.multiHandLandmarks, ctx, canvas.width, canvas.height)
    }

    rollingBufferRef.current.push(frameLandmarks)
    if (rollingBufferRef.current.length > 48) {
      rollingBufferRef.current.shift()
    }

    // Trigger Fingerspelling Inference every 140ms when hands are visible
    const now = Date.now()
    if (
      hasHands &&
      rollingBufferRef.current.length >= 6 &&
      now - lastInferenceTimeRef.current >= 140 &&
      !isInferencingRef.current
    ) {
      lastInferenceTimeRef.current = now
      void runFingerspellingInferenceRef.current([...rollingBufferRef.current.slice(-24)])
    }
  }
  React.useEffect(() => {
    onHandsResultsRef.current = onHandsResults
  })

  // 5. Execute Real Streaming Inference
  const runFingerspellingInference = async (sequence: number[][][]) => {
    isInferencingRef.current = true
    try {
      const res = await fetch("/api/fingerspelling/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sequence: sequence.slice(-24),
          min_char_prob: 0.2,
          target_len: 24
        })
      })

      if (res.ok) {
        const data = await res.json()
        if (typeof data.latency_ms === "number") setLatencyMs(data.latency_ms)

        const rawCandidate = String(data.char || data.text || "").trim().toUpperCase()
        const predictedChar = rawCandidate.length > 0 ? rawCandidate[rawCandidate.length - 1] : ""

        if (predictedChar) {
          setCurrentChar(predictedChar)
          const conf = cleanConfidencePercentage(data.confidence)
          setCharConfidence(conf)
          setCharStability(cleanConfidencePercentage(data.stability))

          // Temporal debouncing: require letter to persist for 2 consecutive predictions
          if (predictedChar === lastEmittedCharRef.current) {
            charHoldCountRef.current += 1
            if (charHoldCountRef.current === 2) {
              setCharHistory((prev) => [...prev.slice(-7), predictedChar])
              setCurrentWord((prev) => `${prev}${predictedChar}`)
            }
          } else {
            lastEmittedCharRef.current = predictedChar
            charHoldCountRef.current = 1
          }
        }
      }
    } catch (err) {
      console.warn("Inference call warning:", err)
    } finally {
      isInferencingRef.current = false
    }
  }

  React.useEffect(() => {
    runFingerspellingInferenceRef.current = runFingerspellingInference
  })

  // 6. User Actions
  const handleBackspace = () => {
    if (currentWord.length > 0) {
      setCurrentWord((prev) => prev.slice(0, -1))
    } else if (sentenceWords.length > 0) {
      const last = sentenceWords[sentenceWords.length - 1]
      setSentenceWords((prev) => prev.slice(0, -1))
      setCurrentWord(last)
    }
  }

  const handleAddSpace = () => {
    if (currentWord.trim().length > 0) {
      setSentenceWords((prev) => [...prev, currentWord.trim()])
      setCurrentWord("")
    }
  }

  const handleClear = () => {
    setCurrentChar(null)
    setCharConfidence(null)
    setCharStability(null)
    setCurrentWord("")
    setSentenceWords([])
    setCharHistory([])
    toast.info("Cleared sentence builder")
  }

  const handleCopy = () => {
    const full = [...sentenceWords, currentWord].filter(Boolean).join(" ")
    if (!full) return
    navigator.clipboard.writeText(full)
    setHasCopied(true)
    toast.success("Copied to clipboard!")
    setTimeout(() => setHasCopied(false), 2000)
  }

  const handleSpeakTTS = async () => {
    const full = [...sentenceWords, currentWord].filter(Boolean).join(" ")
    if (!full) {
      toast.error("Sentence is empty")
      return
    }

    try {
      setIsPlayingTTS(true)
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: full, voice: preferences.ttsVoice || "af_sarah" })
      })

      if (!res.ok) throw new Error("TTS generation failed")

      const data = await res.json()
      const audioSrc = data.audio_url || data.audio_data
      if (audioSrc && audioPlayerRef.current) {
        audioPlayerRef.current.src = audioSrc
        audioPlayerRef.current.onended = () => setIsPlayingTTS(false)
        audioPlayerRef.current.onerror = () => setIsPlayingTTS(false)
        await audioPlayerRef.current.play()
        return
      }

      if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(full)
        u.onend = () => setIsPlayingTTS(false)
        u.onerror = () => setIsPlayingTTS(false)
        window.speechSynthesis.speak(u)
      } else {
        setIsPlayingTTS(false)
      }
    } catch (err: unknown) {
      console.error("TTS error:", err)
      toast.error("Kokoro TTS playback error")
      setIsPlayingTTS(false)
    }
  }

  return (
    <div className="w-full flex flex-col gap-6 select-none">
      {/* Hidden audio element for TTS */}
      <audio ref={audioPlayerRef} className="hidden" />

      {/* Top Header Bar */}
      <div className="sb-glass-card rounded-[26px] px-5 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {onOpenMobileMenu && (
            <button
              onClick={onOpenMobileMenu}
              className="lg:hidden p-2 rounded-xl text-[#ACA5BC] hover:bg-[#251F35]"
              aria-label="Open navigation"
            >
              <Menu className="w-6 h-6" />
            </button>
          )}
          <div>
            <div className="flex flex-wrap items-center gap-2.5">
              <h1 className="text-2xl sm:text-3xl font-extrabold text-[#FAF6F0] tracking-tight">
                ASL Fingerspelling Studio
              </h1>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/12 border border-emerald-400/30 text-emerald-300">
                <span className="w-2 h-2 rounded-full bg-[#1FA968] animate-pulse" />
                {modelHealth.fingerspellingReady ? "Alphabet MLP Ready" : "Connecting..."}
              </span>
            </div>
            <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Continuous A–Z letter recognition via 63-D Isotropic BBox Normalized Landmark MLP
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5">
          {onSwitchToIsolated && (
            <button
              onClick={onSwitchToIsolated}
              className="flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold text-amber-300 bg-[#251F35] hover:bg-amber-400/18 transition border border-amber-400/30 cursor-pointer"
            >
              <Layers className="w-4 h-4" />
              <span>Full ASL Signs</span>
            </button>
          )}

          <div className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-[#1D1829] border border-white/10 text-xs font-bold text-[#ACA5BC]">
            <Cpu className="w-4 h-4 text-amber-400" />
            <span>
              {modelHealth.hardwareDevice ?? "CPU"}
              {latencyMs !== null ? ` (${latencyMs} ms)` : ""}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content Workspace (2 Columns) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Live Webcam (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="relative bg-[#120F1B] rounded-[24px] overflow-hidden aspect-[4/3] shadow-md border border-white/10 flex items-center justify-center">
            {/* Live Indicator Pill */}
            <div className="absolute top-4 left-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md text-white text-xs font-bold border border-white/10">
              <span className="w-2 h-2 rounded-full bg-[#FF453A] animate-ping" />
              <span>Live Camera</span>
            </div>

            {/* FPS & Latency HUD */}
            {(fps !== null || latencyMs !== null) && (
              <div className="absolute top-4 right-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md text-white text-xs font-semibold border border-white/10 font-mono">
                {fps !== null && <span>{fps} FPS</span>}
                {fps !== null && latencyMs !== null && <span className="text-white/40">|</span>}
                {latencyMs !== null && <span className="text-[#1FA968]">{latencyMs} ms</span>}
              </div>
            )}

            {/* Hand Presence Pill */}
            <div className="absolute bottom-4 left-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md text-xs font-bold border border-white/10">
              <Hand className={`w-3.5 h-3.5 ${isHandDetected ? "text-[#22B66B]" : "text-white/40"}`} />
              <span className={isHandDetected ? "text-[#22B66B]" : "text-white/60"}>
                {isHandDetected ? "Hand Tracked" : "Show Hand to Sign"}
              </span>
            </div>

            {/* Webcam Video Stream (mirrored) */}
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-cover transform -scale-x-100 ${
                !isCameraOn ? "hidden" : ""
              }`}
            />

            {/* Landmark Overlay Canvas (mirrored) */}
            <canvas
              ref={canvasRef}
              className={`absolute inset-0 w-full h-full object-cover pointer-events-none transform -scale-x-100 ${
                !isCameraOn ? "hidden" : ""
              }`}
            />

            {/* Camera Off Placeholder */}
            {!isCameraOn && (
              <div className="flex flex-col items-center justify-center p-8 text-center text-white/70">
                <CameraOff className="w-14 h-14 text-white/40 mb-3" />
                <p className="text-base font-bold text-white">Camera is paused</p>
                <p className="text-xs text-white/60 mt-1 max-w-xs">
                  Turn the camera on to begin continuous ASL fingerspelling recognition.
                </p>
                <button
                  onClick={() => setIsCameraOn(true)}
                  className="mt-4 px-5 py-2.5 rounded-xl bg-[#E5560B] text-white font-bold text-sm shadow-md hover:bg-[#F58B35] transition"
                >
                  Start Camera
                </button>
              </div>
            )}

            {/* Camera Error Alert (Real errors only) */}
            {cameraError && (
              <div className="absolute bottom-4 left-4 right-4 bg-red-500/90 backdrop-blur-md text-white p-3 rounded-xl text-xs font-medium z-30">
                {cameraError}
              </div>
            )}
          </div>

          {/* Camera Control Footer Bar */}
          <div className="flex items-center justify-between px-5 py-3 bg-[#1D1829] rounded-2xl border border-white/10 shadow-2xs">
            <div className="flex items-center gap-3">
              <button
                onClick={() => {
                  isCameraRunningRef.current = !isCameraOn
                  setIsCameraOn(!isCameraOn)
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition cursor-pointer ${
                  isCameraOn
                    ? "bg-[#251F35] text-amber-400 hover:bg-amber-400/18"
                    : "bg-[#E5560B] text-white hover:bg-[#F58B35]"
                }`}
              >
                {isCameraOn ? <CameraOff className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
                <span>{isCameraOn ? "Pause Feed" : "Resume Feed"}</span>
              </button>

              <button
                onClick={() => {
                  rollingBufferRef.current = []
                  setCurrentChar(null)
                  setCharConfidence(null)
                  toast.success("Buffer reset")
                }}
                className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold text-[#ACA5BC] hover:bg-[#251F35] transition cursor-pointer"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Reset Buffer</span>
              </button>
            </div>

            <div className="text-xs font-semibold text-[#ACA5BC] flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#E5560B]" />
              <span>21-Landmark 63→128→26 MLP</span>
            </div>
          </div>
        </div>

        {/* Right Column: Live Prediction & Sentence Builder (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          {/* Active Letter Tile Card */}
          <div className="sb-surface-card rounded-[26px] p-6 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[#E5560B]" />
                <span className="text-xs font-bold uppercase tracking-wider text-[#ACA5BC]">
                  Live Recognized Letter
                </span>
              </div>
              {charConfidence !== null ? (
                <span className="px-3 py-1 rounded-full text-xs font-black bg-[#E6F8EF] text-[#22B66B]">
                  {charConfidence}% Confidence
                </span>
              ) : (
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-zinc-100 text-zinc-500">
                  Waiting for sign...
                </span>
              )}
            </div>

            {/* Letter Hero Display */}
            <div className="flex items-center justify-between py-2 min-h-[90px]">
              <div className="flex items-baseline gap-4">
                <span className="text-7xl font-black text-[#FAF6F0] font-mono tracking-tight">
                  {currentChar ?? "—"}
                </span>
                <div className="flex flex-col">
                  <span className="text-xs font-bold text-[#ACA5BC]">ASL Character</span>
                  {charStability !== null && (
                    <span className="text-xs font-medium text-[#22B66B]">
                      Stability {charStability}%
                    </span>
                  )}
                </div>
              </div>

              <SignBridgeMascot
                variant="heart"
                size="sm"
                state={
                  isPlayingTTS
                    ? "speaking"
                    : currentChar
                      ? "detected"
                      : isHandDetected
                        ? "listening"
                        : "idle"
                }
              />
            </div>

            {/* Character History Tape */}
            <div className="mt-4 pt-4 border-t border-white/10">
              <span className="text-xs font-bold text-[#ACA5BC] block mb-2">
                Recent Letter Stream
              </span>
              <div className="flex items-center gap-2 overflow-x-auto pb-1 min-h-[38px]">
                {charHistory.length > 0 ? (
                  charHistory.map((ch, i) => (
                    <span
                      key={i}
                      className={`px-3 py-1.5 rounded-xl font-mono text-sm font-black transition ${
                        i === charHistory.length - 1
                          ? "bg-[#E5560B] text-white shadow-xs scale-105"
                          : "bg-[#251F35] text-[#FAF6F0] border border-amber-400/30"
                      }`}
                    >
                      {ch}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-[#ACA5BC]/70 italic">
                    Letters will stream here as you spell...
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Word & Sentence Builder Card */}
          <div className="sb-surface-card rounded-[26px] p-6 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-[#ACA5BC]">
                Sentence Builder
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleBackspace}
                  className="p-1.5 rounded-lg text-[#ACA5BC] hover:bg-[#251F35] hover:text-[#FAF6F0] transition cursor-pointer"
                  title="Backspace"
                >
                  <Delete className="w-4 h-4" />
                </button>
                <button
                  onClick={handleClear}
                  className="p-1.5 rounded-lg text-[#ACA5BC] hover:bg-[#251F35] hover:text-[#FAF6F0] transition cursor-pointer"
                  title="Clear All"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Assembled Words Display */}
            <div className="min-h-[100px] p-4 rounded-2xl bg-[#1D1829] border border-white/10 flex flex-wrap items-center content-start gap-2">
              {sentenceWords.map((word, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1.5 rounded-xl bg-[#1D1829] border border-white/10 text-sm font-bold text-[#FAF6F0] shadow-2xs"
                >
                  {word}
                </span>
              ))}

              {/* Active editing word */}
              {currentWord && (
                <span className="px-3 py-1.5 rounded-xl bg-amber-400/18 border border-amber-400/30 text-sm font-black text-amber-400 animate-pulse">
                  {currentWord}
                </span>
              )}

              {sentenceWords.length === 0 && !currentWord && (
                <span className="text-xs font-medium text-[#ACA5BC] italic">
                  Signed letters will assemble here automatically into words...
                </span>
              )}
            </div>

            {/* Builder Action Toolbar */}
            <div className="grid grid-cols-2 gap-2 pt-1">
              <button
                onClick={handleAddSpace}
                className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-[#251F35] text-amber-400 text-xs font-bold border border-amber-400/30 hover:bg-amber-400/18 transition cursor-pointer"
              >
                <span>+ Space (Next Word)</span>
              </button>

              <button
                onClick={handleCopy}
                className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-[#1D1829] text-[#FAF6F0] text-xs font-bold border border-white/10 hover:bg-[#2A233C] transition cursor-pointer"
              >
                {hasCopied ? <Check className="w-4 h-4 text-[#22B66B]" /> : <Copy className="w-4 h-4 text-[#ACA5BC]" />}
                <span>{hasCopied ? "Copied!" : "Copy Text"}</span>
              </button>
            </div>

            {/* Kokoro TTS Primary Speak Button */}
            <button
              onClick={handleSpeakTTS}
              disabled={isPlayingTTS || (sentenceWords.length === 0 && !currentWord)}
              className="w-full flex items-center justify-center gap-2.5 py-3 rounded-2xl bg-[#E5560B] text-white font-bold text-sm shadow-sm hover:bg-[#F58B35] transition disabled:opacity-40 cursor-pointer"
            >
              <Volume2 className={`w-4 h-4 ${isPlayingTTS ? "animate-bounce" : ""}`} />
              <span>{isPlayingTTS ? "Playing Speech..." : "Speak (Kokoro TTS)"}</span>
            </button>
          </div>

          {/* Model Specification Card */}
          <div className="p-4 rounded-2xl bg-[#251F35]/70 border border-amber-400/30 flex items-center justify-between text-xs text-[#ACA5BC]">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              <span className="font-bold text-[#FAF6F0]">ASL Alphabet Model</span>
            </div>
            <span className="font-mono text-[11px] text-amber-300 font-bold">
              26 Classes · MLP 63→128→26
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
