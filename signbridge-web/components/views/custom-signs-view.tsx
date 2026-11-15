"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import {
  Sparkles,
  Camera,
  CameraOff,
  CheckCircle2,
  Trash2,
  RefreshCw,
  Hand,
  Search,
  AlertCircle,
  Loader2,
  ShieldCheck,
  Play,
  Cpu,
  Plus
} from "lucide-react"
import { toast } from "sonner"
import {
  clearMediaPipeHandsListener,
  getSharedMediaPipeHands,
  MediaPipeHandsResults,
  sendVideoFrameToSharedHands,
  setMediaPipeHandsListener
} from "@/lib/mediapipe-hands"
import { useSignBridgeStore } from "@/lib/signbridge-store"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"

const QUICK_GLOSSES = [
  "HELLO",
  "THANKYOU",
  "HELP",
  "NAME",
  "FRIEND",
  "WATER",
  "YES",
  "NO"
]

function normalizeCustomGloss(raw: string): string {
  return raw
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 40)
}

function resampleTo32Frames(frames: number[][][]): number[][][] {
  const out: number[][][] = []
  const lastIdx = Math.max(0, frames.length - 1)
  for (let i = 0; i < 32; i++) {
    const idx = Math.min(lastIdx, Math.round((i * lastIdx) / 31))
    out.push(frames[idx])
  }
  return out
}

export function CustomSignsView({ initialGloss }: { initialGloss?: string | null }) {
  const prefersReduced = useReducedMotion()
  const {
    customSigns,
    personalizationEnabled,
    preferences,
    setPersonalizationEnabled,
    refreshPersonalization
  } = useSignBridgeStore()

  const isRoutingActive = Boolean(
    personalizationEnabled || preferences.personalizationEnabled
  )

  const [vocabList, setVocabList] = React.useState<string[]>([])
  const [glossQuery, setGlossQuery] = React.useState("")
  const [selectedGloss, setSelectedGloss] = React.useState(
    normalizeCustomGloss(initialGloss || "FRIEND") || "FRIEND"
  )
  const [prevInitialGloss, setPrevInitialGloss] = React.useState(initialGloss)

  if (initialGloss !== prevInitialGloss) {
    setPrevInitialGloss(initialGloss)
    if (initialGloss) {
      const normalized = normalizeCustomGloss(initialGloss)
      if (normalized) setSelectedGloss(normalized)
    }
  }

  const [isCameraOn, setIsCameraOn] = React.useState(true)
  const [cameraError, setCameraError] = React.useState<string | null>(null)
  const [isHandTracked, setIsHandTracked] = React.useState(false)
  const [captureKind, setCaptureKind] = React.useState<"sample" | "test">("sample")
  const [recordingState, setRecordingState] = React.useState<
    "idle" | "recording" | "uploading"
  >("idle")
  const [recordProgress, setRecordProgress] = React.useState(0)
  const [isTraining, setIsTraining] = React.useState(false)
  const [lastFeedback, setLastFeedback] = React.useState<{
    accepted: boolean
    score?: number
    message: string
  } | null>(null)
  const [liveTestResult, setLiveTestResult] = React.useState<{
    gloss: string
    confidence: number
    matchedCustom: boolean
    similarity?: number | null
    topK: Array<{ gloss: string; confidence: number }>
  } | null>(null)

  const videoRef = React.useRef<HTMLVideoElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const streamRef = React.useRef<MediaStream | null>(null)
  const rafRef = React.useRef<number | null>(null)
  const isMountedRef = React.useRef(true)
  const isProcessingRef = React.useRef(false)
  const handsReadyRef = React.useRef(false)
  const recordingRef = React.useRef(false)
  const capturedFramesRef = React.useRef<number[][][]>([])
  const lastFrameTimeRef = React.useRef(0)

  React.useEffect(() => {
    const t = window.setTimeout(() => {
      fetch("/api/vocabulary?limit=2731")
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (Array.isArray(data?.items)) {
            setVocabList(
              data.items
                .map((item: { gloss?: string }) => String(item.gloss || "").toUpperCase())
                .filter(Boolean)
            )
          }
        })
        .catch(() => {})
    }, 0)
    return () => window.clearTimeout(t)
  }, [])

  const drawOverlayAndCollect = React.useCallback((results: MediaPipeHandsResults) => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (canvas && video && video.videoWidth > 0 && video.videoHeight > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        const hands = results.multiHandLandmarks || []
        hands.forEach((landmarks) => {
          landmarks.forEach((pt, idx) => {
            const px = pt.x * canvas.width
            const py = pt.y * canvas.height
            const isTip = [4, 8, 12, 16, 20].includes(idx)
            ctx.beginPath()
            ctx.arc(px, py, isTip ? 4.5 : 3, 0, Math.PI * 2)
            ctx.fillStyle = isTip ? "#FF7900" : "#FFFFFF"
            ctx.fill()
          })
        })
      }
    }

    const hands = results.multiHandLandmarks || []
    setIsHandTracked(hands.length > 0)

    const frameLandmarks: number[][] = Array.from({ length: 42 }, () => [0, 0, 0])
    for (let h = 0; h < Math.min(2, hands.length); h++) {
      const lm = hands[h]
      const offset = h * 21
      for (let i = 0; i < Math.min(21, lm.length); i++) {
        frameLandmarks[offset + i] = [lm[i].x, lm[i].y, lm[i].z ?? 0]
      }
    }

    if (recordingRef.current) {
      capturedFramesRef.current.push(frameLandmarks)
    }
  }, [])

  const onResultsRef = React.useRef(drawOverlayAndCollect)
  React.useEffect(() => {
    onResultsRef.current = drawOverlayAndCollect
  }, [drawOverlayAndCollect])

  const stableListener = React.useCallback((results: MediaPipeHandsResults) => {
    onResultsRef.current(results)
  }, [])

  React.useEffect(() => {
    isMountedRef.current = true
    if (!isCameraOn) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      clearMediaPipeHandsListener(stableListener)
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      return
    }

    const init = async () => {
      try {
        setCameraError(null)
        setMediaPipeHandsListener(stableListener)
        await getSharedMediaPipeHands({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        })
        handsReadyRef.current = true

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false
        })
        if (!isMountedRef.current) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play().catch(() => {})
        }

        const loop = async () => {
          if (!isMountedRef.current || !isCameraOn) return
          const now = performance.now()
          if (
            now - lastFrameTimeRef.current >= 34 &&
            videoRef.current &&
            videoRef.current.readyState >= 2 &&
            handsReadyRef.current &&
            !isProcessingRef.current
          ) {
            isProcessingRef.current = true
            lastFrameTimeRef.current = now
            try {
              await sendVideoFrameToSharedHands(videoRef.current, {
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
              })
            } catch {
              // Skip frame cleanly
            } finally {
              isProcessingRef.current = false
            }
          }
          rafRef.current = requestAnimationFrame(loop)
        }
        rafRef.current = requestAnimationFrame(loop)
      } catch (err) {
        setCameraError(err instanceof Error ? err.message : "Camera unavailable")
      }
    }

    void init()

    return () => {
      isMountedRef.current = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      clearMediaPipeHandsListener(stableListener)
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [isCameraOn, stableListener])

  const finalizeCapture = async (kind: "sample" | "test") => {
    recordingRef.current = false
    setRecordingState("uploading")

    const raw = [...capturedFramesRef.current]
    if (raw.length < 10) {
      setRecordingState("idle")
      setRecordProgress(0)
      setLastFeedback({
        accepted: false,
        message: "Not enough frames recorded. Keep your hands visible and sign clearly."
      })
      toast.error("Recording too short — keep hands in view and retry")
      return
    }

    const sampled32 = resampleTo32Frames(raw)
    const userId = preferences.userId
    if (!userId) {
      toast.info("Initializing profile — please retry in a moment")
      return
    }
    const modelVariant = preferences.modelVariant || "production_exp13"

    try {
      if (kind === "test") {
        const res = await fetch("/api/predict/sequence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sequence: sampled32,
            model_variant: modelVariant,
            user_id: userId,
            personalization_enabled: true,
            top_k: 5
          })
        })
        const data = await res.json().catch(() => ({}))
        if (!res.ok) {
          toast.error(data?.detail || `Test prediction failed (${res.status})`)
          return
        }
        const pers = data?.personalization || {}
        const rawConf = Number(data?.confidence ?? 0)
        const confPct = rawConf > 0 && rawConf <= 1 ? rawConf * 100 : rawConf
        const topK = Array.isArray(data?.top_k)
          ? data.top_k.slice(0, 4).map((item: { gloss?: string; confidence?: number }) => {
              const c = Number(item?.confidence ?? 0)
              return {
                gloss: String(item?.gloss || ""),
                confidence: c > 0 && c <= 1 ? c * 100 : c
              }
            })
          : []
        const predictedGloss = String(data?.predicted_gloss || data?.gloss || "")
        const matchedCustom = Boolean(
          data?.is_custom_sign || pers?.matched || pers?.personalized
        )
        const sim =
          typeof pers?.personalized_similarity === "number"
            ? pers.personalized_similarity
            : null
        setLiveTestResult({
          gloss: predictedGloss || "NO_SIGN",
          confidence: confPct,
          matchedCustom,
          similarity: sim,
          topK
        })
        if (matchedCustom) {
          toast.success(`Custom Model recognized "${predictedGloss}" (${Math.round(confPct)}%)`)
        } else {
          toast.info(`Predicted "${predictedGloss}" (${Math.round(confPct)}%)`)
        }
        return
      }

      const targetGloss = normalizeCustomGloss(selectedGloss)
      if (!targetGloss) {
        toast.info("Enter a valid custom sign name first")
        return
      }
      const res = await fetch("/api/personalization/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gloss: targetGloss,
          sequence: sampled32,
          user_id: userId,
          model_variant: modelVariant
        })
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const msg =
          data?.quality?.feedback ||
          data?.detail ||
          `Sample rejected (${res.status})`
        setLastFeedback({
          accepted: false,
          score: data?.quality?.score,
          message: msg
        })
        toast.error(msg)
        return
      }

      const duplicate = Boolean(data?.sample?.duplicate)
      const total = data?.sample?.total_samples ?? 0
      const msg = duplicate
        ? `Duplicate motion ignored for ${targetGloss} (${total} saved)`
        : `Sample saved for ${targetGloss} · Quality ${Math.round(data?.quality?.score ?? 100)}% (${total}/3)`

      setLastFeedback({
        accepted: true,
        score: data?.quality?.score,
        message: msg
      })
      toast.success(msg)
      await setPersonalizationEnabled(true)
    } catch {
      toast.error(
        kind === "test"
          ? "Failed to run custom model test prediction"
          : "Failed to upload sample to personalization backend"
      )
    } finally {
      setRecordingState("idle")
      setRecordProgress(0)
    }
  }

  const startTimedCapture = (kind: "sample" | "test") => {
    if (!isCameraOn || recordingState !== "idle") return
    if (kind === "sample" && !normalizeCustomGloss(selectedGloss)) {
      toast.info("Enter or choose a custom sign name first")
      return
    }
    setCaptureKind(kind)
    capturedFramesRef.current = []
    recordingRef.current = true
    setRecordingState("recording")
    setRecordProgress(0)

    const start = performance.now()
    const durationMs = 2200
    const tick = window.setInterval(() => {
      const pct = Math.min(100, Math.round(((performance.now() - start) / durationMs) * 100))
      setRecordProgress(pct)
      if (pct >= 100) {
        window.clearInterval(tick)
        void finalizeCapture(kind)
      }
    }, 80)
  }

  const handleTrainCustomModel = async () => {
    if (isTraining || !preferences.userId) return
    setIsTraining(true)
    try {
      const res = await fetch("/api/personalization/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: preferences.userId })
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        toast.error(data?.detail || `Training failed (${res.status})`)
        return
      }
      await setPersonalizationEnabled(Boolean(data?.enabled ?? true))
      const rawTrained = Array.isArray(data?.trained_signs) ? data.trained_signs : []
      const trainedNames: string[] = rawTrained
        .map((item: string | { gloss?: string }) =>
          typeof item === "string" ? item : String(item?.gloss || "")
        )
        .filter(Boolean)
      const totalSigns = Number(data?.total_signs ?? trainedNames.length)
      const totalSamples = Number(data?.total_samples ?? 0)
      const msg =
        totalSigns > 0
          ? `Custom model trained (${trainedNames.join(", ")} · ${totalSamples} samples) — active in Live Studio!`
          : "Record samples for at least one custom sign first."
      setLastFeedback({ accepted: totalSigns > 0, message: msg })
      if (totalSigns > 0) {
        toast.success(msg)
      } else {
        toast.info(msg)
      }
    } catch {
      toast.error("Backend unreachable while training custom model")
    } finally {
      setIsTraining(false)
    }
  }

  const handleTogglePersonalization = async (next: boolean) => {
    await setPersonalizationEnabled(next)
    toast.success(
      next
        ? "Custom sign prediction enabled in Live Studio"
        : "Custom sign prediction paused"
    )
  }

  const handleForgetSign = async (gloss: string) => {
    if (!preferences.userId) return
    try {
      const res = await fetch(
        `/api/personalization/sign/${encodeURIComponent(gloss)}?user_id=${encodeURIComponent(
          preferences.userId
        )}`,
        { method: "DELETE" }
      )
      if (!res.ok) throw new Error("Delete failed")
      toast.success(`Removed custom samples for ${gloss}`)
      await refreshPersonalization()
    } catch {
      toast.error(`Unable to remove ${gloss}`)
    }
  }

  const handleResetAll = async () => {
    if (!preferences.userId) return
    try {
      const res = await fetch(
        `/api/personalization?user_id=${encodeURIComponent(preferences.userId)}`,
        { method: "DELETE" }
      )
      if (!res.ok) throw new Error("Reset failed")
      await setPersonalizationEnabled(false)
      setLiveTestResult(null)
      toast.success("Cleared all custom sign prototypes")
    } catch {
      toast.error("Unable to reset custom signs")
    }
  }

  const customQueryNormalized = React.useMemo(
    () => normalizeCustomGloss(glossQuery),
    [glossQuery]
  )

  const filteredGlosses = React.useMemo(() => {
    const q = glossQuery.trim().toUpperCase()
    const trainedNames = customSigns.map((s) => s.gloss.toUpperCase())
    const mergedQuick = Array.from(new Set([...trainedNames, ...QUICK_GLOSSES]))
    const source = vocabList.length > 0 ? Array.from(new Set([...trainedNames, ...vocabList])) : mergedQuick
    if (!q) return mergedQuick
    return source.filter((g) => g.includes(q)).slice(0, 12)
  }, [glossQuery, vocabList, customSigns])

  return (
    <div className="w-full flex flex-col gap-5 pb-12">
      {/* Studio Banner */}
      <section className="sb-glass-card rounded-[26px] px-5 sm:px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <SignBridgeMascot
            variant="wave"
            size="sm"
            state={recordingState === "recording" ? "detected" : isHandTracked ? "listening" : "idle"}
          />
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#FAF6F0]">
                Custom Sign Learner
              </h1>
              <span className="px-3 py-0.5 rounded-full bg-[#251F35] border border-amber-400/30 text-[11px] font-bold text-amber-300">
                Few-Shot Conformer Learner
              </span>
            </div>
            <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Type any custom sign name or ASL word, record 3+ samples, train your personal model, and test live prediction.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          <button
            type="button"
            disabled={isTraining || customSigns.length === 0}
            onClick={() => void handleTrainCustomModel()}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold border border-amber-400/40 bg-amber-400/15 text-amber-200 hover:bg-amber-400/25 disabled:opacity-45 transition cursor-pointer"
          >
            {isTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <Cpu className="w-4 h-4" />}
            <span>{isTraining ? "Training Model..." : "Train Custom Model"}</span>
          </button>

          <button
            type="button"
            onClick={() => void handleTogglePersonalization(!isRoutingActive)}
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold border transition cursor-pointer ${
              isRoutingActive
                ? "sb-orange-pill border-transparent text-white"
                : "bg-[#1D1829] border-white/10 text-[#FAF6F0] hover:bg-[#251F35]"
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            <span>
              Prediction: {isRoutingActive ? "Active in Live" : "Standby"}
            </span>
          </button>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left: Target Custom Sign + Webcam Recording & Live Tester (7 cols) */}
        <section className="lg:col-span-7 sb-surface-card rounded-[26px] p-5 space-y-4">
          {/* Custom Sign Selection */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs font-bold uppercase tracking-wider text-[#ACA5BC]">
                1. Create Custom Sign or Choose ASL Word ({vocabList.length || 2731} supported)
              </label>
              <span className="px-2.5 py-0.5 rounded-full bg-[#251F35] text-amber-300 font-mono text-xs font-bold">
                Target: {selectedGloss}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-[#7E7890] absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  value={glossQuery}
                  onChange={(e) => setGlossQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && customQueryNormalized) {
                      setSelectedGloss(customQueryNormalized)
                      setGlossQuery("")
                    }
                  }}
                  placeholder="Type a new custom sign name (e.g. COFFEE, MY_NAME) or search ASL..."
                  className="w-full rounded-full bg-[#161221] border border-white/10 pl-10 pr-4 py-2 text-xs font-semibold text-[#FAF6F0] placeholder:text-[#7E7890] focus:outline-none focus:ring-2 focus:ring-amber-400"
                />
              </div>
              {customQueryNormalized && customQueryNormalized !== selectedGloss && (
                <button
                  type="button"
                  onClick={() => {
                    setSelectedGloss(customQueryNormalized)
                    setGlossQuery("")
                  }}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-amber-400/15 border border-amber-400/35 text-xs font-bold text-amber-200 hover:bg-amber-400/25 shrink-0 cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Use &ldquo;{customQueryNormalized}&rdquo;</span>
                </button>
              )}
            </div>

            <div className="flex flex-wrap gap-1.5">
              {filteredGlosses.map((g) => (
                <button
                  key={g}
                  type="button"
                  onClick={() => setSelectedGloss(g)}
                  className={`px-3 py-1 rounded-full text-xs font-bold border transition cursor-pointer ${
                    selectedGloss === g
                      ? "sb-orange-pill border-transparent text-white"
                      : "bg-[#1D1829] border-white/10 text-[#FAF6F0] hover:bg-[#251F35]"
                  }`}
                >
                  {g}
                </button>
              ))}
            </div>
          </div>

          {/* Camera Stage */}
          <div className="relative aspect-video rounded-[20px] bg-[#120F1B] border border-white/10 overflow-hidden flex items-center justify-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover scale-x-[-1] ${
                isCameraOn && !cameraError ? "opacity-100" : "opacity-0"
              }`}
            />
            <canvas
              ref={canvasRef}
              className={`pointer-events-none absolute inset-0 w-full h-full object-cover scale-x-[-1] ${
                isCameraOn && !cameraError ? "opacity-100" : "opacity-0"
              }`}
            />

            <div className="absolute top-3 left-3 z-10 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/65 backdrop-blur-md text-white text-xs font-bold border border-white/15">
              <Hand className={`w-3.5 h-3.5 ${isHandTracked ? "text-[#1FA968]" : "text-white/50"}`} />
              <span>{isHandTracked ? "Hands Detected" : "Show Hands"}</span>
            </div>

            {(!isCameraOn || cameraError) && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center text-white">
                <CameraOff className="w-8 h-8 text-amber-300 mb-2" />
                <p className="text-xs text-white/80">
                  {cameraError || "Camera paused"}
                </p>
              </div>
            )}

            {recordingState === "recording" && (
              <div className="absolute bottom-4 inset-x-6 z-20 p-3 rounded-2xl bg-black/80 backdrop-blur-md border border-amber-400 text-white space-y-1.5">
                <div className="flex items-center justify-between text-xs font-bold">
                  <span>
                    {captureKind === "test"
                      ? "Capturing live test sign (2.2s)..."
                      : `Recording “${selectedGloss}” sample (2.2s)...`}
                  </span>
                  <span className="font-mono tabular-nums">{recordProgress}%</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-white/20 overflow-hidden">
                  <div
                    className="h-full bg-[#D97706] transition-all duration-75"
                    style={{ width: `${recordProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center justify-between gap-2.5">
            <button
              type="button"
              onClick={() => setIsCameraOn((v) => !v)}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#161221] border border-white/10 text-xs font-bold text-[#FAF6F0] hover:bg-[#251F35] cursor-pointer"
            >
              {isCameraOn ? <CameraOff className="w-3.5 h-3.5" /> : <Camera className="w-3.5 h-3.5" />}
              <span>{isCameraOn ? "Pause Camera" : "Resume Camera"}</span>
            </button>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                disabled={!isCameraOn || Boolean(cameraError) || recordingState !== "idle"}
                onClick={() => startTimedCapture("test")}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full bg-[#241E34] border border-white/15 text-xs font-bold text-[#FAF6F0] hover:bg-[#2E2642] disabled:opacity-45 transition cursor-pointer"
              >
                <Play className="w-3.5 h-3.5 text-emerald-400" />
                <span>
                  {recordingState === "recording" && captureKind === "test"
                    ? "Testing..."
                    : "Live Test Sign (2.2s)"}
                </span>
              </button>

              <motion.button
                type="button"
                whileTap={prefersReduced ? undefined : { scale: 0.97 }}
                disabled={!isCameraOn || Boolean(cameraError) || recordingState !== "idle"}
                onClick={() => startTimedCapture("sample")}
                className="sb-orange-pill inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-bold text-white disabled:opacity-45 cursor-pointer"
              >
                {recordingState === "uploading" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                <span>
                  {recordingState === "recording" && captureKind === "sample"
                    ? "Capturing..."
                    : recordingState === "uploading"
                      ? captureKind === "test"
                        ? "Predicting Sign..."
                        : "Evaluating Sample..."
                      : `Record "${selectedGloss}" Sample`}
                </span>
              </motion.button>
            </div>
          </div>

          {liveTestResult && (
            <div className="p-3.5 rounded-2xl bg-[#161221] border border-amber-400/30 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-extrabold text-white">
                    {liveTestResult.gloss}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                      liveTestResult.matchedCustom
                        ? "bg-emerald-500/15 text-emerald-300 border border-emerald-400/30"
                        : "bg-white/10 text-[#C8C2D6]"
                    }`}
                  >
                    {liveTestResult.matchedCustom ? "Custom Model Match" : "Base Model"}
                  </span>
                </div>
                <span className="font-mono text-xs font-bold text-amber-300">
                  {Math.round(liveTestResult.confidence)}%
                  {typeof liveTestResult.similarity === "number"
                    ? ` · sim ${liveTestResult.similarity.toFixed(2)}`
                    : ""}
                </span>
              </div>
              {liveTestResult.topK.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {liveTestResult.topK.map((cand) => (
                    <span
                      key={cand.gloss}
                      className="rounded-full bg-[#221C31] px-2.5 py-0.5 text-[11px] font-medium text-[#C8C2D6]"
                    >
                      {cand.gloss} · {Math.round(cand.confidence)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {lastFeedback && (
            <div
              className={`p-3.5 rounded-2xl border text-xs flex items-start gap-2.5 ${
                lastFeedback.accepted
                  ? "bg-emerald-500/12 border-emerald-400/30 text-emerald-300"
                  : "bg-rose-500/12 border-rose-400/30 text-rose-300"
              }`}
            >
              {lastFeedback.accepted ? (
                <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
              ) : (
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              )}
              <span className="font-semibold leading-relaxed">{lastFeedback.message}</span>
            </div>
          )}
        </section>

        {/* Right: Saved Personalized Signs & Prototypes (5 cols) */}
        <section className="lg:col-span-5 sb-surface-card rounded-[26px] p-5 flex flex-col justify-between gap-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h2 className="text-base font-extrabold text-[#FAF6F0]">
                  Saved Personalized Signs
                </h2>
                <p className="text-[11px] text-[#ACA5BC]">
                  Prototype routing activates once a gloss reaches 3 valid samples
                </p>
              </div>
              <button
                type="button"
                onClick={() => void refreshPersonalization()}
                aria-label="Refresh personalized signs"
                className="p-2 rounded-full bg-[#161221] border border-white/10 text-[#ACA5BC] hover:text-amber-400 cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>

            {customSigns.length === 0 ? (
              <div className="p-6 rounded-2xl bg-[#161221] border border-dashed border-white/12 text-center space-y-2">
                <Sparkles className="w-6 h-6 text-amber-400 mx-auto" />
                <div className="text-xs font-bold text-[#FAF6F0]">
                  No custom signs recorded yet
                </div>
                <p className="text-[11px] text-[#ACA5BC] leading-relaxed">
                  Pick a gloss on the left and record 3 short signing samples to personalize recognition.
                </p>
              </div>
            ) : (
              <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
                {customSigns.map((sign) => {
                  const pct = Math.min(100, Math.round((sign.active_samples / 3) * 100))
                  const isLearned = sign.active_samples >= 3 && sign.has_prototype
                  return (
                    <div
                      key={sign.gloss}
                      onClick={() => setSelectedGloss(sign.gloss)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") setSelectedGloss(sign.gloss)
                      }}
                      className={`p-3.5 rounded-2xl bg-[#161221] border space-y-2 transition cursor-pointer ${
                        selectedGloss === sign.gloss
                          ? "border-amber-400/55"
                          : "border-white/10 hover:border-white/25"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm font-extrabold text-[#FAF6F0]">
                            {sign.gloss}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                              isLearned
                                ? "bg-emerald-500/12 text-emerald-300"
                                : "bg-[#251F35] text-amber-300"
                            }`}
                          >
                            {isLearned ? "Model Ready" : `${sign.active_samples}/3 Samples`}
                          </span>
                        </div>

                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            void handleForgetSign(sign.gloss)
                          }}
                          aria-label={`Delete custom sign ${sign.gloss}`}
                          className="p-1.5 rounded-full hover:bg-[#1D1829] text-[#7E7890] hover:text-[#EF4444] transition cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            isLearned ? "bg-[#1FA968]" : "bg-[#D97706]"
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {customSigns.length > 0 && (
            <div className="pt-3 border-t border-white/10 flex items-center justify-between">
              <span className="text-[11px] font-medium text-[#ACA5BC]">
                Stored under profile: <strong className="font-mono">{preferences.userId}</strong>
              </span>
              <button
                type="button"
                onClick={() => void handleResetAll()}
                className="text-xs font-bold text-[#EF4444] hover:underline cursor-pointer"
              >
                Reset All
              </button>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
