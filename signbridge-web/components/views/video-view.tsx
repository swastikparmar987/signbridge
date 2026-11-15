"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import {
  UploadCloud,
  FileVideo,
  Volume2,
  Download,
  Sparkles,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Cpu
} from "lucide-react"
import { toast } from "sonner"
import { useSignBridgeStore } from "@/lib/signbridge-store"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"

interface VideoCandidate {
  gloss: string
  confidence: number
}

interface VideoAnalysisResult {
  filename: string
  predicted_gloss: string
  confidence: number
  latency_ms: number | null
  top_k: VideoCandidate[]
}

function cleanPct(raw: unknown): number {
  const n = typeof raw === "number" ? raw : parseFloat(String(raw ?? 0)) || 0
  return Math.min(99.9, Math.max(0.1, Number(n.toFixed(1))))
}

function formatGloss(gloss: string): string {
  return gloss
    .trim()
    .replace(/\d+$/, "")
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ")
}

export function VideoAnalysisView() {
  const prefersReduced = useReducedMotion()
  const { preferences, modelHealth, reportInferenceTelemetry, recordSession } = useSignBridgeStore()

  const fileInputRef = React.useRef<HTMLInputElement | null>(null)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)

  const [selectedFile, setSelectedFile] = React.useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null)
  const [isDragging, setIsDragging] = React.useState(false)
  const [isUploading, setIsUploading] = React.useState(false)
  const [errorMsg, setErrorMsg] = React.useState<string | null>(null)
  const [result, setResult] = React.useState<VideoAnalysisResult | null>(null)
  const [isSpeaking, setIsSpeaking] = React.useState(false)

  React.useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      if (audioRef.current) audioRef.current.pause()
    }
  }, [previewUrl])

  const analyzeFile = async (file: File) => {
    const ext = file.name.toLowerCase().split(".").pop() || ""
    if (!["mp4", "mov", "webm", "avi", "m4v"].includes(ext)) {
      const msg = "Unsupported format. Please upload MP4, MOV, WebM, AVI, or M4V."
      setErrorMsg(msg)
      toast.error(msg)
      return
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(file))
    setSelectedFile(file)
    setErrorMsg(null)
    setResult(null)
    setIsUploading(true)

    try {
      const formData = new FormData()
      formData.append("file", file)
      formData.append("top_k", "5")
      formData.append("model_variant", preferences.modelVariant || "production_exp13")

      const start = performance.now()
      const res = await fetch("/api/predict/video", {
        method: "POST",
        body: formData
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const msg = data?.detail || `Video prediction failed (${res.status})`
        setErrorMsg(msg)
        toast.error(msg)
        return
      }

      const latency =
        typeof data.inference_time_ms === "number"
          ? data.inference_time_ms
          : typeof data.latency_ms === "number"
            ? data.latency_ms
            : Math.round(performance.now() - start)

      const topKRaw = Array.isArray(data.top_k) ? data.top_k : []
      const topK: VideoCandidate[] = topKRaw.slice(0, 5).map(
        (item: { gloss?: string; confidence?: number }) => ({
          gloss: formatGloss(String(item.gloss || "")),
          confidence: cleanPct(item.confidence)
        })
      )

      const predGloss = formatGloss(String(data.predicted_gloss || data.gloss || topK[0]?.gloss || ""))
      const conf = cleanPct(data.confidence ?? topK[0]?.confidence ?? 0)

      setResult({
        filename: String(data.filename || file.name),
        predicted_gloss: predGloss || "No Sign Detected",
        confidence: conf,
        latency_ms: latency,
        top_k: topK
      })

      reportInferenceTelemetry(latency, conf)
      if (predGloss) {
        recordSession({
          mode: "Word",
          text: predGloss,
          glosses: [predGloss],
          confidence: conf,
          title: `Video Clip – ${file.name.slice(0, 28)}`
        })
      }
      toast.success(
        `Analyzed "${file.name}" via 32-frame ${
          preferences.modelVariant === "demo_50" ? "Demo-50" : "Ensemble V2"
        } inference`
      )
    } catch {
      const msg = "Unable to reach SignBridge video inference backend."
      setErrorMsg(msg)
      toast.error(msg)
    } finally {
      setIsUploading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) void analyzeFile(f)
  }

  const handleSpeakResult = async () => {
    if (!result?.predicted_gloss || isSpeaking) return
    setIsSpeaking(true)
    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: result.predicted_gloss,
          voice: preferences.ttsVoice || "af_sarah"
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
          audio.onerror = () => setIsSpeaking(false)
          await audio.play()
          return
        }
      }
    } catch {
      // Fallback below
    }
    if ("speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(result.predicted_gloss)
      u.onend = () => setIsSpeaking(false)
      u.onerror = () => setIsSpeaking(false)
      window.speechSynthesis.speak(u)
    } else {
      setIsSpeaking(false)
    }
  }

  const handleExportJson = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${result.filename.replace(/\.[^.]+$/, "")}_signbridge_prediction.json`
    a.click()
    URL.revokeObjectURL(url)
    toast.success("Downloaded prediction JSON report")
  }

  return (
    <div className="w-full flex flex-col gap-5 pb-12">
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp4,.mov,.webm,.avi,.m4v,video/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void analyzeFile(f)
        }}
      />

      {/* Top Banner */}
      <section className="sb-glass-card rounded-[26px] px-5 sm:px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl sb-orange-pill text-white flex items-center justify-center shrink-0">
            <UploadCloud className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#FAF6F0]">
                Video Clip Inspector
              </h1>
              <span className="px-3 py-0.5 rounded-full bg-[#251F35] border border-amber-400/30 font-mono text-[11px] font-bold text-amber-300">
                MP4 · MOV · WebM · M4V
              </span>
            </div>
            <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Upload a recorded ASL clip to run real server-side 32-frame MediaPipe extraction & EXP13 inference.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={handleExportJson}
          disabled={!result}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#2C2440] hover:bg-[#382E50] hover:bg-[#332720] text-white text-xs font-bold disabled:opacity-35 transition cursor-pointer shrink-0 self-start sm:self-auto"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export JSON</span>
        </button>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left: Video Player & Dropzone (7 cols) */}
        <section className="lg:col-span-7 sb-surface-card rounded-[26px] p-5 space-y-4">
          {previewUrl ? (
            <div className="relative aspect-video rounded-[20px] overflow-hidden bg-[#120F1B] border border-white/10">
              <video
                src={previewUrl}
                controls
                playsInline
                className="w-full h-full object-contain"
              />
            </div>
          ) : (
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`aspect-video rounded-[20px] border-2 border-dashed flex flex-col items-center justify-center p-8 text-center transition cursor-pointer ${
                isDragging
                  ? "border-amber-400 bg-[#251F35]"
                  : "border-[#E6D2C0] bg-[#161221] hover:border-amber-400 hover:bg-[#251F35]/60"
              }`}
            >
              <SignBridgeMascot size="md" variant="wave" state="idle" />
              <h2 className="text-base font-extrabold text-[#FAF6F0] mt-3">
                Drop an ASL Video Clip Here
              </h2>
              <p className="text-xs text-[#ACA5BC] mt-1 max-w-sm leading-relaxed">
                Supports single-sign video files (.mp4, .mov, .webm, .avi, .m4v) processed through{" "}
                <code className="font-mono text-amber-300">/api/predict/video</code>.
              </p>
              <span className="sb-orange-pill mt-4 inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-bold text-white">
                <UploadCloud className="w-4 h-4" />
                <span>Browse Video File</span>
              </span>
            </div>
          )}

          {/* Action Footer */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-[#ACA5BC] min-w-0">
              <FileVideo className="w-4 h-4 text-amber-400 shrink-0" />
              <span className="truncate font-semibold">
                {selectedFile ? selectedFile.name : "No clip selected yet"}
              </span>
            </div>

            <div className="flex items-center gap-2">
              {selectedFile && (
                <button
                  type="button"
                  onClick={() => void analyzeFile(selectedFile)}
                  disabled={isUploading}
                  className="px-4 py-2 rounded-full bg-[#251F35] hover:bg-amber-400/18 border border-amber-400/30 text-xs font-bold text-amber-300 cursor-pointer disabled:opacity-50"
                >
                  Re-run Inference
                </button>
              )}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="sb-orange-pill px-4 py-2 rounded-full text-xs font-bold text-white cursor-pointer disabled:opacity-50"
              >
                {isUploading ? "Analyzing..." : "Choose Clip"}
              </button>
            </div>
          </div>

          {errorMsg && (
            <div className="p-3.5 rounded-2xl bg-rose-500/12 border border-rose-400/30 text-xs font-semibold text-rose-300 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </section>

        {/* Right: Real Top-5 Prediction Breakdown (5 cols) */}
        <section className="lg:col-span-5 sb-surface-card rounded-[26px] p-5 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#ACA5BC]">
                Clip Classification
              </span>
              <h2 className="text-base font-extrabold text-[#FAF6F0] mt-0.5">
                EXP13 Inference Output
              </h2>
            </div>

            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#161221] border border-white/10 text-[11px] font-mono font-bold text-[#ACA5BC]">
              <Cpu className="w-3.5 h-3.5 text-amber-400" />
              <span>{modelHealth.hardwareDevice ?? "Backend"}</span>
            </div>
          </div>

          {isUploading ? (
            <div className="py-12 flex flex-col items-center justify-center text-center gap-3">
              <Loader2 className="w-8 h-8 text-amber-400 animate-spin" />
              <div className="text-sm font-bold text-[#FAF6F0]">
                Extracting 32-frame MediaPipe sequence...
              </div>
              <p className="text-xs text-[#ACA5BC]">
                Running server-side pose normalization & GRU classification
              </p>
            </div>
          ) : result ? (
            <motion.div
              initial={prefersReduced ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-5"
            >
              {/* Primary Gloss Banner */}
              <div className="p-4 rounded-2xl bg-[#161221] border border-amber-400/30 flex items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-1.5 text-[11px] font-bold text-emerald-300">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Top Predicted Sign</span>
                  </div>
                  <div className="text-3xl font-black text-[#FAF6F0] tracking-tight mt-1">
                    {result.predicted_gloss}
                  </div>
                  <div className="text-xs font-mono text-[#ACA5BC] mt-1 tabular-nums">
                    {result.confidence.toFixed(1)}% confidence
                    {result.latency_ms !== null ? ` · ${result.latency_ms} ms` : ""}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => void handleSpeakResult()}
                  disabled={isSpeaking}
                  className="sb-orange-pill px-4 py-2 rounded-full text-xs font-bold text-white inline-flex items-center gap-1.5 cursor-pointer shrink-0"
                >
                  <Volume2 className="w-4 h-4" />
                  <span>{isSpeaking ? "Speaking..." : "Speak"}</span>
                </button>
              </div>

              {/* Ranked Top-5 Candidates */}
              <div className="space-y-2.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#ACA5BC] block">
                  Top-5 Vocabulary Candidates
                </span>
                {result.top_k.map((item, idx) => (
                  <div
                    key={`${item.gloss}-${idx}`}
                    className="p-3 rounded-2xl bg-[#1D1829] border border-white/10 space-y-1.5"
                  >
                    <div className="flex items-center justify-between text-xs font-bold">
                      <span className="text-[#FAF6F0]">
                        #{idx + 1} {item.gloss}
                      </span>
                      <span className="font-mono text-amber-400 tabular-nums">
                        {item.confidence.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-[#2C2440] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-[#FF9436] to-[#E85D04]"
                        style={{ width: `${Math.min(100, item.confidence)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          ) : (
            <div className="p-6 rounded-2xl bg-[#161221] border border-dashed border-white/12 text-center space-y-2">
              <Sparkles className="w-6 h-6 text-amber-400 mx-auto" />
              <div className="text-xs font-bold text-[#FAF6F0]">Waiting for a video clip</div>
              <p className="text-[11px] text-[#ACA5BC] leading-relaxed">
                Upload an isolated ASL sign clip on the left to inspect real Top-5 gloss predictions and latency.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
