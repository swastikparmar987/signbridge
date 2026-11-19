"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import {
  Mic,
  MicOff,
  Volume2,
  Camera,
  Sparkles,
  Send,
  RotateCcw,
  Languages,
  ArrowDownUp,
  Hand,
  Copy,
  Check,
  Clock
} from "lucide-react"
import { toast } from "sonner"
import { useAslRecognition } from "@/lib/use-asl-recognition"
import { useSignBridgeStore } from "@/lib/signbridge-store"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"

interface ConversationEntry {
  id: string
  speaker: "signer" | "hearing"
  text: string
  glosses?: string[]
  timestamp: string
}

const QUICK_CHIPS = [
  "Hello, how can I help you?",
  "Nice to meet you",
  "Please repeat that slower",
  "Thank you very much"
]

interface SpeechRecognitionAlternativeLike {
  transcript: string
}
interface SpeechRecognitionResultLike {
  0: SpeechRecognitionAlternativeLike
  isFinal: boolean
}
interface SpeechRecognitionEventLike {
  results: ArrayLike<SpeechRecognitionResultLike>
}
interface BrowserSpeechRecognition {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((e: SpeechRecognitionEventLike) => void) | null
  onerror: ((e: { error?: string }) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

export function CommunicationHubView() {
  const prefersReduced = useReducedMotion()
  const { preferences, modelHealth, pipelineStatus } = useSignBridgeStore()

  const {
    videoRef,
    canvasRef,
    isStreaming,
    cameraError,
    isHandDetected,
    recognizedText,
    glosses,
    activePrediction,
    isSpeaking,
    retryCamera,
    clearOutput,
    speakOutput
  } = useAslRecognition("Continuous")

  const [entries, setEntries] = React.useState<ConversationEntry[]>([])
  const [hearingDraft, setHearingDraft] = React.useState("")
  const [activeHearingPrompt, setActiveHearingPrompt] = React.useState("")
  const [isMicListening, setIsMicListening] = React.useState(false)
  const [speechSupported, setSpeechSupported] = React.useState(false)
  const [vocabSet, setVocabSet] = React.useState<Set<string>>(() => new Set())
  const [copiedId, setCopiedId] = React.useState<string | null>(null)
  const recognitionRef = React.useRef<BrowserSpeechRecognition | null>(null)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      if (typeof window !== "undefined") {
        const w = window as unknown as {
          SpeechRecognition?: new () => BrowserSpeechRecognition
          webkitSpeechRecognition?: new () => BrowserSpeechRecognition
        }
        setSpeechSupported(Boolean(w.SpeechRecognition || w.webkitSpeechRecognition))
      }

      fetch("/api/vocabulary?limit=2731")
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (Array.isArray(data?.items)) {
            setVocabSet(
              new Set(
                data.items
                  .map((item: { gloss?: string }) => String(item.gloss || "").toUpperCase())
                  .filter(Boolean)
              )
            )
          }
        })
        .catch(() => {})
    }, 0)

    return () => {
      window.clearTimeout(timer)
      try {
        recognitionRef.current?.stop()
      } catch {
        // Ignore cleanup stop errors
      }
      if (audioRef.current) {
        audioRef.current.pause()
      }
    }
  }, [])

  const pushSignerTurn = () => {
    if (!recognizedText.trim()) {
      toast.info("Sign a phrase first to send it to the conversation")
      return
    }
    const entry: ConversationEntry = {
      id: `signer-${Date.now()}`,
      speaker: "signer",
      text: recognizedText.trim(),
      glosses: [...glosses],
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    }
    setEntries((prev) => [...prev, entry])
    toast.success("Added signed translation to conversation")
  }

  const pushHearingTurn = (rawText: string) => {
    const clean = rawText.trim()
    if (!clean) return
    setActiveHearingPrompt(clean)
    setHearingDraft("")
    const entry: ConversationEntry = {
      id: `hearing-${Date.now()}`,
      speaker: "hearing",
      text: clean,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    }
    setEntries((prev) => [...prev, entry])
  }

  const toggleMicrophone = () => {
    if (isMicListening) {
      recognitionRef.current?.stop()
      setIsMicListening(false)
      return
    }

    const w = window as unknown as {
      SpeechRecognition?: new () => BrowserSpeechRecognition
      webkitSpeechRecognition?: new () => BrowserSpeechRecognition
    }
    const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition
    if (!Ctor) {
      toast.info("Browser speech-to-text is unavailable here. Type your message instead.")
      return
    }

    try {
      const rec = new Ctor()
      rec.continuous = false
      rec.interimResults = true
      rec.lang = "en-US"
      rec.onresult = (e: SpeechRecognitionEventLike) => {
        const latest = e.results[e.results.length - 1]
        if (!latest) return
        const text = String(latest[0]?.transcript || "").trim()
        setHearingDraft(text)
        if (latest.isFinal && text) {
          pushHearingTurn(text)
        }
      }
      rec.onerror = () => {
        setIsMicListening(false)
        toast.error("Microphone speech capture encountered an error")
      }
      rec.onend = () => {
        setIsMicListening(false)
      }
      recognitionRef.current = rec
      rec.start()
      setIsMicListening(true)
    } catch {
      setIsMicListening(false)
      toast.error("Unable to start browser speech recognition")
    }
  }

  const speakTextWithKokoro = async (text: string) => {
    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
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
          await audio.play()
          return
        }
      }
    } catch {
      // Fallback below
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))
    }
  }

  const parsedHearingTokens = React.useMemo(() => {
    const source = activeHearingPrompt || hearingDraft
    if (!source.trim()) return []
    return source
      .replace(/[^A-Za-z0-9\s'-]/g, "")
      .split(/\s+/)
      .filter(Boolean)
      .map((word) => {
        const upper = word.toUpperCase()
        return {
          word,
          upper,
          inVocab: vocabSet.has(upper)
        }
      })
  }, [activeHearingPrompt, hearingDraft, vocabSet])

  const nlpLabel = pipelineStatus.groq?.available
    ? "Groq Qwen3 Smoothing"
    : pipelineStatus.qwen?.available
      ? "Local Qwen Smoothing"
      : "Rule Smoothing"

  return (
    <div className="w-full flex flex-col gap-5 pb-12">
      {/* Studio Banner */}
      <section className="sb-glass-card rounded-[26px] px-5 sm:px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl sb-orange-pill text-white flex items-center justify-center shrink-0">
            <ArrowDownUp className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#FAF6F0]">
                Two-Way Translation Hub
              </h1>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/12 border border-emerald-400/30 text-[11px] font-bold text-emerald-300">
                <Languages className="w-3.5 h-3.5" />
                {nlpLabel}
              </span>
            </div>
            <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Real-time ASL signing to spoken English on the left, live high-contrast text & gloss cues on the right.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          <div className="px-3.5 py-2 rounded-full bg-[#1D1829] border border-white/10 text-xs font-mono font-bold text-[#FAF6F0]">
            {modelHealth.lastLatencyMs !== null ? `${Math.round(modelHealth.lastLatencyMs)} ms` : "Ready"}
          </div>
        </div>
      </section>

      {/* Dual Stage Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
        {/* LEFT POD: Real ASL Signer Stage */}
        <section className="sb-surface-card rounded-[26px] p-5 flex flex-col justify-between gap-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <span className="px-2.5 py-1 rounded-xl bg-[#251F35] border border-amber-400/30 text-amber-300 text-xs font-extrabold">
                ASL
              </span>
              <div>
                <h2 className="text-base font-extrabold text-[#FAF6F0]">Deaf Signer Stage</h2>
                <p className="text-[11px] text-[#ACA5BC]">
                  Live camera → MediaPipe → EXP13 → English & Kokoro speech
                </p>
              </div>
            </div>

            {isHandDetected && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/12 text-emerald-300 text-[11px] font-bold">
                <Hand className="w-3.5 h-3.5" /> Tracked
              </span>
            )}
          </div>

          {/* Real Mirrored Webcam + Skeleton */}
          <div className="relative aspect-video rounded-[20px] bg-[#120F1B] border border-white/10 overflow-hidden flex items-center justify-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover scale-x-[-1] ${
                isStreaming && !cameraError ? "opacity-100" : "opacity-0"
              }`}
            />
            <canvas
              ref={canvasRef}
              className={`pointer-events-none absolute inset-0 w-full h-full object-cover scale-x-[-1] ${
                isStreaming && !cameraError ? "opacity-100" : "opacity-0"
              }`}
            />

            {(!isStreaming || cameraError) && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center text-white bg-[#120F1B]/90">
                <SignBridgeMascot size="sm" state="listening" />
                <p className="text-xs font-bold mt-2">
                  {cameraError ? cameraError.message : "Starting webcam stream..."}
                </p>
                {cameraError && (
                  <button
                    type="button"
                    onClick={retryCamera}
                    className="mt-3 sb-orange-pill px-4 py-1.5 rounded-full text-xs font-bold text-white cursor-pointer"
                  >
                    Retry Camera
                  </button>
                )}
              </div>
            )}

            {/* Live Prediction Pill */}
            <div className="absolute top-3 left-3 z-10 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/65 backdrop-blur-md text-white text-xs font-semibold border border-white/15">
              <span className="w-2 h-2 rounded-full bg-[#E5560B]" />
              <span>
                {activePrediction
                  ? `${activePrediction.label} (${activePrediction.confidence.toFixed(1)}%)`
                  : "Waiting for ASL sign..."}
              </span>
            </div>
          </div>

          {/* Live English Output + Actions */}
          <div className="rounded-2xl bg-[#161221] border border-white/10 p-4 flex flex-col gap-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#ACA5BC]">
                Recognized Output
              </span>
              <div className="flex items-center gap-1.5">
                {recognizedText && (
                  <button
                    type="button"
                    onClick={clearOutput}
                    className="p-1.5 rounded-full hover:bg-[#1D1829] text-[#ACA5BC] transition cursor-pointer"
                    title="Clear recognized output"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={speakOutput}
                  disabled={!recognizedText || isSpeaking}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#1D1829] border border-white/10 hover:border-amber-400 text-xs font-bold text-[#FAF6F0] disabled:opacity-40 transition cursor-pointer"
                >
                  <Volume2 className="w-3.5 h-3.5 text-amber-400" />
                  <span>{isSpeaking ? "Speaking..." : "Speak"}</span>
                </button>
              </div>
            </div>

            <p className="text-base sm:text-lg font-extrabold text-[#FAF6F0] min-h-[28px]">
              {recognizedText || (
                <span className="text-sm font-medium text-[#7E7890] italic">
                  Perform ASL signs in view to translate into natural English...
                </span>
              )}
            </p>

            <div className="flex items-center justify-between gap-2 pt-1">
              <span className="text-[11px] font-mono text-[#ACA5BC]">
                {glosses.length} gloss{glosses.length === 1 ? "" : "es"} buffered
              </span>
              <button
                type="button"
                onClick={pushSignerTurn}
                disabled={!recognizedText}
                className="sb-orange-pill inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-bold text-white disabled:opacity-40 cursor-pointer"
              >
                <Send className="w-3.5 h-3.5" />
                <span>Log Signer Turn</span>
              </button>
            </div>
          </div>
        </section>

        {/* RIGHT POD: Hearing Speaker -> Deaf Readable Display & Vocabulary Cues */}
        <section className="sb-surface-card rounded-[26px] p-5 flex flex-col justify-between gap-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <span className="px-2.5 py-1 rounded-xl bg-emerald-500/12 border border-emerald-400/30 text-emerald-300 text-xs font-extrabold">
                VOICE
              </span>
              <div>
                <h2 className="text-base font-extrabold text-[#FAF6F0]">Hearing Speaker Stage</h2>
                <p className="text-[11px] text-[#ACA5BC]">
                  Type or use browser dictation for high-contrast Deaf participant display
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={toggleMicrophone}
              disabled={!speechSupported}
              title={
                speechSupported
                  ? "Use browser Speech Recognition"
                  : "Web Speech API not supported in this browser"
              }
              className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-bold transition cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed ${
                isMicListening
                  ? "bg-[#EF4444] text-white animate-pulse"
                  : "bg-[#251F35] border border-amber-400/30 text-amber-300 hover:bg-amber-400/18"
              }`}
            >
              {isMicListening ? <Mic className="w-3.5 h-3.5" /> : <MicOff className="w-3.5 h-3.5" />}
              <span>
                {isMicListening ? "Listening..." : speechSupported ? "Dictate" : "Dictation N/A"}
              </span>
            </button>
          </div>

          {/* High-Contrast Teleprompter Card */}
          <div className="rounded-[20px] bg-[#2C2440] hover:bg-[#382E50] text-white p-5 flex flex-col justify-between min-h-[210px] border border-[#3B2F28]">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[10px] font-mono uppercase tracking-widest text-amber-300">
                Deaf Participant Teleprompter
              </span>
              <span
                className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-white/10 text-[10px] font-semibold text-white/80"
                title="Planned future feature per roadmap"
              >
                <Clock className="w-3 h-3 text-[#FFB347]" />
                <span>3D Avatar: Planned</span>
              </span>
            </div>

            <div className="my-4">
              {activeHearingPrompt || hearingDraft ? (
                <p className="text-xl sm:text-2xl font-extrabold leading-snug text-white">
                  “{hearingDraft || activeHearingPrompt}”
                </p>
              ) : (
                <div className="flex items-center gap-3 text-white/75">
                  <Camera className="w-5 h-5 text-amber-300 shrink-0" />
                  <p className="text-sm leading-relaxed">
                    Type a reply below, tap a quick phrase, or dictate aloud to show large readable text and ASL vocabulary matches here.
                  </p>
                </div>
              )}
            </div>

            {/* ASL Vocabulary Gloss Highlights */}
            <div className="flex flex-wrap items-center gap-1.5 pt-3 border-t border-white/10">
              {parsedHearingTokens.length > 0 ? (
                parsedHearingTokens.map((tok, i) => (
                  <span
                    key={`${tok.upper}-${i}`}
                    className={`px-2.5 py-1 rounded-lg text-[11px] font-mono font-bold ${
                      tok.inVocab
                        ? "bg-[#E5560B]/25 text-[#FFB347] border border-amber-400/45"
                        : "bg-white/10 text-white/70"
                    }`}
                    title={
                      tok.inVocab
                        ? "Present in SignBridge 2,731-class ASL vocabulary"
                        : "Word not matched directly in ASL gloss list"
                    }
                  >
                    {tok.upper}
                  </span>
                ))
              ) : (
                <span className="text-[11px] text-white/50">
                  ASL gloss token cues appear here automatically
                </span>
              )}
            </div>
          </div>

          {/* Typed Reply Input + Quick Chips */}
          <div className="space-y-3">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                pushHearingTurn(hearingDraft)
              }}
              className="flex items-center gap-2"
            >
              <input
                value={hearingDraft}
                onChange={(e) => setHearingDraft(e.target.value)}
                placeholder="Type message for Deaf participant..."
                className="flex-1 rounded-full bg-[#161221] border border-white/10 px-4 py-2.5 text-sm font-medium text-[#FAF6F0] placeholder:text-[#7E7890] focus:outline-none focus:ring-2 focus:ring-amber-400"
              />
              <button
                type="submit"
                disabled={!hearingDraft.trim()}
                className="sb-orange-pill px-4 py-2.5 rounded-full text-xs font-bold text-white disabled:opacity-40 cursor-pointer shrink-0"
              >
                Send
              </button>
            </form>

            <div className="flex flex-wrap items-center gap-1.5">
              {QUICK_CHIPS.map((chip) => (
                <motion.button
                  key={chip}
                  type="button"
                  whileTap={prefersReduced ? undefined : { scale: 0.97 }}
                  onClick={() => pushHearingTurn(chip)}
                  className="px-3 py-1.5 rounded-full bg-[#161221] hover:bg-amber-400/18 border border-white/10 text-xs font-semibold text-[#FAF6F0] transition cursor-pointer"
                >
                  {chip}
                </motion.button>
              ))}
            </div>
          </div>
        </section>
      </div>

      {/* Real Chronological Conversation Transcript */}
      <section className="sb-surface-card rounded-[26px] p-5">
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-extrabold text-[#FAF6F0]">Conversation Log</h3>
          </div>
          {entries.length > 0 && (
            <button
              type="button"
              onClick={() => setEntries([])}
              className="text-xs font-bold text-[#ACA5BC] hover:text-[#FAF6F0] cursor-pointer"
            >
              Clear Log
            </button>
          )}
        </div>

        {entries.length === 0 ? (
          <p className="text-xs text-[#ACA5BC] py-3">
            No turns logged yet. Sign a sentence or send a message above to start a bilingual transcript.
          </p>
        ) : (
          <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
            {entries.map((item) => (
              <div
                key={item.id}
                className={`p-3.5 rounded-2xl border flex items-start justify-between gap-3 ${
                  item.speaker === "signer"
                    ? "bg-[#161221] border-amber-400/30"
                    : "bg-[#1D1829] border-white/10"
                }`}
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-[11px] font-bold">
                    <span
                      className={
                        item.speaker === "signer" ? "text-amber-300" : "text-emerald-300"
                      }
                    >
                      {item.speaker === "signer" ? "ASL Signer" : "Hearing Speaker"}
                    </span>
                    <span className="text-[#7E7890]">· {item.timestamp}</span>
                  </div>
                  <p className="text-sm font-semibold text-[#FAF6F0] mt-0.5">{item.text}</p>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={() => void speakTextWithKokoro(item.text)}
                    aria-label="Speak transcript line"
                    className="p-2 rounded-full hover:bg-[#251F35] text-[#ACA5BC] hover:text-amber-400 cursor-pointer"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={async () => {
                      await navigator.clipboard.writeText(item.text)
                      setCopiedId(item.id)
                      window.setTimeout(() => setCopiedId(null), 1500)
                    }}
                    aria-label="Copy transcript line"
                    className="p-2 rounded-full hover:bg-[#251F35] text-[#ACA5BC] cursor-pointer"
                  >
                    {copiedId === item.id ? (
                      <Check className="w-3.5 h-3.5 text-[#1FA968]" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
