"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import {
  Sliders,
  Cpu,
  Volume2,
  ShieldCheck,
  Server,
  Save,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  UserPlus
} from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select"
import { toast } from "sonner"
import {
  KokoroVoiceId,
  useSignBridgeStore
} from "@/lib/signbridge-store"

const KOKORO_VOICES: Array<{ id: KokoroVoiceId; label: string }> = [
  { id: "af_sarah", label: "af_sarah (Balanced Studio Default)" },
  { id: "af_bella", label: "af_bella (Warm Expressive)" },
  { id: "af_nicole", label: "af_nicole (Clear Narration)" },
  { id: "am_adam", label: "am_adam (Deep Male Voice)" }
]

export function SettingsView() {
  const prefersReduced = useReducedMotion()
  const {
    preferences,
    updatePreferences,
    resetPreferences,
    startFreshProfile,
    modelHealth,
    pipelineStatus,
    customSigns,
    personalizationEnabled,
    setPersonalizationEnabled,
    refreshHealth,
    refreshPipelineStatus,
    refreshPersonalization
  } = useSignBridgeStore()

  const [isTestingVoice, setIsTestingVoice] = React.useState(false)
  const [isResettingProfile, setIsResettingProfile] = React.useState(false)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)

  React.useEffect(() => {
    return () => {
      if (audioRef.current) audioRef.current.pause()
    }
  }, [])

  const handleTogglePersonalization = async (checked: boolean) => {
    await setPersonalizationEnabled(checked)
    toast.success(
      checked ? "Custom sign routing enabled" : "Custom sign routing disabled"
    )
  }

  const handleTestVoice = async () => {
    if (isTestingVoice) return
    setIsTestingVoice(true)
    const sampleText = "SignBridge Kokoro neural speech output is active."
    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: sampleText,
          voice: preferences.ttsVoice
        })
      })
      if (res.ok) {
        const data = await res.json()
        const src = data.audio_url || data.audio_data
        if (src) {
          audioRef.current?.pause()
          const audio = new Audio(src)
          audioRef.current = audio
          audio.onended = () => setIsTestingVoice(false)
          audio.onerror = () => setIsTestingVoice(false)
          await audio.play()
          toast.success(`Playing voice: ${preferences.ttsVoice}`)
          return
        }
      }
      throw new Error("Audio missing")
    } catch {
      if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(sampleText)
        u.onend = () => setIsTestingVoice(false)
        u.onerror = () => setIsTestingVoice(false)
        window.speechSynthesis.speak(u)
        toast.info("Using browser speech fallback")
      } else {
        setIsTestingVoice(false)
        toast.error("Audio playback unavailable")
      }
    }
  }

  const isOnline = modelHealth.status === "online"

  return (
    <div className="w-full max-w-[1240px] mx-auto flex flex-col gap-5 pb-12">
      {/* Header Banner */}
      <section className="sb-glass-card rounded-[26px] px-5 sm:px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl sb-orange-pill text-white flex items-center justify-center shrink-0">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#FAF6F0]">
                System & AI Engine Settings
              </h1>
              <span className="px-3 py-0.5 rounded-full bg-[#251F35] border border-amber-400/30 font-mono text-[11px] font-bold text-amber-300">
                {modelHealth.hardwareDevice ?? modelHealth.device ?? "Checking Device"}
              </span>
            </div>
            <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Persistent recognition thresholds, Kokoro ONNX voice selection, and live FastAPI health diagnostics.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0 self-start sm:self-auto">
          <button
            type="button"
            onClick={() => {
              resetPreferences()
              toast.info("Preferences restored to defaults")
            }}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-[#1D1829] border border-white/10 hover:bg-[#251F35] text-xs font-bold text-[#ACA5BC] cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Defaults</span>
          </button>

          <motion.button
            type="button"
            whileTap={prefersReduced ? undefined : { scale: 0.97 }}
            onClick={() => {
              void refreshHealth()
              void refreshPipelineStatus()
              void refreshPersonalization()
              toast.success("All settings and profile state synced")
            }}
            className="sb-orange-pill inline-flex items-center gap-2 px-5 py-2 rounded-full text-xs font-bold text-white cursor-pointer"
          >
            <Save className="w-3.5 h-3.5" />
            <span>Sync & Save</span>
          </motion.button>
        </div>
      </section>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Card 1: Recognition Model & Thresholds */}
        <section className="sb-surface-card rounded-[26px] p-6 space-y-5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#251F35] border border-amber-400/30 text-amber-400 flex items-center justify-center">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-[#FAF6F0]">Recognition Engine</h2>
              <p className="text-xs text-[#ACA5BC]">ASL Citizen checkpoint & stabilization settings</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-bold text-[#FAF6F0]">Default Model Scope</label>
              <Select
                value={preferences.modelVariant}
                onValueChange={(val) => {
                  if (val === "production_exp13" || val === "demo_50") {
                    updatePreferences({ modelVariant: val })
                  }
                }}
              >
                <SelectTrigger className="w-full rounded-2xl bg-[#161221] border-white/10 text-xs font-semibold">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="rounded-2xl">
                  <SelectItem value="production_exp13">
                    Production Ensemble V2 (2,731 ASL Classes · 71.89% Top-1)
                  </SelectItem>
                  <SelectItem value="demo_50">
                    Demo 50-Sign Subset (Quick Pitch Verification)
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#FAF6F0]">Confidence Cutoff Threshold</span>
                <span className="font-mono font-bold text-amber-400 tabular-nums">
                  {preferences.confidenceCutoff}%
                </span>
              </div>
              <Slider
                value={[preferences.confidenceCutoff]}
                onValueChange={(vals) => {
                  const arr = Array.isArray(vals) ? vals : [vals]
                  updatePreferences({ confidenceCutoff: arr[0] ?? 45 })
                }}
                min={10}
                max={90}
                step={5}
              />
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-white/10">
              <div>
                <span className="text-xs font-bold text-[#FAF6F0] block">
                  Personalization Routing ({customSigns.length} signs)
                </span>
                <span className="text-[11px] text-[#ACA5BC]">
                  Rerank live predictions using your recorded custom sign prototypes
                </span>
              </div>
              <Switch
                checked={Boolean(personalizationEnabled || preferences.personalizationEnabled)}
                onCheckedChange={(checked) => void handleTogglePersonalization(checked)}
              />
            </div>
          </div>
        </section>

        {/* Card 2: Kokoro ONNX Neural TTS */}
        <section className="sb-surface-card rounded-[26px] p-6 space-y-5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-emerald-500/12 border border-emerald-400/30 text-emerald-300 flex items-center justify-center">
              <Volume2 className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-[#FAF6F0]">Kokoro ONNX Neural TTS</h2>
              <p className="text-xs text-[#ACA5BC]">Voice synthesis for spoken English translations</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-[#FAF6F0]">Active Kokoro Voice</label>
                <button
                  type="button"
                  onClick={() => void handleTestVoice()}
                  disabled={isTestingVoice}
                  className="text-xs font-bold text-amber-400 hover:text-amber-300 cursor-pointer disabled:opacity-50"
                >
                  {isTestingVoice ? "Playing..." : "Test Voice"}
                </button>
              </div>

              <Select
                value={preferences.ttsVoice}
                onValueChange={(val) => {
                  if (
                    val === "af_sarah" ||
                    val === "af_bella" ||
                    val === "af_nicole" ||
                    val === "am_adam"
                  ) {
                    updatePreferences({ ttsVoice: val })
                  }
                }}
              >
                <SelectTrigger className="w-full rounded-2xl bg-[#161221] border-white/10 text-xs font-semibold">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="rounded-2xl">
                  {KOKORO_VOICES.map((v) => (
                    <SelectItem key={v.id} value={v.id}>
                      {v.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-white/10">
              <div>
                <span className="text-xs font-bold text-[#FAF6F0] block">
                  Show Hand & Latency HUD
                </span>
                <span className="text-[11px] text-[#ACA5BC]">
                  Display MediaPipe tracking pill & measured inference speed overlay
                </span>
              </div>
              <Switch
                checked={preferences.telemetryHud}
                onCheckedChange={(checked) => updatePreferences({ telemetryHud: checked })}
              />
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-white/10">
              <div>
                <span className="text-xs font-bold text-[#FAF6F0] block">
                  Auto-Speak Completed Sentences
                </span>
                <span className="text-[11px] text-[#ACA5BC]">
                  Automatically voice smoothed English phrases via Kokoro TTS
                </span>
              </div>
              <Switch
                checked={preferences.autoSpeakTTS}
                onCheckedChange={(checked) => updatePreferences({ autoSpeakTTS: checked })}
              />
            </div>
          </div>
        </section>

        {/* Card 3: Privacy & Landmark Architecture */}
        <section className="sb-surface-card rounded-[26px] p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#EEF2FF] border border-[#C7D2FE] text-[#4F46E5] flex items-center justify-center">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-[#FAF6F0]">Privacy & Landmark Pipeline</h2>
              <p className="text-xs text-[#ACA5BC]">How camera streams are processed</p>
            </div>
          </div>

          <div className="space-y-3 pt-1">
            <div className="space-y-1.5">
              <label className="text-xs font-bold text-[#FAF6F0]">Profile Display Name</label>
              <input
                type="text"
                value={preferences.displayName}
                maxLength={40}
                onChange={(e) => updatePreferences({ displayName: e.target.value })}
                placeholder="Learner"
                className="w-full px-3.5 py-2 rounded-2xl bg-[#161221] border border-white/10 text-xs font-semibold text-[#FAF6F0] focus:outline-none focus:border-amber-400/50"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-2xl bg-[#161221] border border-white/10 text-xs">
              <div>
                <span className="font-bold text-[#FAF6F0] block">Isolated Custom Model Profile</span>
                <span className="font-mono text-[11px] text-[#ACA5BC]">
                  {preferences.userId || "Initializing..."} · {customSigns.length} custom signs
                </span>
              </div>
              <button
                type="button"
                disabled={isResettingProfile}
                onClick={async () => {
                  setIsResettingProfile(true)
                  try {
                    const eraseRemote = Boolean(
                      preferences.userId && preferences.userId !== "default"
                    )
                    await startFreshProfile(eraseRemote)
                    toast.success("Started a fresh isolated workspace with clean custom model")
                  } finally {
                    setIsResettingProfile(false)
                  }
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-500/15 hover:bg-amber-500/25 border border-amber-400/35 text-xs font-bold text-amber-200 transition cursor-pointer disabled:opacity-50"
              >
                <UserPlus className="w-3.5 h-3.5" />
                <span>{isResettingProfile ? "Resetting..." : "Start Fresh User"}</span>
              </button>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-emerald-500/12 border border-emerald-400/30 text-xs text-emerald-300 space-y-1.5">
            <div className="font-extrabold flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>On-Device Landmark Extraction Only</span>
            </div>
            <p className="leading-relaxed">
              Live mode extracts 42 3D hand coordinates per frame in local browser WASM and sends only normalized coordinate tensors <code className="font-mono">(32, 42, 3)</code> to your FastAPI backend — never raw webcam video frames.
            </p>
          </div>
        </section>

        {/* Card 4: Live Backend Diagnostics (Strictly Verified API State) */}
        <section className="sb-surface-card rounded-[26px] p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#251F35] border border-amber-400/30 text-amber-400 flex items-center justify-center">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-[#FAF6F0]">Live Service Diagnostics</h2>
              <p className="text-xs text-[#ACA5BC]">Real telemetry from `/api/health` & `/api/pipeline/status`</p>
            </div>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between p-3 rounded-2xl bg-[#161221] border border-white/10">
              <span className="font-bold text-[#FAF6F0]">FastAPI Recognition Server</span>
              <span
                className={`font-mono font-bold inline-flex items-center gap-1.5 ${
                  isOnline ? "text-emerald-300" : "text-[#EF4444]"
                }`}
              >
                {isOnline ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-[#1FA968] animate-pulse" />
                    Online ({modelHealth.device ?? "cpu"} · {modelHealth.numClasses ?? 0} classes)
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-3.5 h-3.5" />
                    Offline
                  </>
                )}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-2xl bg-[#161221] border border-white/10">
              <span className="font-bold text-[#FAF6F0]">NLP Gloss Smoothing</span>
              <span className="font-mono font-bold text-[#FAF6F0] truncate max-w-[200px]">
                {pipelineStatus.groq?.available
                  ? `Groq (${pipelineStatus.groq.model})`
                  : pipelineStatus.qwen?.available
                    ? `Local (${pipelineStatus.qwen.model})`
                    : "Unavailable / Rule Fallback"}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-2xl bg-[#161221] border border-white/10">
              <span className="font-bold text-[#FAF6F0]">Kokoro Neural TTS</span>
              <span className="font-mono font-bold text-emerald-300">
                {pipelineStatus.tts?.available
                  ? `${pipelineStatus.tts.status} (${pipelineStatus.tts.voice})`
                  : "Unavailable"}
              </span>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
