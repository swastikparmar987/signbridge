"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import {
  ArrowLeft,
  BarChart3,
  Clock,
  Copy,
  Cpu,
  History,
  RefreshCw,
  Sparkles,
  Trash2,
  TrendingUp,
  Volume2
} from "lucide-react"
import { toast } from "sonner"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"
import {
  RecognitionMode,
  formatSessionTimestamp,
  useSignBridgeStore
} from "@/lib/signbridge-store"

interface ProgressViewProps {
  onBackToHome: () => void
}

type ModeFilter = "All" | RecognitionMode

export function ProgressView({ onBackToHome }: ProgressViewProps) {
  const prefersReduced = useReducedMotion()
  const {
    sessions,
    modelHealth,
    pipelineStatus,
    customSigns,
    personalizationEnabled,
    preferences,
    refreshHealth,
    refreshPipelineStatus,
    clearSessions
  } = useSignBridgeStore()

  const [modeFilter, setModeFilter] = React.useState<ModeFilter>("All")
  const [refreshing, setRefreshing] = React.useState(false)

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await Promise.all([refreshHealth(), refreshPipelineStatus()])
      toast.success("Telemetry refreshed")
    } finally {
      setRefreshing(false)
    }
  }

  const speakText = async (text: string) => {
    if (!text) return
    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice: preferences.ttsVoice })
      })
      if (res.ok) {
        const data = await res.json()
        const src = data.audio_url || data.audio_data
        if (src) {
          const audio = new Audio(src)
          await audio.play()
          return
        }
      }
    } catch {
      // Fallback below
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))
    }
  }

  const copyText = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success("Copied transcript")
    } catch {
      toast.error("Clipboard unavailable")
    }
  }

  const top1Primary = modelHealth.productionTop1 ?? modelHealth.exp13Top1
  const top5Primary = modelHealth.productionTop5 ?? modelHealth.exp13Top5
  const readyPrototypes = customSigns.filter((s) => s.has_prototype).length

  const modeCounts = React.useMemo(() => {
    const counts: Record<RecognitionMode, number> = {
      Continuous: 0,
      Word: 0,
      Fingerspelling: 0
    }
    for (const s of sessions) {
      counts[s.mode] = (counts[s.mode] ?? 0) + 1
    }
    return counts
  }, [sessions])

  const filteredSessions = React.useMemo(() => {
    if (modeFilter === "All") return sessions
    return sessions.filter((s) => s.mode === modeFilter)
  }, [sessions, modeFilter])

  const summaryCards = [
    {
      label: "Production Top-1",
      value: top1Primary !== null ? `${top1Primary.toFixed(2)}%` : "—",
      sub: modelHealth.activeExperiment
        ? `${modelHealth.activeExperiment} · 2,731 glosses`
        : "Held-out ASL Citizen test split",
      icon: Sparkles
    },
    {
      label: "Production Top-5",
      value: top5Primary !== null ? `${top5Primary.toFixed(2)}%` : "—",
      sub: "Top-5 gloss recall across full vocabulary",
      icon: TrendingUp
    },
    {
      label: "Inference Latency",
      value: modelHealth.lastLatencyMs !== null ? `${modelHealth.lastLatencyMs} ms` : "Ready",
      sub: modelHealth.hardwareDevice
        ? `${modelHealth.hardwareDevice} (${modelHealth.device || "local"})`
        : modelHealth.device || "Local compute target",
      icon: Cpu
    },
    {
      label: "Recorded Sessions",
      value: `${sessions.length}`,
      sub:
        customSigns.length > 0
          ? `${readyPrototypes}/${customSigns.length} custom sign prototypes ready`
          : sessions.length > 0
            ? "Saved from live studio runs"
            : "Sign live to populate telemetry",
      icon: History
    }
  ]

  const benchmarks = [
    ...(modelHealth.productionTop1 !== null || modelHealth.productionTop5 !== null
      ? [
          {
            label: `Production (${modelHealth.activeExperiment || "exp14c_combined"}) Top-1`,
            value: modelHealth.productionTop1,
            barClass: "bg-gradient-to-r from-[#FF9436] to-[#E85D04]",
            textClass: "text-amber-300"
          },
          {
            label: `Production (${modelHealth.activeExperiment || "exp14c_combined"}) Top-5`,
            value: modelHealth.productionTop5,
            barClass: "bg-gradient-to-r from-[#FFA959] to-[#FF7900]",
            textClass: "text-amber-400"
          }
        ]
      : []),
    {
      label: "EXP13 Held-Out Top-1 (ArcFace + SkeletonWarp)",
      value: modelHealth.exp13Top1,
      barClass: "bg-gradient-to-r from-[#FF7900] to-[#C2410C]",
      textClass: "text-amber-300"
    },
    {
      label: "EXP13 Held-Out Top-5 Recall",
      value: modelHealth.exp13Top5,
      barClass: "bg-gradient-to-r from-[#F59E0B] to-[#D97706]",
      textClass: "text-[#B45309]"
    }
  ]

  return (
    <div className="w-full flex flex-col gap-5">
      {/* Studio Header */}
      <section className="sb-glass-card rounded-[26px] px-5 sm:px-6 py-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5 min-w-0">
          <SignBridgeMascot
            variant="wave"
            size="sm"
            state={modelHealth.status === "online" ? "idle" : "helper"}
          />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#FAF6F0]">
                Analytics & Session Telemetry
              </h1>
              <span className="px-3 py-0.5 rounded-full bg-[#251F35] border border-amber-400/30 text-[11px] font-bold text-amber-300">
                {modelHealth.numClasses !== null
                  ? `${modelHealth.numClasses.toLocaleString()} Glosses`
                  : "ASL Telemetry"}
              </span>
            </div>
            <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Verified held-out ASL Citizen accuracy benchmarks, pipeline status, and recorded recognition runs.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 self-start lg:self-auto shrink-0">
          <button
            type="button"
            onClick={() => void handleRefresh()}
            disabled={refreshing}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-[#1D1829] border border-white/10 text-xs font-bold text-[#FAF6F0] hover:bg-[#251F35] transition cursor-pointer disabled:opacity-60"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-amber-400 ${refreshing ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
          <button
            type="button"
            onClick={onBackToHome}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-[#1D1829] border border-white/10 text-xs font-bold text-[#FAF6F0] hover:bg-[#251F35] transition cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5 text-amber-400" />
            <span>Back to Home</span>
          </button>
        </div>
      </section>

      {/* Top Real Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryCards.map((card) => {
          const Icon = card.icon
          return (
            <motion.div
              key={card.label}
              initial={prefersReduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: prefersReduced ? 0 : 0.2 }}
              className="sb-surface-card rounded-[22px] p-4 sm:p-5 flex flex-col gap-2"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-[#ACA5BC]">
                  {card.label}
                </span>
                <div className="w-8 h-8 rounded-xl bg-[#251F35] text-amber-400 flex items-center justify-center shrink-0">
                  <Icon className="w-4 h-4" />
                </div>
              </div>
              <div className="text-2xl sm:text-[26px] font-extrabold tracking-tight text-[#FAF6F0] tabular-nums">
                {card.value}
              </div>
              <div className="text-[11px] font-medium text-[#ACA5BC] leading-snug">
                {card.sub}
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Held-Out Benchmarks & Session Log */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Benchmarks & Pipeline Breakdown */}
        <div className="lg:col-span-5 sb-surface-card rounded-[24px] p-5 space-y-5">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-[#251F35] text-amber-400 flex items-center justify-center">
                <BarChart3 className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-base font-extrabold text-[#FAF6F0]">
                  Verified Held-Out Evaluation
                </h2>
                <p className="text-[11px] font-medium text-[#ACA5BC]">
                  Reported directly by backend `/api/health`
                </p>
              </div>
            </div>
            <span
              className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                modelHealth.status === "online"
                  ? "bg-emerald-500/15 text-emerald-300 border border-emerald-400/30"
                  : "bg-amber-400/12 text-amber-300 border border-amber-400/30"
              }`}
            >
              {modelHealth.status === "online" ? "Online" : "Checking"}
            </span>
          </div>

          <div className="space-y-3">
            {benchmarks.map((item) => {
              const pct = item.value !== null ? Math.min(100, Math.max(0, item.value)) : 0
              return (
                <div
                  key={item.label}
                  className="p-3.5 rounded-2xl bg-[#161221] border border-white/10 space-y-2"
                >
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="font-semibold text-[#B6AFC6] truncate">
                      {item.label}
                    </span>
                    <span className={`font-extrabold tabular-nums shrink-0 ${item.textClass}`}>
                      {item.value !== null ? `${item.value.toFixed(2)}%` : "Backend offline"}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                    <motion.div
                      initial={prefersReduced ? false : { width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: prefersReduced ? 0 : 0.45 }}
                      className={`h-full rounded-full ${item.barClass}`}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Pipeline Service Health */}
          <div className="pt-1 space-y-2.5">
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-[#ACA5BC]">
              Active Pipeline Services
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
              <div className="p-3 rounded-xl bg-[#161221] border border-white/10">
                <div className="text-[11px] font-semibold text-[#ACA5BC]">
                  Sentence Polishing
                </div>
                <div className="font-bold text-[#FAF6F0] mt-0.5 truncate">
                  {pipelineStatus.groq?.available
                    ? pipelineStatus.groq.model
                    : pipelineStatus.qwen?.available
                      ? pipelineStatus.qwen.model
                      : "Rule-based fallback"}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-[#161221] border border-white/10">
                <div className="text-[11px] font-semibold text-[#ACA5BC]">
                  Speech Output
                </div>
                <div className="font-bold text-[#FAF6F0] mt-0.5 truncate">
                  {pipelineStatus.tts?.available
                    ? `Kokoro · ${preferences.ttsVoice}`
                    : `Browser fallback · ${preferences.ttsVoice}`}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-[#161221] border border-white/10">
                <div className="text-[11px] font-semibold text-[#ACA5BC]">
                  Fingerspelling MLP
                </div>
                <div className="font-bold text-[#FAF6F0] mt-0.5">
                  {modelHealth.fingerspellingReady ? "63→128→26 Ready" : "Not loaded"}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-[#161221] border border-white/10">
                <div className="text-[11px] font-semibold text-[#ACA5BC]">
                  Custom Routing
                </div>
                <div className="font-bold text-[#FAF6F0] mt-0.5">
                  {personalizationEnabled ? `Enabled (${readyPrototypes} ready)` : "Standby"}
                </div>
              </div>
            </div>
          </div>

          {/* Recorded Session Breakdown */}
          <div className="pt-1 space-y-2">
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-[#ACA5BC]">
              Session Mode Mix
            </h3>
            <div className="grid grid-cols-3 gap-2 text-center">
              {(["Continuous", "Word", "Fingerspelling"] as RecognitionMode[]).map((m) => (
                <div
                  key={m}
                  className="p-2.5 rounded-xl bg-[#161221] border border-white/10"
                >
                  <div className="text-base font-extrabold text-[#FAF6F0] tabular-nums">
                    {modeCounts[m]}
                  </div>
                  <div className="text-[11px] font-semibold text-[#ACA5BC] truncate">
                    {m}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Full Session Timeline */}
        <div className="lg:col-span-7 sb-surface-card rounded-[24px] p-5 flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-[#251F35] text-amber-400 flex items-center justify-center">
                <Clock className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-base font-extrabold text-[#FAF6F0]">
                  Recorded Recognition Sessions ({filteredSessions.length})
                </h2>
                <p className="text-[11px] font-medium text-[#ACA5BC]">
                  Replay speech via Kokoro TTS or copy recognized transcripts
                </p>
              </div>
            </div>

            {sessions.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  clearSessions()
                  toast.info("Cleared session history")
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#1D1829] border border-white/10 text-xs font-semibold text-[#ACA5BC] hover:text-red-600 hover:border-red-200 transition cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear All</span>
              </button>
            )}
          </div>

          {/* Mode Filter Pills */}
          <div className="flex flex-wrap items-center gap-1.5">
            {(["All", "Continuous", "Word", "Fingerspelling"] as ModeFilter[]).map((tab) => {
              const active = modeFilter === tab
              return (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setModeFilter(tab)}
                  className={`px-3 py-1 rounded-full text-xs font-bold transition cursor-pointer ${
                    active
                      ? "sb-orange-pill text-white"
                      : "bg-[#161221] border border-white/10 text-[#ACA5BC] hover:text-[#FAF6F0]"
                  }`}
                >
                  {tab}
                </button>
              )
            })}
          </div>

          {filteredSessions.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center py-14 px-4 rounded-2xl bg-[#161221] border border-dashed border-white/14">
              <div className="w-10 h-10 rounded-full bg-amber-400/18 text-amber-400 flex items-center justify-center mb-2.5">
                <History className="w-5 h-5" />
              </div>
              <p className="text-sm font-bold text-[#FAF6F0]">
                {sessions.length === 0
                  ? "No recorded sessions yet"
                  : `No ${modeFilter} sessions yet`}
              </p>
              <p className="text-xs text-[#ACA5BC] mt-1 max-w-sm leading-relaxed">
                Start live recognition or fingerspelling to capture real recognition runs, timestamps, and speech transcripts.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-white/10 max-h-[460px] overflow-y-auto pr-1">
              {filteredSessions.map((s) => (
                <div
                  key={s.id}
                  className="py-3.5 first:pt-1 last:pb-1 flex flex-col sm:flex-row sm:items-start justify-between gap-3"
                >
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-extrabold text-[#FAF6F0] truncate">
                        {s.title}
                      </span>
                      <span className="px-2 py-0.5 rounded-full bg-[#251F35] border border-amber-400/30 text-amber-300 text-[10px] font-bold shrink-0">
                        {s.mode}
                      </span>
                      {typeof s.confidence === "number" && (
                        <span className="px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-400/30 text-emerald-300 text-[10px] font-bold tabular-nums shrink-0">
                          {Math.round(s.confidence)}% conf
                        </span>
                      )}
                    </div>

                    <p className="text-xs sm:text-sm font-medium text-[#D4CEE2] break-words leading-relaxed">
                      {s.text}
                    </p>

                    {s.glosses.length > 0 && (
                      <div className="flex flex-wrap gap-1 pt-0.5">
                        {s.glosses.slice(0, 8).map((gloss, idx) => (
                          <span
                            key={`${s.id}-${gloss}-${idx}`}
                            className="px-2 py-0.5 rounded-md bg-[#161221] border border-white/10 text-[10px] font-mono font-semibold text-[#ACA5BC]"
                          >
                            {gloss}
                          </span>
                        ))}
                        {s.glosses.length > 8 && (
                          <span className="text-[10px] font-semibold text-[#ACA5BC] self-center">
                            +{s.glosses.length - 8} more
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="flex sm:flex-col items-center sm:items-end justify-between gap-2 shrink-0">
                    <span className="text-[11px] font-medium text-[#ACA5BC] tabular-nums">
                      {formatSessionTimestamp(s.createdAt)}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => void copyText(s.text)}
                        aria-label={`Copy transcript for ${s.title}`}
                        className="p-1.5 rounded-lg bg-[#161221] border border-white/10 text-[#ACA5BC] hover:text-[#FAF6F0] hover:bg-[#251F35] transition cursor-pointer"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => void speakText(s.text)}
                        aria-label={`Speak transcript for ${s.title}`}
                        className="p-1.5 rounded-lg bg-[#161221] border border-white/10 text-amber-400 hover:bg-[#251F35] transition cursor-pointer"
                      >
                        <Volume2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
