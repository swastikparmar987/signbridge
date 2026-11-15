"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import { Sparkles, Cpu, Volume2, MessageSquareText, ArrowUpRight, Video } from "lucide-react"
import { SignBridgeMascot } from "@/components/brand/signbridge-mascot"
import { useSignBridgeStore } from "@/lib/signbridge-store"

interface PipelineCompanionCardProps {
  onNavigate: (tab: string) => void
}

export function PipelineCompanionCard({ onNavigate }: PipelineCompanionCardProps) {
  const prefersReduced = useReducedMotion()
  const {
    modelHealth,
    pipelineStatus,
    customSigns,
    personalizationEnabled
  } = useSignBridgeStore()

  const isOnline = modelHealth.status === "online"
  const llmLabel = pipelineStatus.groq?.available
    ? "Groq Qwen3 Active"
    : pipelineStatus.qwen?.available
      ? "Local Qwen Ready"
      : "Rule Fallback"
  const ttsLabel = pipelineStatus.tts?.available
    ? `Kokoro ONNX · ${pipelineStatus.tts.voice || "af_sarah"}`
    : "Browser Speech"

  return (
    <motion.section
      initial={prefersReduced ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: prefersReduced ? 0 : 0.25, delay: prefersReduced ? 0 : 0.08 }}
      className="sb-surface-card relative rounded-[26px] p-5 flex flex-col justify-between overflow-hidden"
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-10 -right-10 w-40 h-40 rounded-full bg-radial from-[#FF9436]/18 to-transparent"
      />

      <div className="relative z-10 flex items-start justify-between gap-3">
        <div>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#251F35] border border-amber-400/30 text-[11px] font-bold text-amber-300">
            <Sparkles className="w-3 h-3" />
            AI Sign Companion
          </span>
          <h3 className="text-[16px] font-extrabold text-[#FAF6F0] tracking-tight mt-2.5">
            Personalize & Connect
          </h3>
          <p className="text-xs text-[#ACA5BC] mt-1 leading-relaxed">
            Live neural pipeline across gesture extraction, sentence smoothing, and custom signs.
          </p>
        </div>

        <SignBridgeMascot
          variant="heart"
          state={isOnline ? "helper" : "idle"}
          size="md"
          className="shrink-0 -mt-1"
        />
      </div>

      {/* Live Pipeline Rows (Strictly Backed By Real API Status) */}
      <div className="relative z-10 mt-4 space-y-2 text-xs">
        <div className="flex items-center justify-between px-3 py-2 rounded-2xl bg-[#161221] border border-white/10">
          <span className="inline-flex items-center gap-2 font-semibold text-[#FAF6F0]">
            <Cpu className="w-3.5 h-3.5 text-amber-400" />
            ASL Backbone
          </span>
          <span className="font-mono text-[11px] font-bold text-emerald-300">
            {modelHealth.numClasses ? `${modelHealth.numClasses.toLocaleString()} classes` : "Offline"}
          </span>
        </div>

        <div className="flex items-center justify-between px-3 py-2 rounded-2xl bg-[#161221] border border-white/10">
          <span className="inline-flex items-center gap-2 font-semibold text-[#FAF6F0]">
            <MessageSquareText className="w-3.5 h-3.5 text-amber-400" />
            NLP Engine
          </span>
          <span className="font-mono text-[11px] font-semibold text-[#ACA5BC] truncate max-w-[140px]">
            {llmLabel}
          </span>
        </div>

        <div className="flex items-center justify-between px-3 py-2 rounded-2xl bg-[#161221] border border-white/10">
          <span className="inline-flex items-center gap-2 font-semibold text-[#FAF6F0]">
            <Volume2 className="w-3.5 h-3.5 text-amber-400" />
            Neural Speech
          </span>
          <span className="font-mono text-[11px] font-semibold text-[#ACA5BC] truncate max-w-[145px]">
            {ttsLabel}
          </span>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="relative z-10 mt-4 pt-3 border-t border-white/10 flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => onNavigate("custom")}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-[#251F35] hover:bg-amber-400/18 border border-amber-400/30 text-xs font-bold text-amber-300 transition cursor-pointer"
        >
          <span>
            My Signs ({customSigns.length})
            {personalizationEnabled ? " · On" : ""}
          </span>
          <ArrowUpRight className="w-3.5 h-3.5" />
        </button>

        <button
          type="button"
          onClick={() => onNavigate("live")}
          className="sb-orange-pill inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-bold text-white transition hover:brightness-105 cursor-pointer"
        >
          <Video className="w-3.5 h-3.5" />
          <span>Full Studio</span>
        </button>
      </div>
    </motion.section>
  )
}

