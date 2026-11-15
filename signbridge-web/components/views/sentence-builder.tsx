"use client"

import * as React from "react"
import { Volume2 } from "lucide-react"
import { motion } from "framer-motion"

interface SentenceBuilderProps {
  sentence?: string
  activeGloss?: string
  isAutoBuilding?: boolean
  onListen?: () => void
  className?: string
}

export function SentenceBuilderCard({
  sentence = "Hello, nice to meet you. Thank you for being here today.",
  activeGloss = "Thank you",
  isAutoBuilding = true,
  onListen,
  className
}: SentenceBuilderProps) {
  const words = sentence.trim().split(/\s+/).filter(Boolean)
  const wordCount = words.length

  // Highlight the active gloss dynamically within the sentence
  const renderHighlightedContent = () => {
    if (!activeGloss || !sentence.toLowerCase().includes(activeGloss.toLowerCase())) {
      return <span>“{sentence}”</span>
    }

    const regex = new RegExp(`(${activeGloss})`, "gi")
    const parts = sentence.split(regex)

    return (
      <span>
        “
        {parts.map((part, i) => {
          if (part.toLowerCase() === activeGloss.toLowerCase()) {
            return (
              <motion.span
                key={i}
                initial={{ scale: 0.96 }}
                animate={{ scale: 1 }}
                className="inline-block mx-1.5 px-3 py-1 rounded-xl bg-[#132E24] text-[#34D399] border border-[#059669]/50 font-semibold shadow-[0_0_15px_rgba(52,211,153,0.18)]"
              >
                {part}
              </motion.span>
            )
          }
          return <span key={i}>{part}</span>
        })}
        ”
      </span>
    )
  }

  return (
    <div
      className={`rounded-3xl bg-[#11141C] text-white border border-white/10 shadow-[0_12px_40px_rgba(0,0,0,0.5)] flex flex-col justify-between p-6 transition-all duration-300 ${className || ""}`}
    >
      {/* Top Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <span className="text-[11px] font-mono font-bold tracking-widest uppercase text-[#94A3B8]">
            SENTENCE BUILDER
          </span>
          <h3 className="text-xl font-bold tracking-tight text-white">Natural language</h3>
        </div>

        <button
          onClick={onListen}
          type="button"
          className="rounded-full bg-[#1F2533] hover:bg-[#2B3448] text-white border border-white/10 text-xs font-semibold px-4 py-2 flex items-center gap-2 transition active:scale-95 cursor-pointer shadow-sm"
        >
          <Volume2 className="size-4 text-[#FF5A2A]" />
          <span>Listen</span>
        </button>
      </div>

      {/* Main Quote with Highlighted Gloss */}
      <div className="py-5 text-xl sm:text-2xl font-medium tracking-tight leading-relaxed text-white">
        {renderHighlightedContent()}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-[#94A3B8] border-t border-white/10 pt-4">
        <div className="flex items-center gap-2">
          <span className="relative flex size-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full size-2.5 bg-emerald-500" />
          </span>
          <span className="font-medium text-emerald-400">
            {isAutoBuilding ? "Auto-building" : "Idle"}
          </span>
        </div>
        <span className="font-mono text-[#94A3B8] font-medium">{wordCount} words</span>
      </div>
    </div>
  )
}
