"use client"

import * as React from "react"
import Image from "next/image"
import { motion, useReducedMotion } from "framer-motion"
import { Play, UploadCloud, Keyboard, Languages, Sparkles } from "lucide-react"
import { BridgeScene3D } from "@/components/visuals/bridge-scene-3d"

interface QuickActionsCardProps {
  onNavigate: (tab: string) => void
}

const QUICK_TILES: Array<{
  id: string
  label: string
  sublabel: string
  tab: string
  icon: "play" | "translate" | "video" | "custom"
}> = [
  {
    id: "recognition",
    label: "Live Studio",
    sublabel: "Real-time sign to text",
    tab: "live",
    icon: "play"
  },
  {
    id: "translate",
    label: "Translate",
    sublabel: "ASL ↔ English & speech",
    tab: "comm",
    icon: "translate"
  },
  {
    id: "fingerspelling",
    label: "Fingerspelling",
    sublabel: "Spell words A – Z",
    tab: "fingerspelling",
    icon: "video"
  },
  {
    id: "video-upload",
    label: "Video Upload",
    sublabel: "Analyze saved clips",
    tab: "video",
    icon: "custom"
  }
]

function QuickTileIcon({ kind }: { kind: "play" | "translate" | "video" | "custom" }) {
  const baseClass =
    "w-9 h-9 rounded-2xl bg-[#251F35] border border-amber-400/30 flex items-center justify-center shrink-0 shadow-[0_4px_12px_-6px_rgba(255,121,0,0.35)] text-amber-400"

  if (kind === "play") {
    return (
      <div className={baseClass}>
        <Play className="w-4 h-4 fill-amber-400 ml-0.5" />
      </div>
    )
  }
  if (kind === "translate") {
    return (
      <div className={baseClass}>
        <Languages className="w-4 h-4" />
      </div>
    )
  }
  if (kind === "video") {
    return (
      <div className={baseClass}>
        <Keyboard className="w-4 h-4" />
      </div>
    )
  }
  return (
    <div className={baseClass}>
      <UploadCloud className="w-4 h-4" />
    </div>
  )
}

export function QuickActionsCard({ onNavigate }: QuickActionsCardProps) {
  const prefersReduced = useReducedMotion()

  return (
    <motion.section
      initial={prefersReduced ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: prefersReduced ? 0 : 0.25, delay: prefersReduced ? 0 : 0.04 }}
      className="sb-surface-card relative rounded-[26px] p-5 overflow-hidden"
    >
      {/* Subtle Silk Wave Accent Artwork */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
        <Image
          src="/illustrations/quick-actions-wave.jpg"
          alt=""
          fill
          sizes="(max-width: 1024px) 100vw, 420px"
          className="object-cover object-top opacity-15 mix-blend-screen"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-[#1D1829]/25 via-[#1D1829]/82 to-[#1D1829]" />
      </div>

      {/* Interactive 3D SignBridge Arch Header */}
      <div className="relative z-10 flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="w-7 h-7 rounded-xl bg-[#251F35] border border-amber-400/30 flex items-center justify-center text-amber-400">
            <Sparkles className="w-3.5 h-3.5" />
          </span>
          <div>
            <h2 className="text-[16px] font-extrabold text-[#FAF6F0] tracking-tight leading-none">
              Quick Actions
            </h2>
            <p className="text-[11px] font-medium text-[#ACA5BC] mt-0.5">
              Sign • Understand • Connect
            </p>
          </div>
        </div>
      </div>

      {/* Lightweight GPU 3D Sign-to-Voice Arch */}
      <div className="relative z-10 rounded-2xl bg-gradient-to-r from-[#251E34]/95 via-[#1D1829]/95 to-[#162922]/75 border border-white/10 mb-3.5 overflow-hidden">
        <BridgeScene3D className="h-[78px]" />
        <div className="pointer-events-none flex items-center justify-between px-3 pb-1.5 -mt-2 text-[10px] font-bold tracking-wide text-[#ACA5BC]">
          <span className="inline-flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[#E5560B]" />
            ASL Gestures
          </span>
          <span className="inline-flex items-center gap-1">
            Kokoro Speech
            <span className="w-1.5 h-1.5 rounded-full bg-[#1FA968]" />
          </span>
        </div>
      </div>

      {/* 2x2 Tactile Tile Grid */}
      <div className="relative z-10 grid grid-cols-2 gap-2.5">
        {QUICK_TILES.map((tile) => (
          <motion.button
            key={tile.id}
            type="button"
            whileHover={prefersReduced ? undefined : { y: -2, scale: 1.01 }}
            whileTap={prefersReduced ? undefined : { scale: 0.98 }}
            transition={{ duration: 0.16 }}
            onClick={() => onNavigate(tile.tab)}
            className="text-left p-3 rounded-[18px] bg-[#1D1829]/95 hover:bg-[#231D32] border border-white/10 hover:border-amber-400/45 shadow-[0_8px_20px_-14px_rgba(30,23,19,0.2)] transition cursor-pointer flex flex-col gap-2 min-w-0"
          >
            <QuickTileIcon kind={tile.icon} />
            <div className="min-w-0">
              <div className="text-[13px] font-bold text-[#FAF6F0] leading-tight truncate">
                {tile.label}
              </div>
              <div className="text-[11px] font-medium text-[#ACA5BC] mt-0.5 leading-snug line-clamp-1">
                {tile.sublabel}
              </div>
            </div>
          </motion.button>
        ))}
      </div>
    </motion.section>
  )
}
