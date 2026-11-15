"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import { BarChart3, ArrowRight } from "lucide-react"
import { useSignBridgeStore } from "@/lib/signbridge-store"

function CircularGauge({
  valuePct,
  label,
  displayText,
  color,
  trackColor,
  subtitle
}: {
  valuePct: number
  label: string
  displayText: string
  color: string
  trackColor: string
  subtitle?: string
}) {
  const prefersReduced = useReducedMotion()
  const radius = 34
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(100, valuePct))
  const dashOffset = circumference - (clamped / 100) * circumference

  return (
    <div
      className="flex flex-col items-center justify-center rounded-2xl bg-[#231D32] border border-white/10 px-2.5 py-3 text-center"
      title={subtitle}
    >
      <div className="relative w-[78px] h-[78px] flex items-center justify-center">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 80 80" aria-hidden="true">
          <circle
            cx="40"
            cy="40"
            r={radius}
            fill="none"
            stroke={trackColor}
            strokeWidth="6.5"
          />
          <motion.circle
            cx="40"
            cy="40"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="6.5"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={
              prefersReduced
                ? { strokeDashoffset: dashOffset }
                : { strokeDashoffset: circumference }
            }
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: prefersReduced ? 0 : 0.7, ease: "easeOut" }}
          />
        </svg>
        <span className="absolute text-[13px] font-extrabold tracking-tight text-[#FAF6F0]">
          {displayText}
        </span>
      </div>
      <span className="mt-1.5 text-[10.5px] font-semibold text-[#ACA5BC] leading-tight">
        {label}
      </span>
    </div>
  )
}

export function ModelPerformanceCard({
  onNavigateTab
}: {
  onNavigateTab: (tabId: string) => void
}) {
  const prefersReduced = useReducedMotion()
  const { modelHealth } = useSignBridgeStore()

  const top1Val = modelHealth.productionTop1 ?? modelHealth.exp13Top1
  const top5Val = modelHealth.productionTop5 ?? modelHealth.exp13Top5
  const isV2 = Boolean(modelHealth.activeExperiment?.includes("ensemble_v2"))
  const latencyVal = modelHealth.lastLatencyMs

  // Scale latency arc cleanly onto 0..100 for 10..150ms range when measured
  const latencyFillPct =
    latencyVal !== null
      ? Math.max(18, Math.min(95, 100 - (latencyVal / 160) * 70))
      : modelHealth.status === "online"
        ? 100
        : 0

  return (
    <motion.section
      initial={prefersReduced ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: prefersReduced ? 0 : 0.25, delay: 0.08 }}
      aria-label="Model Performance"
      className="sb-dashboard-card rounded-[24px] p-4 sm:p-5 flex flex-col gap-3.5"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-[#251F35] text-amber-400 flex items-center justify-center shrink-0">
            <BarChart3 className="w-4 h-4" />
          </div>
          <h2 className="text-[16px] font-extrabold tracking-tight text-[#FAF6F0] truncate">
            Model Performance
          </h2>
          {modelHealth.activeExperiment && (
            <span
              title="Verified offline held-out test metrics from model artifacts (not live frame confidence)"
              className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full bg-amber-400/18 text-amber-300 text-[10px] font-bold shrink-0"
            >
              {isV2 ? "Ensemble V2" : modelHealth.productionTop1 !== null ? "Production" : "EXP13"}
            </span>
          )}
        </div>

        <motion.button
          type="button"
          whileHover={prefersReduced ? undefined : { x: 2 }}
          whileTap={prefersReduced ? undefined : { scale: 0.94 }}
          onClick={() => onNavigateTab("progress")}
          aria-label="View full analytics"
          className="p-1.5 rounded-lg text-amber-400 hover:bg-[#251F35] transition cursor-pointer shrink-0"
        >
          <ArrowRight className="w-4 h-4" />
        </motion.button>
      </div>

      <div className="grid grid-cols-3 gap-2.5">
        <CircularGauge
          valuePct={top1Val ?? 0}
          displayText={top1Val !== null ? `${top1Val.toFixed(2)}%` : "—"}
          label="Top-1 Accuracy"
          color="#FF7900"
          trackColor="rgba(245, 158, 11, 0.18)"
          subtitle="Verified active model held-out 2,731-class Top-1 test accuracy"
        />
        <CircularGauge
          valuePct={top5Val ?? 0}
          displayText={top5Val !== null ? `${top5Val.toFixed(2)}%` : "—"}
          label="Top-5 Accuracy"
          color="#7C3AED"
          trackColor="rgba(167, 139, 250, 0.18)"
          subtitle="Verified active model held-out 2,731-class Top-5 test accuracy"
        />
        <CircularGauge
          valuePct={latencyFillPct}
          displayText={
            latencyVal !== null
              ? `${latencyVal} ms`
              : modelHealth.status === "online"
                ? "Ready"
                : "Offline"
          }
          label="Latency"
          color="#22B66B"
          trackColor="rgba(52, 211, 153, 0.18)"
          subtitle="Measured live per real recognition request"
        />
      </div>
    </motion.section>
  )
}

