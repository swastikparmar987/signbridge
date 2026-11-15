"use client"

import * as React from "react"
import Image from "next/image"
import { Cpu, Layers } from "lucide-react"
import { LiveRecognitionCard } from "@/components/dashboard/home/live-recognition-card"
import { QuickActionsCard } from "@/components/dashboard/home/quick-actions-card"
import { ModelPerformanceCard } from "@/components/dashboard/home/model-performance-card"
import { RecentSessionsCard } from "@/components/dashboard/home/recent-sessions-card"
import { PipelineCompanionCard } from "@/components/dashboard/home/pipeline-companion-card"
import {
  DEFAULT_DISPLAY_NAME,
  getTimeGreetingForName,
  useSignBridgeStore
} from "@/lib/signbridge-store"

interface DashboardViewProps {
  onNavigate: (tab: string) => void
}

const emptySubscribe = () => () => {}

export function DashboardView({ onNavigate }: DashboardViewProps) {
  const { modelHealth, customSigns, preferences } = useSignBridgeStore()
  const greeting = React.useSyncExternalStore(
    emptySubscribe,
    () => getTimeGreetingForName(preferences.displayName),
    () =>
      preferences.displayName && preferences.displayName !== DEFAULT_DISPLAY_NAME
        ? `Welcome back, ${preferences.displayName}`
        : "Welcome to SignBridge"
  )

  const isOnline = modelHealth.status === "online"
  const isChecking = modelHealth.status === "checking"

  return (
    <div className="w-full max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 pt-2 pb-12 space-y-5">
      {/* Contextual Hero Greeting Banner (seamlessly sitting on top of warm 3D ribbon) */}
      <section className="relative overflow-hidden rounded-[28px] sb-glass-card px-5 sm:px-7 py-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 overflow-hidden"
        >
          <Image
            src="/illustrations/header-sunrise.jpg"
            alt=""
            fill
            priority
            sizes="100vw"
            className="object-cover object-[70%_50%] opacity-35"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-[#1D1829] via-[#1D1829]/85 to-[#15121E]/65" />
        </div>

        <div className="relative z-10 min-w-0">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#251F35] border border-amber-400/30 text-[11px] font-bold text-amber-300 mb-2">
            <Layers className="w-3.5 h-3.5" />
            <span>
              {modelHealth.activeExperiment?.includes("ensemble_v2")
                ? "Ensemble V2 Neural ASL Workspace · 2,731 Glosses"
                : "SignBridge Neural ASL Workspace · 2,731 Glosses"}
            </span>
          </div>
          <h1 className="text-[24px] sm:text-[30px] font-extrabold tracking-tight text-[#FAF6F0] leading-tight">
            {greeting} <span aria-hidden="true">👋</span>
          </h1>
          <p className="text-xs sm:text-sm font-medium text-[#ACA5BC] mt-1">
            Ready to turn signs into natural words and speech today?
          </p>
        </div>

        {/* Live Telemetry Pill Strip */}
        <div className="relative z-10 flex flex-wrap items-center gap-2.5 shrink-0">
          <div className="inline-flex items-center gap-2.5 px-4 py-2 rounded-full bg-[#1D1829]/90 border border-white/10 shadow-xs text-xs font-bold">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isOnline
                  ? "bg-[#1FA968] shadow-[0_0_0_3px_rgba(31,169,104,0.18)]"
                  : isChecking
                    ? "bg-[#F59E0B] animate-pulse"
                    : "bg-[#EF4444]"
              }`}
            />
            <span className={isOnline ? "text-emerald-300" : "text-[#FAF6F0]"}>
              {isOnline ? "Model Online" : isChecking ? "Connecting..." : "Model Offline"}
            </span>
            {modelHealth.hardwareDevice && (
              <>
                <span className="h-3.5 w-px bg-white/15" aria-hidden="true" />
                <span className="inline-flex items-center gap-1.5 text-[#C6BFD6] font-semibold">
                  <Cpu className="w-3.5 h-3.5 text-amber-400" />
                  <span>{modelHealth.hardwareDevice}</span>
                </span>
              </>
            )}
          </div>

          <button
            type="button"
            onClick={() => onNavigate("custom")}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-[#251F35] hover:bg-amber-400/18 border border-amber-400/30 text-xs font-bold text-amber-300 transition cursor-pointer"
          >
            <span>Custom Signs: {customSigns.length}</span>
          </button>
        </div>
      </section>

      {/* Main Row: Hero Live Stage (8 cols) + Quick Actions & 3D Bridge + Performance (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div className="lg:col-span-8 min-w-0">
          <LiveRecognitionCard onExpandStudio={() => onNavigate("live")} />
        </div>

        <div className="lg:col-span-4 flex flex-col gap-5 min-w-0">
          <QuickActionsCard onNavigate={onNavigate} />
          <ModelPerformanceCard onNavigateTab={onNavigate} />
        </div>
      </div>

      {/* Bottom Row: Recent Sessions Timeline (8 cols) + SignBridge AI Companion (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        <div className="lg:col-span-8 min-w-0">
          <RecentSessionsCard onNavigateTab={onNavigate} />
        </div>

        <div className="lg:col-span-4 min-w-0">
          <PipelineCompanionCard onNavigate={onNavigate} />
        </div>
      </div>
    </div>
  )
}
