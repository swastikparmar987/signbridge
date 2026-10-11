"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"

export interface NotchTab {
  id: string
  label: string
  icon?: string
}

interface NotchNavProps {
  tabs: NotchTab[]
  activeTab: string
  onTabChange: (id: string) => void
  onToggleTheme?: () => void
  onOpenSettings?: () => void
  className?: string
}

export function NotchNav({
  tabs,
  activeTab,
  onTabChange,
  onToggleTheme,
  onOpenSettings,
  className
}: NotchNavProps) {
  const [isOpen, setIsOpen] = React.useState(false)
  const currentTab = tabs.find((t) => t.id === activeTab) || tabs[0]

  return (
    <header
      className={cn(
        "fixed top-0 left-1/2 -translate-x-1/2 z-50 flex items-center select-none",
        className
      )}
    >
      {/* Notch Inverse Left Ear Curve */}
      <svg
        className="w-[18px] h-[18px] text-[#111318]/95 dark:text-[#111318]/95 block shrink-0"
        viewBox="0 0 20 20"
        fill="currentColor"
      >
        <path d="M0 0 C10 0 20 10 20 20 L20 0 Z" />
      </svg>

      {/* Main Notch Pill */}
      <div className="flex items-center gap-3 px-4 py-1.5 bg-[#111318]/95 text-white backdrop-blur-2xl border-b border-x border-white/10 shadow-[0_12px_32px_rgba(0,0,0,0.5)] rounded-b-2xl">
        {/* Left: Logo Notch */}
        <div
          className="flex items-center gap-2 cursor-pointer pr-3 border-r border-white/10 hover:opacity-90 transition"
          onClick={() => onTabChange("home")}
          title="SignBridge Home"
        >
          <div className="w-6 h-6 rounded-full bg-white flex items-center justify-center p-1 shadow-sm">
            <span className="text-[14px]">🤟</span>
          </div>
          <span className="font-bold text-xs tracking-tight text-white">
            SignBridge
          </span>
        </div>

        {/* Center: Dropdown Notch */}
        <div className="relative">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center gap-2 px-3 py-1 rounded-xl bg-white/10 hover:bg-white/15 text-white font-medium text-xs transition cursor-pointer"
          >
            <span className="w-2 h-2 rounded-full bg-[#FF5A2A]" />
            <span className="font-semibold text-xs">{currentTab?.label}</span>
            <svg
              className={cn(
                "w-3.5 h-3.5 text-neutral-400 transition-transform duration-200",
                isOpen && "rotate-180"
              )}
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>

          <AnimatePresence>
            {isOpen && (
              <motion.div
                initial={{ opacity: 0, y: -6, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.95 }}
                transition={{ duration: 0.16 }}
                className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-64 p-2 rounded-2xl bg-[#111318]/95 backdrop-blur-2xl border border-white/10 shadow-2xl space-y-1 z-50"
              >
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => {
                      onTabChange(tab.id)
                      setIsOpen(false)
                    }}
                    className={cn(
                      "flex items-center gap-3 w-full px-3 py-2 rounded-xl text-left text-xs transition cursor-pointer",
                      tab.id === activeTab
                        ? "bg-[#FF5A2A] text-white font-semibold"
                        : "text-neutral-300 hover:text-white hover:bg-white/10"
                    )}
                  >
                    {tab.icon && <span>{tab.icon}</span>}
                    <span>{tab.label}</span>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right: Action Notch (No model names, clean controls) */}
        <div className="flex items-center gap-1.5 pl-3 border-l border-white/10">
          {onToggleTheme && (
            <button
              onClick={onToggleTheme}
              className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-300 hover:text-white hover:bg-white/10 transition cursor-pointer"
              title="Toggle Light/Dark Theme"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
              </svg>
            </button>
          )}

          {onOpenSettings && (
            <button
              onClick={onOpenSettings}
              className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-300 hover:text-white hover:bg-white/10 transition cursor-pointer"
              title="Settings"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Notch Inverse Right Ear Curve */}
      <svg
        className="w-[18px] h-[18px] text-[#111318]/95 dark:text-[#111318]/95 block shrink-0"
        viewBox="0 0 20 20"
        fill="currentColor"
      >
        <path d="M20 0 C10 0 0 10 0 20 L0 0 Z" />
      </svg>
    </header>
  )
}
