"use client"

import * as React from "react"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import {
  ArrowLeft,
  Home,
  Video,
  Hand,
  Languages,
  Sparkles,
  BarChart3,
  UploadCloud,
  Settings,
  Search,
  Bell,
  ChevronDown,
  Menu,
  X,
  Cpu,
  CheckCheck
} from "lucide-react"
import { SignBridgeBridgeLogo } from "@/components/brand/signbridge-bridge-logo"
import {
  DEFAULT_DISPLAY_NAME,
  getDisplayInitials,
  useSignBridgeStore
} from "@/lib/signbridge-store"

export interface FloatingNavbarProps {
  activeTab: string
  onSelectTab: (tab: string) => void
  onOpenCommandPalette: () => void
  onBackToMenu?: () => void
}

const PRIMARY_NAV_ITEMS: Array<{
  id: string
  label: string
  icon: React.ComponentType<{ className?: string }>
}> = [
  { id: "home", label: "Home", icon: Home },
  { id: "live", label: "Live", icon: Video },
  { id: "fingerspelling", label: "Fingerspelling", icon: Hand },
  { id: "comm", label: "Translate", icon: Languages },
  { id: "custom", label: "My Signs", icon: Sparkles },
  { id: "progress", label: "Analytics", icon: BarChart3 }
]

const ALL_MOBILE_ITEMS: Array<{
  id: string
  label: string
  icon: React.ComponentType<{ className?: string }>
}> = [
  ...PRIMARY_NAV_ITEMS,
  { id: "video", label: "Video Upload", icon: UploadCloud },
  { id: "settings", label: "Settings", icon: Settings }
]

export function FloatingNavbar({
  activeTab,
  onSelectTab,
  onOpenCommandPalette,
  onBackToMenu
}: FloatingNavbarProps) {
  const prefersReduced = useReducedMotion()
  const {
    modelHealth,
    notifications,
    unreadCount,
    markAllNotificationsRead,
    customSigns,
    preferences
  } = useSignBridgeStore()
  const displayName = preferences.displayName || DEFAULT_DISPLAY_NAME
  const initials = getDisplayInitials(displayName)

  const [isNotifOpen, setIsNotifOpen] = React.useState(false)
  const [isProfileOpen, setIsProfileOpen] = React.useState(false)
  const [isMobileOpen, setIsMobileOpen] = React.useState(false)
  const navRef = React.useRef<HTMLElement | null>(null)

  React.useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (navRef.current && !navRef.current.contains(e.target as Node)) {
        setIsNotifOpen(false)
        setIsProfileOpen(false)
      }
    }
    document.addEventListener("mousedown", onDocClick)
    return () => document.removeEventListener("mousedown", onDocClick)
  }, [])

  const isOnline = modelHealth.status === "online"
  const isChecking = modelHealth.status === "checking"

  return (
    <>
      <header
        ref={navRef}
        className="sticky top-0 z-40 w-full px-3 sm:px-6 lg:px-8 pt-3.5 sm:pt-4 pb-2"
      >
        <div className="max-w-[1440px] mx-auto">
          <nav
            aria-label="Primary SignBridge navigation"
            className="sb-glass-nav relative rounded-full px-3 sm:px-4 lg:px-5 py-2 flex items-center justify-between gap-2"
          >
            {/* Left: Menu Back Pill + Bridge Logo Lockup */}
            <div className="flex items-center gap-1.5 shrink-0">
              {onBackToMenu && (
                <button
                  type="button"
                  onClick={onBackToMenu}
                  title="Return to Unified Studio Menu"
                  className="inline-flex items-center gap-1.5 rounded-full border border-white/12 bg-white/6 px-3 py-1.5 text-xs font-semibold text-[#FAF6F0] transition hover:border-amber-400/40 hover:bg-amber-400/12 cursor-pointer"
                >
                  <ArrowLeft className="h-3.5 w-3.5 text-amber-300" />
                  <span className="hidden sm:inline">Menu</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => onSelectTab("home")}
                aria-label="Go to SignBridge Home"
                className="flex items-center pl-1 pr-2 py-0.5 rounded-full hover:bg-white/6 transition cursor-pointer shrink-0"
              >
                <SignBridgeBridgeLogo compact />
              </button>
            </div>

            {/* Center: Floating Glass Pill Nav Items (Desktop) */}
            <div className="hidden lg:flex items-center gap-1 min-w-0">
              {PRIMARY_NAV_ITEMS.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onSelectTab(item.id)}
                    aria-current={isActive ? "page" : undefined}
                    className={`relative px-3 py-2 rounded-full text-[13px] font-bold flex items-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap ${
                      isActive
                        ? "text-white"
                        : "text-[#D4CEE2] hover:text-amber-200 hover:bg-white/8"
                    }`}
                  >
                    {isActive && (
                      <motion.span
                        layoutId="sb-nav-active-pill"
                        transition={
                          prefersReduced
                            ? { duration: 0 }
                            : { type: "spring", stiffness: 420, damping: 32 }
                        }
                        className="sb-orange-pill absolute inset-0 rounded-full"
                      />
                    )}
                    <Icon
                      className={`relative z-10 w-4 h-4 shrink-0 ${
                        isActive ? "text-white" : "text-[#ACA5BC]"
                      }`}
                    />
                    <span className="relative z-10">{item.label}</span>
                    {isActive && (
                      <span
                        aria-hidden="true"
                        className="absolute -bottom-[9px] left-1/2 -translate-x-1/2 w-8 h-[3px] rounded-full bg-gradient-to-r from-[#FFB347] via-[#FF7900] to-[#FFB347] shadow-[0_0_10px_#FF7900]"
                      />
                    )}
                  </button>
                )
              })}
            </div>

            {/* Right: Divider + Search + Quick Video/Settings + Notifications + SP Avatar */}
            <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
              <span
                className="hidden lg:block h-5 w-px bg-white/12 mr-1"
                aria-hidden="true"
              />

              {/* Command Search Button */}
              <button
                type="button"
                onClick={onOpenCommandPalette}
                title="Search workspace & ASL vocabulary (Cmd+K)"
                aria-label="Search workspace and ASL vocabulary"
                className="w-9 h-9 rounded-full bg-white/6 hover:bg-white/12 border border-white/12 text-[#FAF6F0] flex items-center justify-center transition cursor-pointer shadow-xs"
              >
                <Search className="w-4 h-4" />
              </button>

              {/* Video Upload Shortcut (Desktop) */}
              <button
                type="button"
                onClick={() => onSelectTab("video")}
                title="Video Clip Upload & Analysis"
                aria-label="Video Upload"
                className={`hidden sm:flex w-9 h-9 rounded-full border items-center justify-center transition cursor-pointer shadow-xs ${
                  activeTab === "video"
                    ? "sb-orange-pill border-transparent text-white"
                    : "bg-white/6 hover:bg-white/12 border-white/12 text-[#FAF6F0]"
                }`}
              >
                <UploadCloud className="w-4 h-4" />
              </button>

              {/* Settings Shortcut (Desktop) */}
              <button
                type="button"
                onClick={() => onSelectTab("settings")}
                title="Settings & Backend Engine Diagnostics"
                aria-label="Settings"
                className={`hidden sm:flex w-9 h-9 rounded-full border items-center justify-center transition cursor-pointer shadow-xs ${
                  activeTab === "settings"
                    ? "sb-orange-pill border-transparent text-white"
                    : "bg-white/6 hover:bg-white/12 border-white/12 text-[#FAF6F0]"
                }`}
              >
                <Settings className="w-4 h-4" />
              </button>

              {/* Notification Bell */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => {
                    setIsNotifOpen((o) => !o)
                    setIsProfileOpen(false)
                  }}
                  aria-label="Notifications"
                  aria-expanded={isNotifOpen}
                  className="relative w-9 h-9 rounded-full bg-white/6 hover:bg-white/12 border border-white/12 text-[#FAF6F0] flex items-center justify-center transition cursor-pointer shadow-xs"
                >
                  <Bell className="w-4 h-4" />
                  {unreadCount > 0 && (
                    <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-amber-400 ring-2 ring-[#1D1829]" />
                  )}
                </button>

                <AnimatePresence>
                  {isNotifOpen && (
                    <motion.div
                      initial={prefersReduced ? false : { opacity: 0, y: 6, scale: 0.97 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={prefersReduced ? undefined : { opacity: 0, y: 4, scale: 0.97 }}
                      transition={{ duration: 0.14 }}
                      className="absolute right-0 mt-3 w-80 rounded-[22px] bg-[#1D1829] border border-white/12 shadow-[0_22px_50px_-14px_rgba(0,0,0,0.85)] overflow-hidden z-50"
                    >
                      <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
                        <span className="text-xs font-extrabold text-[#FAF6F0]">
                          System & Session Activity
                        </span>
                        {unreadCount > 0 && (
                          <button
                            type="button"
                            onClick={markAllNotificationsRead}
                            className="inline-flex items-center gap-1 text-[11px] font-bold text-amber-300 hover:text-amber-200 cursor-pointer"
                          >
                            <CheckCheck className="w-3.5 h-3.5" />
                            Mark read
                          </button>
                        )}
                      </div>
                      <div className="max-h-72 overflow-y-auto divide-y divide-white/8">
                        {notifications.length === 0 ? (
                          <div className="px-4 py-6 text-center text-xs text-[#ACA5BC]">
                            All caught up. Start signing to log sessions.
                          </div>
                        ) : (
                          notifications.map((item) => (
                            <div
                              key={item.id}
                              className={`px-4 py-3 text-xs ${
                                item.unread ? "bg-amber-400/8" : "bg-transparent"
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <span className="font-bold text-[#FAF6F0]">{item.title}</span>
                                <span className="text-[10px] font-medium text-[#7E7890]">
                                  {item.time}
                                </span>
                              </div>
                              <p className="text-[11px] text-[#ACA5BC] mt-0.5 line-clamp-2">
                                {item.detail}
                              </p>
                            </div>
                          ))
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Profile Pill ("SP v") + System Telemetry Popover */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => {
                    setIsProfileOpen((o) => !o)
                    setIsNotifOpen(false)
                  }}
                  aria-label="User profile and backend status"
                  aria-expanded={isProfileOpen}
                  className="inline-flex items-center gap-1.5 pl-1 pr-2 py-1 rounded-full bg-white/8 hover:bg-white/14 border border-white/12 transition cursor-pointer shadow-xs"
                >
                  <span className="relative w-7 h-7 rounded-full bg-amber-400/20 border border-amber-400/40 text-amber-200 text-[11px] font-extrabold flex items-center justify-center">
                    {initials}
                    <span
                      className={`absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full ring-2 ring-[#1D1829] ${
                        isOnline
                          ? "bg-emerald-400"
                          : isChecking
                            ? "bg-amber-400"
                            : "bg-rose-400"
                      }`}
                    />
                  </span>
                  <ChevronDown className="w-3.5 h-3.5 text-[#ACA5BC]" />
                </button>

                <AnimatePresence>
                  {isProfileOpen && (
                    <motion.div
                      initial={prefersReduced ? false : { opacity: 0, y: 6, scale: 0.97 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={prefersReduced ? undefined : { opacity: 0, y: 4, scale: 0.97 }}
                      transition={{ duration: 0.14 }}
                      className="absolute right-0 mt-3 w-64 rounded-[22px] bg-[#1D1829] border border-white/12 shadow-[0_22px_50px_-14px_rgba(0,0,0,0.85)] p-2 z-50"
                    >
                      <div className="px-3 py-2.5 border-b border-white/10 mb-1">
                        <div className="text-xs font-extrabold text-[#FAF6F0]">
                          {displayName}
                        </div>
                        <div className="flex items-center gap-1.5 mt-1 text-[11px] font-semibold">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              isOnline
                                ? "bg-emerald-400"
                                : isChecking
                                  ? "bg-amber-400"
                                  : "bg-rose-400"
                            }`}
                          />
                          <span className={isOnline ? "text-emerald-300" : "text-[#ACA5BC]"}>
                            {isOnline
                              ? `Online • ${modelHealth.hardwareDevice ?? modelHealth.device ?? "Ready"}`
                              : isChecking
                                ? "Checking model..."
                                : "Backend offline"}
                          </span>
                        </div>
                        <div className="mt-1 text-[10px] font-mono text-[#7E7890] flex items-center gap-1">
                          <Cpu className="w-3 h-3" />
                          <span>
                            {modelHealth.numClasses ?? 2731} ASL glosses • {customSigns.length} custom
                          </span>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => {
                          onSelectTab("video")
                          setIsProfileOpen(false)
                        }}
                        className="w-full px-3 py-2 rounded-xl text-left text-xs font-semibold text-[#FAF6F0] hover:bg-white/8 flex items-center gap-2.5 transition cursor-pointer"
                      >
                        <UploadCloud className="w-3.5 h-3.5 text-amber-400" />
                        <span>Video Upload & Analysis</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          onSelectTab("custom")
                          setIsProfileOpen(false)
                        }}
                        className="w-full px-3 py-2 rounded-xl text-left text-xs font-semibold text-[#FAF6F0] hover:bg-white/8 flex items-center gap-2.5 transition cursor-pointer"
                      >
                        <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                        <span>My Custom Signs</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          onSelectTab("settings")
                          setIsProfileOpen(false)
                        }}
                        className="w-full px-3 py-2 rounded-xl text-left text-xs font-semibold text-[#FAF6F0] hover:bg-white/8 flex items-center gap-2.5 transition cursor-pointer"
                      >
                        <Settings className="w-3.5 h-3.5 text-amber-400" />
                        <span>Settings & Diagnostics</span>
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Mobile Hamburger Menu Button */}
              <button
                type="button"
                onClick={() => setIsMobileOpen(true)}
                aria-label="Open navigation menu"
                className="lg:hidden w-9 h-9 rounded-full bg-white/8 border border-white/12 text-[#FAF6F0] flex items-center justify-center cursor-pointer"
              >
                <Menu className="w-4 h-4" />
              </button>
            </div>
          </nav>
        </div>
      </header>

      {/* Mobile Navigation Drawer */}
      <AnimatePresence>
        {isMobileOpen && (
          <div className="fixed inset-0 z-50 lg:hidden">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsMobileOpen(false)}
              className="fixed inset-0 bg-[#1E1713]/30 backdrop-blur-xs"
            />
            <motion.aside
              initial={prefersReduced ? false : { x: "-100%" }}
              animate={{ x: 0 }}
              exit={prefersReduced ? undefined : { x: "-100%" }}
              transition={{ type: "spring", stiffness: 360, damping: 32 }}
              className="fixed inset-y-0 left-0 w-[290px] max-w-[85vw] bg-[#15121E] border-r border-white/12 shadow-2xl p-5 flex flex-col justify-between overflow-y-auto"
            >
              <div>
                <div className="flex items-center justify-between mb-6">
                  <SignBridgeBridgeLogo compact />
                  <button
                    type="button"
                    onClick={() => setIsMobileOpen(false)}
                    aria-label="Close navigation menu"
                    className="w-8 h-8 rounded-full bg-[#1D1829] border border-white/12 flex items-center justify-center text-[#FAF6F0] cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {onBackToMenu && (
                  <button
                    type="button"
                    onClick={() => {
                      setIsMobileOpen(false)
                      onBackToMenu()
                    }}
                    className="mb-3 w-full flex items-center gap-2.5 px-4 py-2.5 rounded-2xl border border-amber-400/30 bg-amber-400/10 text-xs font-bold text-amber-200 cursor-pointer"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    <span>Unified Studio Menu</span>
                  </button>
                )}

                <div className="space-y-1.5">
                  {ALL_MOBILE_ITEMS.map((item) => {
                    const Icon = item.icon
                    const isActive = activeTab === item.id
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => {
                          onSelectTab(item.id)
                          setIsMobileOpen(false)
                        }}
                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-bold transition cursor-pointer ${
                          isActive
                            ? "sb-orange-pill text-white"
                            : "text-[#D4CEE2] hover:bg-white/8"
                        }`}
                      >
                        <Icon className="w-4 h-4 shrink-0" />
                        <span>{item.label}</span>
                      </button>
                    )
                  })}
                </div>
              </div>

              <div className="mt-6 p-3.5 rounded-2xl bg-[#1D1829] border border-white/12 text-xs">
                <div className="flex items-center gap-2 font-bold text-[#FAF6F0]">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      isOnline ? "bg-emerald-400" : isChecking ? "bg-amber-400" : "bg-rose-400"
                    }`}
                  />
                  <span>{isOnline ? "Model Online" : isChecking ? "Connecting..." : "Model Offline"}</span>
                </div>
                <div className="text-[11px] text-[#ACA5BC] mt-1">
                  {modelHealth.hardwareDevice ?? modelHealth.device ?? "Local FastAPI Engine"}
                </div>
              </div>
            </motion.aside>
          </div>
        )}
      </AnimatePresence>
    </>
  )
}

