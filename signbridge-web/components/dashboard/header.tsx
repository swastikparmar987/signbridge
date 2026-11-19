"use client"

import * as React from "react"
import Image from "next/image"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import {
  Bell,
  Menu,
  CheckCheck,
  User,
  BarChart3,
  Settings as SettingsIcon,
  Cpu,
  WifiOff
} from "lucide-react"
import {
  DEFAULT_DISPLAY_NAME,
  getDisplayInitials,
  getTimeGreetingForName,
  useSignBridgeStore
} from "@/lib/signbridge-store"

interface TopHeaderProps {
  onOpenMobileMenu?: () => void
  onNavigateTab?: (tabId: string) => void
  showScenicBackdrop?: boolean
}

export function TopHeader({
  onOpenMobileMenu,
  onNavigateTab,
  showScenicBackdrop = true
}: TopHeaderProps) {
  const prefersReduced = useReducedMotion()
  const {
    modelHealth,
    notifications,
    unreadCount,
    markAllNotificationsRead,
    preferences
  } = useSignBridgeStore()

  const greeting = React.useSyncExternalStore(
    () => () => {},
    () => getTimeGreetingForName(preferences.displayName),
    () => "Welcome to SignBridge"
  )
  const displayName = preferences.displayName || DEFAULT_DISPLAY_NAME
  const initials = getDisplayInitials(displayName)
  const [showNotifications, setShowNotifications] = React.useState(false)
  const [showProfileMenu, setShowProfileMenu] = React.useState(false)
  const headerRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const handleOutside = (e: MouseEvent) => {
      if (headerRef.current && !headerRef.current.contains(e.target as Node)) {
        setShowNotifications(false)
        setShowProfileMenu(false)
      }
    }
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setShowNotifications(false)
        setShowProfileMenu(false)
      }
    }
    document.addEventListener("mousedown", handleOutside)
    window.addEventListener("keydown", handleEscape)
    return () => {
      document.removeEventListener("mousedown", handleOutside)
      window.removeEventListener("keydown", handleEscape)
    }
  }, [])

  const isOnline = modelHealth.status === "online"
  const isChecking = modelHealth.status === "checking"
  const isAppleChip = Boolean(
    modelHealth.hardwareDevice && /^M\d/i.test(modelHealth.hardwareDevice.trim())
  )

  return (
    <header
      ref={headerRef}
      className="relative z-20 w-full overflow-visible px-4 sm:px-6 lg:px-8 pt-5 pb-6"
    >
      {/* Wide Scenic Mountain Sunrise Backdrop strictly bounded inside header */}
      {showScenicBackdrop && (
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
            className="object-cover object-[56%_48%] opacity-15 mix-blend-screen"
          />
          {/* Left-side & bottom soft cream blend so greeting copy pops cleanly */}
          <div className="absolute inset-0 bg-gradient-to-r from-[#15121E] via-[#15121E]/65 to-[#15121E]/85" />
          <div className="absolute inset-x-0 bottom-0 h-8 bg-gradient-to-t from-[#1D1829] to-transparent" />
        </div>
      )}

      {/* Header Content Row */}
      <div className="relative z-20 flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Left: Mobile Trigger + Greeting */}
        <div className="flex items-start sm:items-center gap-3 min-w-0">
          {onOpenMobileMenu && (
            <motion.button
              type="button"
              whileTap={prefersReduced ? undefined : { scale: 0.94 }}
              onClick={onOpenMobileMenu}
              aria-label="Open navigation menu"
              className="lg:hidden mt-0.5 sm:mt-0 p-2 rounded-xl bg-[#1D1829]/90 border border-white/10 text-[#FAF6F0] hover:bg-amber-400/20 transition cursor-pointer shrink-0"
            >
              <Menu className="w-5 h-5" />
            </motion.button>
          )}

          <div className="min-w-0">
            <h1 className="text-[22px] sm:text-[26px] font-extrabold tracking-tight text-[#FAF6F0] leading-tight truncate">
              {greeting} <span aria-hidden="true">👋</span>
            </h1>
            <p className="text-[13px] sm:text-sm font-medium text-[#ACA5BC] mt-0.5">
              Ready to turn signs into words today?
            </p>
          </div>
        </div>

        {/* Right: Model + Device Status Pill, Notifications, Profile Avatar */}
        <div className="flex items-center gap-2.5 sm:gap-3 self-start md:self-auto shrink-0">
          {/* Unified Model & Hardware Device Status Pill */}
          <div
            title={
              isOnline
                ? `SignBridge backend healthy (${modelHealth.numClasses ?? 2731} ASL classes)`
                : isChecking
                  ? "Connecting to SignBridge inference server..."
                  : "Backend offline — start FastAPI server on port 8000"
            }
            className="inline-flex items-center gap-2.5 px-3.5 py-2 rounded-full bg-[#1D1829]/92 backdrop-blur-md border border-white/10 shadow-[0_6px_18px_-8px_rgba(33,26,22,0.12)] text-xs font-semibold text-[#FAF6F0]"
          >
            <span className="inline-flex items-center gap-1.5">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  isOnline
                    ? "bg-[#22B66B] shadow-[0_0_0_3px_rgba(34,182,107,0.18)]"
                    : isChecking
                      ? "bg-amber-400 animate-pulse"
                      : "bg-red-500"
                }`}
              />
              <span className={isOnline ? "text-[#1E6B43] font-bold" : "text-[#FAF6F0]"}>
                {isOnline ? "Model Online" : isChecking ? "Connecting..." : "Model Offline"}
              </span>
            </span>

            {modelHealth.hardwareDevice && (
              <>
                <span className="h-3.5 w-px bg-[#E6D6C8]" aria-hidden="true" />
                <span className="inline-flex items-center gap-1.5 text-[#3B302A] font-semibold">
                  {isAppleChip ? (
                    <svg
                      viewBox="0 0 14 17"
                      className="w-3.5 h-3.5 fill-current text-[#FAF6F0]"
                      aria-hidden="true"
                    >
                      <path d="M11.6 9.1c0-2 1.6-3 1.7-3.1-.9-1.4-2.4-1.6-2.9-1.6-1.2-.1-2.4.7-3 .7-.6 0-1.6-.7-2.6-.7-1.3 0-2.6.8-3.3 2-.1.2-1.2 2.2-.3 5.3.6 1.7 1.4 3.6 2.5 3.6 1 0 1.4-.6 2.6-.6 1.2 0 1.5.6 2.6.6 1.1 0 1.8-1.7 2.4-3.4.8-1.9 1.1-2.8 1.1-2.9-.1 0-1.8-.7-1.8-2.9ZM9.5 3.1c.6-.7 1-1.7.9-2.7-.8 0-1.9.6-2.5 1.3-.5.6-1 1.6-.9 2.6.9.1 1.9-.5 2.5-1.2Z" />
                    </svg>
                  ) : (
                    <Cpu className="w-3.5 h-3.5 text-[#ACA5BC]" />
                  )}
                  <span>{modelHealth.hardwareDevice}</span>
                </span>
              </>
            )}
          </div>

          {/* Notification Bell Button + Dropdown */}
          <div className="relative">
            <motion.button
              type="button"
              whileHover={prefersReduced ? undefined : { scale: 1.05 }}
              whileTap={prefersReduced ? undefined : { scale: 0.94 }}
              transition={{ duration: 0.18 }}
              onClick={() => {
                setShowNotifications((v) => !v)
                setShowProfileMenu(false)
              }}
              aria-label="Notifications"
              aria-expanded={showNotifications}
              className="relative w-10 h-10 rounded-full bg-[#1D1829]/92 backdrop-blur-md border border-white/10 flex items-center justify-center text-[#FAF6F0] shadow-[0_6px_18px_-8px_rgba(33,26,22,0.12)] hover:border-amber-400/45 hover:bg-[#251F35] transition cursor-pointer"
            >
              <Bell className="w-[18px] h-[18px]" strokeWidth={2} />
              {unreadCount > 0 && (
                <span className="absolute top-2 right-2.5 w-2 h-2 rounded-full bg-[#E5560B] ring-2 ring-[#1D1829]" />
              )}
            </motion.button>

            <AnimatePresence>
              {showNotifications && (
                <motion.div
                  initial={{ opacity: 0, y: 8, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.96 }}
                  transition={{ duration: prefersReduced ? 0 : 0.18 }}
                  className="absolute right-0 mt-2.5 w-80 rounded-2xl bg-[#1D1829] border border-white/10 shadow-[0_20px_50px_-15px_rgba(33,26,22,0.2)] p-3.5 z-50"
                >
                  <div className="flex items-center justify-between pb-2.5 mb-2 border-b border-white/10">
                    <span className="text-xs font-extrabold uppercase tracking-wider text-[#FAF6F0]">
                      System Activity
                    </span>
                    {unreadCount > 0 && (
                      <button
                        type="button"
                        onClick={markAllNotificationsRead}
                        className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-400 hover:text-amber-300 cursor-pointer"
                      >
                        <CheckCheck className="w-3.5 h-3.5" />
                        <span>Mark read</span>
                      </button>
                    )}
                  </div>

                  {notifications.length === 0 ? (
                    <div className="py-6 text-center text-xs text-[#ACA5BC]">
                      All caught up. Model telemetry is quiet.
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {notifications.map((n) => (
                        <div
                          key={n.id}
                          className={`p-2.5 rounded-xl border text-left transition ${
                            n.unread
                              ? "bg-[#251F35] border-[#FFDCC0]"
                              : "bg-[#1D1829]/70 border-white/10"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-bold text-[#FAF6F0] flex items-center gap-1.5">
                              {n.kind === "system" && (
                                <WifiOff className="w-3.5 h-3.5 text-red-500 shrink-0" />
                              )}
                              {n.title}
                            </span>
                            <span className="text-[10px] font-medium text-[#ACA5BC] shrink-0">
                              {n.time}
                            </span>
                          </div>
                          <p className="text-[11px] text-[#ACA5BC] mt-0.5 line-clamp-2">
                            {n.detail}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Profile Avatar ('S') + Dropdown */}
          <div className="relative">
            <motion.button
              type="button"
              whileHover={prefersReduced ? undefined : { scale: 1.05 }}
              whileTap={prefersReduced ? undefined : { scale: 0.94 }}
              transition={{ duration: 0.18 }}
              onClick={() => {
                setShowProfileMenu((v) => !v)
                setShowNotifications(false)
              }}
              aria-label="Open user profile menu"
              aria-expanded={showProfileMenu}
              className="w-10 h-10 rounded-full bg-[#2C2440] text-white font-bold text-sm flex items-center justify-center shadow-[0_8px_18px_-6px_rgba(33,26,22,0.45)] ring-2 ring-[#1D1829] hover:ring-amber-400/45 transition cursor-pointer"
            >
              {initials}
            </motion.button>

            <AnimatePresence>
              {showProfileMenu && (
                <motion.div
                  initial={{ opacity: 0, y: 8, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.96 }}
                  transition={{ duration: prefersReduced ? 0 : 0.18 }}
                  className="absolute right-0 mt-2.5 w-64 rounded-2xl bg-[#1D1829] border border-white/10 shadow-[0_20px_50px_-15px_rgba(33,26,22,0.2)] p-3 z-50"
                >
                  <div className="p-3 rounded-xl bg-[#251F35] border border-[#FFE0C4] mb-2">
                    <div className="flex items-center gap-2.5">
                      <div className="w-9 h-9 rounded-full bg-[#2C2440] text-white font-bold text-sm flex items-center justify-center shrink-0">
                        {initials}
                      </div>
                      <div className="min-w-0">
                        <div className="text-xs font-extrabold text-[#FAF6F0] truncate">
                          {displayName}
                        </div>
                        <div className="text-[11px] font-medium text-[#ACA5BC]">
                          {isOnline
                            ? `${modelHealth.numClasses?.toLocaleString() ?? "2,731"} ASL classes ready`
                            : "Model offline"}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-1 text-xs font-semibold text-[#FAF6F0]">
                    <button
                      type="button"
                      onClick={() => {
                        onNavigateTab?.("progress")
                        setShowProfileMenu(false)
                      }}
                      className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl hover:bg-amber-400/20 transition cursor-pointer"
                    >
                      <BarChart3 className="w-4 h-4 text-[#ACA5BC]" />
                      <span>Analytics Overview</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        onNavigateTab?.("settings")
                        setShowProfileMenu(false)
                      }}
                      className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl hover:bg-amber-400/20 transition cursor-pointer"
                    >
                      <SettingsIcon className="w-4 h-4 text-[#ACA5BC]" />
                      <span>Settings</span>
                    </button>
                    <div className="h-px bg-white/12 my-1" />
                    <button
                      type="button"
                      onClick={() => setShowProfileMenu(false)}
                      className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-[#ACA5BC] hover:bg-[#251F35] transition cursor-pointer"
                    >
                      <User className="w-4 h-4" />
                      <span>Close Menu</span>
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </header>
  )
}
