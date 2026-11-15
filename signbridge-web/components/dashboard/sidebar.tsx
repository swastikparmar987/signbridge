"use client"

import * as React from "react"
import Image from "next/image"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import {
  Home,
  Video,
  CaseSensitive,
  BarChart3,
  Settings,
  X,
  LucideIcon
} from "lucide-react"

export interface NavItem {
  id: string
  label: string
  icon: LucideIcon
  matchIds?: string[]
}

export const NAV_ITEMS: NavItem[] = [
  { id: "home", label: "Home", icon: Home },
  { id: "live", label: "Live Recognition", icon: Video },
  { id: "fingerspelling", label: "Fingerspelling", icon: CaseSensitive },
  { id: "progress", label: "Analytics", icon: BarChart3 },
  { id: "settings", label: "Settings", icon: Settings }
]

interface SidebarProps {
  activeTab: string
  onSelectTab: (id: string) => void
  isMobileOpen?: boolean
  onCloseMobile?: () => void
}

function SignBridgeLogoMark() {
  return (
    <div className="relative flex items-center justify-center w-11 h-11 rounded-2xl bg-gradient-to-br from-[#FF8D1A] via-[#FF7900] to-[#E85D04] shadow-[0_8px_20px_-6px_rgba(255,121,0,0.6)] shrink-0">
      <svg
        viewBox="0 0 36 36"
        className="w-7 h-7 text-white"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <path
          d="M11 21.5V12.5C11 11.4 11.9 10.5 13 10.5C14.1 10.5 15 11.4 15 12.5V18"
          stroke="currentColor"
          strokeWidth="2.3"
          strokeLinecap="round"
        />
        <path
          d="M15 17.5V9.5C15 8.4 15.9 7.5 17 7.5C18.1 7.5 19 8.4 19 9.5V17.5"
          stroke="currentColor"
          strokeWidth="2.3"
          strokeLinecap="round"
        />
        <path
          d="M19 17.5V11C19 9.9 19.9 9 21 9C22.1 9 23 9.9 23 11V18.5"
          stroke="currentColor"
          strokeWidth="2.3"
          strokeLinecap="round"
        />
        <path
          d="M23 18.5V14.2C23 13.2 23.8 12.4 24.8 12.4C25.8 12.4 26.6 13.2 26.6 14.2V21.5C26.6 25.9 23.1 29.2 18.7 29.2C14.7 29.2 11.5 26.4 11 22.6L8.7 19.2C8.1 18.3 8.4 17 9.4 16.5C10.3 16 11.5 16.3 12.1 17.2L13.5 19.3"
          stroke="currentColor"
          strokeWidth="2.3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M7 10.5C8.2 8.7 10 7.4 12.1 6.8"
          stroke="rgba(255,255,255,0.8)"
          strokeWidth="1.8"
          strokeLinecap="round"
        />
      </svg>
    </div>
  )
}

function SidebarContent({
  activeTab,
  onSelectTab,
  onCloseMobile
}: {
  activeTab: string
  onSelectTab: (id: string) => void
  onCloseMobile?: () => void
}) {
  const prefersReduced = useReducedMotion()

  const isNavItemActive = (item: NavItem) => {
    if (item.id === activeTab) return true
    if (item.matchIds && item.matchIds.includes(activeTab)) return true
    return false
  }

  return (
    <div className="relative h-full w-full flex flex-col justify-between bg-gradient-to-b from-[#1D1829] via-[#181423] to-[#15121E] overflow-hidden select-none">
      {/* Scenic Vertical Torii Illustration anchored cleanly at bottom */}
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[315px] overflow-hidden"
        aria-hidden="true"
      >
        <Image
          src="/illustrations/sidebar-torii.jpg"
          alt=""
          fill
          priority
          sizes="250px"
          className="object-cover object-bottom opacity-20 mix-blend-screen"
        />
        {/* Smooth Cream Top Fade so nav items remain high-contrast */}
        <div className="absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-[#221C32] via-[#1B1626]/80 to-transparent" />
      </div>

      {/* Top Section: Brand + Navigation */}
      <div className="relative z-10 flex flex-col px-4 pt-6 pb-4">
        {/* Mobile close button */}
        {onCloseMobile && (
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation drawer"
            className="lg:hidden absolute top-4 right-4 p-1.5 rounded-xl text-[#ACA5BC] hover:bg-amber-400/20 hover:text-[#FAF6F0] transition cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        )}

        {/* Brand Identity */}
        <button
          type="button"
          onClick={() => {
            onSelectTab("home")
            onCloseMobile?.()
          }}
          className="flex flex-col items-center text-center pt-1 pb-6 px-2 group cursor-pointer focus:outline-none"
        >
          <motion.div
            whileHover={prefersReduced ? undefined : { scale: 1.05, y: -1 }}
            whileTap={prefersReduced ? undefined : { scale: 0.96 }}
            transition={{ duration: 0.2 }}
          >
            <SignBridgeLogoMark />
          </motion.div>

          <div className="mt-3 text-[21px] font-extrabold tracking-tight leading-none">
            <span className="text-[#FAF6F0]">Sign</span>
            <span className="text-amber-400">Bridge</span>
          </div>
          <p className="mt-1.5 text-[11px] font-medium tracking-wide text-[#ACA5BC]">
            Sign • Understand • Connect
          </p>
        </button>

        {/* Primary Navigation Links */}
        <nav aria-label="Primary navigation" className="flex flex-col gap-1.5 mt-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            const active = isNavItemActive(item)

            return (
              <motion.button
                key={item.id}
                type="button"
                onClick={() => {
                  onSelectTab(item.id)
                  onCloseMobile?.()
                }}
                whileHover={prefersReduced ? undefined : { x: 2 }}
                whileTap={prefersReduced ? undefined : { scale: 0.985 }}
                transition={{ duration: 0.18 }}
                aria-current={active ? "page" : undefined}
                className={`relative flex items-center gap-3.5 px-4 py-3 rounded-2xl text-[13.5px] font-semibold transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400 ${
                  active
                    ? "text-white"
                    : "text-[#4D4139] hover:text-[#FAF6F0] hover:bg-amber-400/20"
                }`}
              >
                {active && (
                  <motion.div
                    layoutId="signbridge-sidebar-active"
                    transition={
                      prefersReduced
                        ? { duration: 0 }
                        : { type: "spring", stiffness: 420, damping: 34 }
                    }
                    className="absolute inset-0 rounded-2xl bg-gradient-to-r from-[#FF9028] via-[#FF7900] to-[#E85D04] shadow-[0_8px_20px_-6px_rgba(232,93,4,0.55)]"
                  />
                )}
                <Icon
                  className={`relative z-10 w-[18px] h-[18px] shrink-0 transition-transform ${
                    active ? "text-white" : "text-[#ACA5BC]"
                  }`}
                  strokeWidth={active ? 2.25 : 1.9}
                />
                <span className="relative z-10 truncate">{item.label}</span>
              </motion.button>
            )
          })}
        </nav>
      </div>

      {/* Bottom Spacer leaving clean room for the scenic sunset/torii */}
      <div className="relative z-10 h-44 shrink-0 pointer-events-none" />
    </div>
  )
}

export function Sidebar({
  activeTab,
  onSelectTab,
  isMobileOpen = false,
  onCloseMobile
}: SidebarProps) {
  const prefersReduced = useReducedMotion()

  React.useEffect(() => {
    if (!isMobileOpen || !onCloseMobile) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseMobile()
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [isMobileOpen, onCloseMobile])

  return (
    <>
      {/* Desktop Fixed Vertical Sidebar (~250px) */}
      <aside
        aria-label="Desktop navigation sidebar"
        className="hidden lg:block fixed top-0 left-0 bottom-0 z-30 w-[250px] border-r border-white/10 shadow-[8px_0_32px_-16px_rgba(33,26,22,0.08)]"
      >
        <SidebarContent activeTab={activeTab} onSelectTab={onSelectTab} />
      </aside>

      {/* Mobile Drawer Navigation */}
      <AnimatePresence>
        {isMobileOpen && (
          <div className="fixed inset-0 z-50 lg:hidden flex">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: prefersReduced ? 0 : 0.2 }}
              onClick={onCloseMobile}
              className="fixed inset-0 bg-[#2C2440]/40 backdrop-blur-xs"
            />
            <motion.aside
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={
                prefersReduced
                  ? { duration: 0 }
                  : { type: "spring", stiffness: 360, damping: 32 }
              }
              aria-label="Mobile navigation sidebar"
              className="relative z-10 w-[250px] h-full border-r border-white/10 shadow-2xl"
            >
              <SidebarContent
                activeTab={activeTab}
                onSelectTab={onSelectTab}
                onCloseMobile={onCloseMobile}
              />
            </motion.aside>
          </div>
        )}
      </AnimatePresence>
    </>
  )
}
