"use client"

import * as React from "react"
import Image from "next/image"

type LogoMarkSize = "xs" | "sm" | "md" | "lg"

const LOGO_CONTAINER_STYLES: Record<LogoMarkSize, string> = {
  xs: "h-6 w-6 rounded-lg p-0.5",
  sm: "h-9 w-9 rounded-xl p-1",
  md: "h-10 w-10 rounded-xl p-1",
  lg: "h-11 w-11 rounded-2xl p-1.5",
}

interface SignBridgeLogoMarkProps {
  size?: LogoMarkSize
  className?: string
  priority?: boolean
}

/** Shared brand mark rendering the SignBridge orange hand-with-star logo. */
export function SignBridgeLogoMark({
  size = "sm",
  className = "",
  priority = false,
}: SignBridgeLogoMarkProps) {
  return (
    <div
      className={`relative flex items-center justify-center shrink-0 bg-[#1F1A2E]/90 ring-1 ring-amber-400/30 shadow-[0_6px_18px_-6px_rgba(255,121,0,0.45)] ${LOGO_CONTAINER_STYLES[size]} ${className}`}
    >
      <Image
        src="/logo.png"
        alt="SignBridge logo"
        width={221}
        height={288}
        priority={priority}
        className="h-full w-full object-contain drop-shadow-[0_2px_6px_rgba(255,121,0,0.35)]"
      />
    </div>
  )
}

interface SignBridgeLogoProps {
  compact?: boolean
  showTagline?: boolean
  className?: string
}

/** Full SignBridge logo mark paired with wordmark and optional tagline. */
export function SignBridgeBridgeLogo({
  compact = false,
  showTagline = true,
  className = ""
}: SignBridgeLogoProps) {
  return (
    <div className={`inline-flex items-center gap-2.5 select-none ${className}`}>
      <SignBridgeLogoMark size={compact ? "sm" : "md"} priority />

      <div className="leading-none">
        <span
          className={`block font-extrabold tracking-tight text-[#FAF6F0] ${
            compact ? "text-[17px]" : "text-[19px]"
          }`}
        >
          Sign<span className="text-amber-400">Bridge</span>
        </span>
        {showTagline && (
          <span className="block text-[10px] font-medium tracking-[0.015em] text-[#ACA5BC] mt-0.5 whitespace-nowrap">
            Sign. Understand. Connect.
          </span>
        )}
      </div>
    </div>
  )
}

