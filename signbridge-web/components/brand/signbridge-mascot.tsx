"use client"

import * as React from "react"
import Image from "next/image"
import { motion, useReducedMotion } from "framer-motion"

export type MascotState = "idle" | "listening" | "detected" | "speaking" | "helper"

interface SignBridgeMascotProps {
  state?: MascotState
  variant?: "wave" | "heart"
  size?: "sm" | "md" | "lg"
  caption?: string
  className?: string
}

const SIZE_CLASSES: Record<NonNullable<SignBridgeMascotProps["size"]>, string> = {
  sm: "w-14 h-14",
  md: "w-20 h-20",
  lg: "w-28 h-28"
}

export function SignBridgeMascot({
  state = "idle",
  variant = "wave",
  size = "md",
  caption,
  className = ""
}: SignBridgeMascotProps) {
  const prefersReduced = useReducedMotion()
  // Strictly preserve original owl mascot image assets untouched
  const src = variant === "heart" ? "/mascot-hero.png" : "/mascot.png"

  const animationProps = React.useMemo(() => {
    if (prefersReduced) return {}

    switch (state) {
      case "listening":
        return {
          animate: { y: [0, -3, 0], rotate: [0, -2, 0] },
          transition: { duration: 2.2, repeat: Infinity, ease: "easeInOut" as const }
        }
      case "detected":
        return {
          animate: { y: [0, -7, 0], scale: [1, 1.06, 1] },
          transition: { duration: 0.7, ease: "easeOut" as const }
        }
      case "speaking":
        return {
          animate: { y: [0, -4, 0], scale: [1, 1.03, 1] },
          transition: { duration: 0.9, repeat: Infinity, ease: "easeInOut" as const }
        }
      case "helper":
        return {
          whileHover: { scale: 1.06, rotate: 2 },
          animate: { y: [0, -3, 0] },
          transition: { duration: 3.2, repeat: Infinity, ease: "easeInOut" as const }
        }
      default:
        return {
          animate: { y: [0, -4, 0] },
          transition: { duration: 3.6, repeat: Infinity, ease: "easeInOut" as const }
        }
    }
  }, [state, prefersReduced])

  return (
    <div className={`inline-flex flex-col items-center gap-1.5 select-none ${className}`}>
      <div className="relative inline-flex items-center justify-center">
        {/* Soft warm halo */}
        <div
          aria-hidden="true"
          className={`absolute inset-1 rounded-full blur-md transition-opacity duration-300 ${
            state === "detected" || state === "speaking"
              ? "bg-[#FF7900]/30 opacity-100"
              : "bg-[#FFE6CC]/70 opacity-75"
          }`}
        />

        <motion.div
          className={`relative ${SIZE_CLASSES[size]} flex items-center justify-center`}
          {...animationProps}
        >
          <Image
            src={src}
            alt="SignBridge orange owl mascot"
            width={160}
            height={160}
            loading="eager"
            className="w-full h-full object-contain drop-shadow-[0_10px_18px_rgba(232,93,4,0.16)]"
          />
        </motion.div>

        {state === "speaking" && (
          <span
            aria-hidden="true"
            className="absolute -top-1 -right-1 inline-flex items-center gap-0.5 px-2 py-1 rounded-full bg-[#FF7900] text-white shadow-sm"
          >
            <span className="w-1 h-2 bg-white rounded-full animate-pulse" />
            <span className="w-1 h-3 bg-white rounded-full animate-pulse delay-75" />
            <span className="w-1 h-2 bg-white rounded-full animate-pulse delay-150" />
          </span>
        )}
      </div>

      {caption && (
        <span className="text-[11px] font-semibold text-[#6F645D] text-center leading-snug max-w-[190px]">
          {caption}
        </span>
      )}
    </div>
  )
}
