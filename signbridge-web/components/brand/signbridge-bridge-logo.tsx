"use client"

import * as React from "react"

interface SignBridgeLogoProps {
  compact?: boolean
  showTagline?: boolean
  className?: string
}

export function SignBridgeBridgeLogo({
  compact = false,
  showTagline = true,
  className = ""
}: SignBridgeLogoProps) {
  return (
    <div className={`inline-flex items-center gap-2.5 select-none ${className}`}>
      {/* Sculpted Orange Suspension Bridge Mark (matching reference) */}
      <div className="relative flex items-center justify-center shrink-0">
        <svg
          viewBox="0 0 48 34"
          className={compact ? "w-9 h-6.5" : "w-11 h-8"}
          fill="none"
          aria-hidden="true"
        >
          <defs>
            <linearGradient id="sbBridgeGrad" x1="2" y1="28" x2="46" y2="6" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#FF9A3C" />
              <stop offset="55%" stopColor="#FF7900" />
              <stop offset="100%" stopColor="#E85D04" />
            </linearGradient>
          </defs>
          {/* Main sweeping span */}
          <path
            d="M3 25C10 11 22 6 35 12C40 14.3 43.5 17.5 45.5 21"
            stroke="url(#sbBridgeGrad)"
            strokeWidth="4.2"
            strokeLinecap="round"
          />
          {/* Secondary inner structural arch */}
          <path
            d="M8.5 27.5C15 16.5 25.5 12.5 36.5 17.5C40 19 42.5 21.5 44 24"
            stroke="url(#sbBridgeGrad)"
            strokeWidth="2.8"
            strokeLinecap="round"
            opacity="0.92"
          />
          {/* Bridge deck subtle curve */}
          <path
            d="M4.5 28.5C16.5 24 31 23 44 26.5"
            stroke="#E85D04"
            strokeWidth="2.2"
            strokeLinecap="round"
            opacity="0.7"
          />
          {/* Vertical support piers */}
          <path d="M16 14.5V25.2" stroke="#FFF8F1" strokeWidth="1.4" strokeLinecap="round" opacity="0.85" />
          <path d="M23 11.2V24.5" stroke="#FFF8F1" strokeWidth="1.4" strokeLinecap="round" opacity="0.85" />
          <path d="M30 11.5V24.6" stroke="#FFF8F1" strokeWidth="1.4" strokeLinecap="round" opacity="0.85" />
          {/* Warm apex spark dot */}
          <circle cx="24" cy="9.2" r="2.1" fill="#FFB347" />
        </svg>
      </div>

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

