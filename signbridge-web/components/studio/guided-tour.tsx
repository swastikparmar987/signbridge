"use client"

import * as React from "react"
import { ArrowLeft, ArrowRight, Sparkles, X } from "lucide-react"

export type TourPlacement = "top" | "bottom" | "left" | "right" | "auto"

export interface TourStep {
  id: string
  target: string
  title: string
  description: string
  placement?: TourPlacement
  badge?: string
  radius?: number
  onBeforeEnter?: () => void
}

interface GuidedTourProps {
  open: boolean
  steps: TourStep[]
  initialStepIndex?: number
  finishLabel?: string
  secondaryFinishAction?: {
    label: string
    onClick: () => void
  }
  onClose: () => void
}

interface BoxRect {
  top: number
  left: number
  width: number
  height: number
  right: number
  bottom: number
}

type ResolvedPlacement = "top" | "bottom" | "left" | "right"

interface BubbleGeometry {
  top: number
  left: number
  width: number
  placement: ResolvedPlacement
  arrowOffset: number
}

const VIEWPORT_MARGIN = 12
const BUBBLE_GAP = 14
const SPOTLIGHT_PAD = 6

function isElementVisible(el: HTMLElement | null): el is HTMLElement {
  if (!el) return false
  const rect = el.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return false
  const style = window.getComputedStyle(el)
  return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0"
}

function scrollTargetIntoViewIfNeeded(selector: string) {
  const el = document.querySelector<HTMLElement>(selector)
  if (!isElementVisible(el)) return
  const r = el.getBoundingClientRect()
  const topSafe = el.closest("header") ? 4 : 76
  const bottomSafe = window.innerHeight - 24
  if (r.top < topSafe || r.bottom > bottomSafe) {
    el.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" })
  }
}

function measureTarget(selector: string): BoxRect | null {
  const el = document.querySelector<HTMLElement>(selector)
  if (!isElementVisible(el)) return null
  const r = el.getBoundingClientRect()
  const top = Math.max(4, r.top - SPOTLIGHT_PAD)
  const left = Math.max(4, r.left - SPOTLIGHT_PAD)
  const right = Math.min(window.innerWidth - 4, r.right + SPOTLIGHT_PAD)
  const bottom = Math.min(window.innerHeight - 4, r.bottom + SPOTLIGHT_PAD)
  return {
    top,
    left,
    right,
    bottom,
    width: Math.max(16, right - left),
    height: Math.max(16, bottom - top),
  }
}

function computeBubbleGeometry(
  box: BoxRect,
  preferred: TourPlacement,
  bubbleW: number,
  bubbleH: number,
  vw: number,
  vh: number
): BubbleGeometry {
  const spaceTop = box.top - VIEWPORT_MARGIN - BUBBLE_GAP
  const spaceBottom = vh - box.bottom - VIEWPORT_MARGIN - BUBBLE_GAP
  const spaceLeft = box.left - VIEWPORT_MARGIN - BUBBLE_GAP
  const spaceRight = vw - box.right - VIEWPORT_MARGIN - BUBBLE_GAP

  let placement: ResolvedPlacement = "bottom"
  if (preferred === "top") {
    placement = spaceTop >= bubbleH || spaceTop > spaceBottom ? "top" : "bottom"
  } else if (preferred === "bottom") {
    placement = spaceBottom >= bubbleH || spaceBottom >= spaceTop ? "bottom" : "top"
  } else if (preferred === "left") {
    placement = spaceLeft >= bubbleW ? "left" : spaceBottom >= bubbleH ? "bottom" : "top"
  } else if (preferred === "right") {
    placement = spaceRight >= bubbleW ? "right" : spaceBottom >= bubbleH ? "bottom" : "top"
  } else {
    placement = spaceBottom >= bubbleH || spaceBottom >= spaceTop ? "bottom" : "top"
  }

  const centerX = box.left + box.width / 2
  const centerY = box.top + box.height / 2

  let top = 0
  let left = 0
  let arrowOffset = 24

  if (placement === "bottom" || placement === "top") {
    left = Math.min(
      Math.max(VIEWPORT_MARGIN, centerX - bubbleW / 2),
      vw - VIEWPORT_MARGIN - bubbleW
    )
    top =
      placement === "bottom"
        ? Math.min(vh - VIEWPORT_MARGIN - bubbleH, box.bottom + BUBBLE_GAP)
        : Math.max(VIEWPORT_MARGIN, box.top - bubbleH - BUBBLE_GAP)
    arrowOffset = Math.min(Math.max(20, centerX - left), bubbleW - 20)
  } else {
    top = Math.min(
      Math.max(VIEWPORT_MARGIN, centerY - bubbleH / 2),
      vh - VIEWPORT_MARGIN - bubbleH
    )
    left =
      placement === "right"
        ? Math.min(vw - VIEWPORT_MARGIN - bubbleW, box.right + BUBBLE_GAP)
        : Math.max(VIEWPORT_MARGIN, box.left - bubbleW - BUBBLE_GAP)
    arrowOffset = Math.min(Math.max(20, centerY - top), bubbleH - 20)
  }

  return { top, left, width: bubbleW, placement, arrowOffset }
}

/** Interactive button-by-button spotlight walkthrough with floating speech bubble. */
export function GuidedTour(props: GuidedTourProps) {
  if (!props.open || props.steps.length === 0) return null
  return <GuidedTourOverlay {...props} />
}

function GuidedTourOverlay({
  steps,
  initialStepIndex = 0,
  finishLabel = "Finish",
  secondaryFinishAction,
  onClose,
}: GuidedTourProps) {
  const maskId = React.useId().replace(/:/g, "_")
  const [stepIndex, setStepIndex] = React.useState(initialStepIndex)
  const [direction, setDirection] = React.useState<1 | -1>(1)
  const [targetBox, setTargetBox] = React.useState<BoxRect | null>(null)
  const [bubbleHeight, setBubbleHeight] = React.useState(196)
  const [viewport, setViewport] = React.useState<{ w: number; h: number }>(() =>
    typeof window !== "undefined"
      ? { w: window.innerWidth, h: window.innerHeight }
      : { w: 1280, h: 760 }
  )
  const bubbleRef = React.useRef<HTMLDivElement | null>(null)

  const clampedIndex = Math.max(0, Math.min(stepIndex, Math.max(0, steps.length - 1)))
  const activeStep = steps[clampedIndex]

  React.useEffect(() => {
    if (!activeStep) return

    activeStep.onBeforeEnter?.()

    let cancelled = false
    const updateMeasurement = (allowSkip: boolean) => {
      if (cancelled) return
      setViewport({ w: window.innerWidth, h: window.innerHeight })
      const measured = measureTarget(activeStep.target)
      if (measured) {
        setTargetBox(measured)
        return
      }
      if (allowSkip && steps.length > 1) {
        const nextIdx = clampedIndex + direction
        if (nextIdx >= 0 && nextIdx < steps.length) {
          setStepIndex(nextIdx)
        } else if (direction === 1 && clampedIndex > 0) {
          onClose()
        }
      }
    }

    const raf = window.requestAnimationFrame(() => {
      scrollTargetIntoViewIfNeeded(activeStep.target)
      updateMeasurement(false)
    })
    const t1 = window.setTimeout(() => {
      scrollTargetIntoViewIfNeeded(activeStep.target)
      updateMeasurement(false)
    }, 60)
    const t2 = window.setTimeout(() => updateMeasurement(true), 220)
    const t3 = window.setTimeout(() => updateMeasurement(false), 420)

    const onReflow = () => updateMeasurement(false)
    window.addEventListener("resize", onReflow)
    window.addEventListener("scroll", onReflow, true)

    return () => {
      cancelled = true
      window.cancelAnimationFrame(raf)
      window.clearTimeout(t1)
      window.clearTimeout(t2)
      window.clearTimeout(t3)
      window.removeEventListener("resize", onReflow)
      window.removeEventListener("scroll", onReflow, true)
    }
  }, [activeStep, clampedIndex, direction, steps.length, onClose])

  React.useEffect(() => {
    const el = bubbleRef.current
    if (!el || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver((entries) => {
      const h = entries[0]?.borderBoxSize?.[0]?.blockSize ?? el.offsetHeight
      if (h > 0) setBubbleHeight(h)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [clampedIndex])

  const handleNext = React.useCallback(() => {
    if (clampedIndex >= steps.length - 1) {
      onClose()
      return
    }
    setDirection(1)
    setStepIndex((i) => Math.min(steps.length - 1, i + 1))
  }, [clampedIndex, steps.length, onClose])

  const handleBack = React.useCallback(() => {
    if (clampedIndex <= 0) return
    setDirection(-1)
    setStepIndex((i) => Math.max(0, i - 1))
  }, [clampedIndex])

  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault()
        onClose()
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        e.preventDefault()
        handleNext()
      } else if (e.key === "ArrowLeft") {
        e.preventDefault()
        handleBack()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [onClose, handleNext, handleBack])

  if (!activeStep) return null

  const bubbleW = Math.min(344, Math.max(260, viewport.w - VIEWPORT_MARGIN * 2))
  const fallbackBox: BoxRect = {
    top: 120,
    left: Math.max(24, viewport.w / 2 - 100),
    width: 200,
    height: 48,
    right: Math.min(viewport.w - 24, viewport.w / 2 + 100),
    bottom: 168,
  }
  const box = targetBox ?? fallbackBox
  const geom = computeBubbleGeometry(
    box,
    activeStep.placement ?? "auto",
    bubbleW,
    bubbleHeight,
    viewport.w,
    viewport.h
  )
  const isLast = clampedIndex >= steps.length - 1
  const spotRadius = activeStep.radius ?? 18

  return (
    <div
      data-tour-overlay="true"
      role="dialog"
      aria-modal="true"
      aria-label={`Guided tour step ${clampedIndex + 1} of ${steps.length}: ${activeStep.title}`}
      className="fixed inset-0 z-[100] select-none"
    >
      {/* Fullscreen cutout overlay */}
      <svg
        aria-hidden="true"
        className="fixed inset-0 h-full w-full pointer-events-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <defs>
          <mask id={maskId}>
            <rect x="0" y="0" width="100%" height="100%" fill="white" />
            {targetBox && (
              <rect
                x={box.left}
                y={box.top}
                width={box.width}
                height={box.height}
                rx={spotRadius}
                ry={spotRadius}
                fill="black"
              />
            )}
          </mask>
        </defs>
        <rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          fill="rgba(10, 8, 17, 0.72)"
          mask={`url(#${maskId})`}
        />
      </svg>

      {/* Glowing Amber Ring Around Highlighted Button */}
      {targetBox && (
        <div
          aria-hidden="true"
          style={{
            top: box.top,
            left: box.left,
            width: box.width,
            height: box.height,
            borderRadius: spotRadius,
          }}
          className="pointer-events-none fixed border-2 border-amber-400 shadow-[0_0_0_4px_rgba(245,158,11,0.22),0_0_32px_rgba(245,158,11,0.5)] transition-all duration-200 ease-out"
        />
      )}

      {/* Floating Speech / Text Bubble */}
      <div
        ref={bubbleRef}
        style={{
          top: geom.top,
          left: geom.left,
          width: geom.width,
        }}
        className="fixed z-[101] rounded-2xl border border-amber-400/40 bg-[#1E192B]/96 p-4 text-[#FAF6F0] shadow-[0_24px_64px_rgba(0,0,0,0.78)] backdrop-blur-xl transition-all duration-200 ease-out"
      >
        {/* Pointer Arrow */}
        {targetBox && geom.placement === "bottom" && (
          <div
            aria-hidden="true"
            style={{ left: geom.arrowOffset - 7 }}
            className="absolute -top-[7px] h-3.5 w-3.5 rotate-45 border-l border-t border-amber-400/40 bg-[#1E192B]"
          />
        )}
        {targetBox && geom.placement === "top" && (
          <div
            aria-hidden="true"
            style={{ left: geom.arrowOffset - 7 }}
            className="absolute -bottom-[7px] h-3.5 w-3.5 rotate-45 border-r border-b border-amber-400/40 bg-[#1E192B]"
          />
        )}
        {targetBox && geom.placement === "right" && (
          <div
            aria-hidden="true"
            style={{ top: geom.arrowOffset - 7 }}
            className="absolute -left-[7px] h-3.5 w-3.5 rotate-45 border-b border-l border-amber-400/40 bg-[#1E192B]"
          />
        )}
        {targetBox && geom.placement === "left" && (
          <div
            aria-hidden="true"
            style={{ top: geom.arrowOffset - 7 }}
            className="absolute -right-[7px] h-3.5 w-3.5 rotate-45 border-t border-r border-amber-400/40 bg-[#1E192B]"
          />
        )}

        {/* Header Row */}
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-400/15 border border-amber-400/35 px-2.5 py-0.5 text-[11px] font-bold text-amber-300 tabular-nums">
              <Sparkles className="h-3 w-3" />
              <span>
                Step {clampedIndex + 1} of {steps.length}
              </span>
            </span>
            {activeStep.badge && (
              <span className="truncate rounded-full bg-white/8 px-2 py-0.5 text-[10px] font-semibold text-[#C9C3D8]">
                {activeStep.badge}
              </span>
            )}
          </div>

          <button
            type="button"
            aria-label="Close tour"
            onClick={onClose}
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-white/55 transition hover:bg-white/10 hover:text-white cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Feature Title & Description */}
        <h2 className="font-display text-sm font-bold text-white leading-snug">
          {activeStep.title}
        </h2>
        <p className="mt-1.5 text-xs leading-relaxed text-[#CFC8DF]">
          {activeStep.description}
        </p>

        {/* Step Controls */}
        <div className="mt-4 flex items-center justify-between gap-2 pt-2.5 border-t border-white/10">
          <div className="flex items-center gap-1.5">
            {clampedIndex > 0 ? (
              <button
                type="button"
                onClick={handleBack}
                className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/6 px-3 py-1.5 text-[11px] font-semibold text-white/85 transition hover:bg-white/12 cursor-pointer"
              >
                <ArrowLeft className="h-3 w-3" />
                <span>Back</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={onClose}
                className="rounded-full px-2.5 py-1.5 text-[11px] font-medium text-white/60 transition hover:bg-white/8 hover:text-white cursor-pointer"
              >
                Skip tour
              </button>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            {isLast && secondaryFinishAction ? (
              <>
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-full border border-white/15 bg-white/6 px-3 py-1.5 text-[11px] font-semibold text-white/85 transition hover:bg-white/12 cursor-pointer"
                >
                  {finishLabel}
                </button>
                <button
                  type="button"
                  onClick={secondaryFinishAction.onClick}
                  className="inline-flex items-center gap-1 rounded-full bg-gradient-to-b from-[#FBBF24] to-[#F59E0B] px-3.5 py-1.5 text-[11px] font-bold text-[#17110B] shadow-[0_6px_18px_rgba(245,158,11,0.4)] transition hover:brightness-105 cursor-pointer"
                >
                  <span>{secondaryFinishAction.label}</span>
                  <ArrowRight className="h-3 w-3" />
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={handleNext}
                className="inline-flex items-center gap-1 rounded-full bg-gradient-to-b from-[#FBBF24] to-[#F59E0B] px-3.5 py-1.5 text-[11px] font-bold text-[#17110B] shadow-[0_6px_18px_rgba(245,158,11,0.4)] transition hover:brightness-105 cursor-pointer"
              >
                <span>{isLast ? finishLabel : "Next"}</span>
                {!isLast && <ArrowRight className="h-3 w-3" />}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

interface TourAskPromptProps {
  open: boolean
  badge?: string
  title: string
  description: string
  stepCount: number
  onStartTour: () => void
  onDismiss: () => void
  onDontShowAgain: () => void
}

/** Non-blocking first-visit card asking whether the user wants a guided tour or opts out. */
export function TourAskPrompt({
  open,
  badge = "Quick Guide",
  title,
  description,
  stepCount,
  onStartTour,
  onDismiss,
  onDontShowAgain,
}: TourAskPromptProps) {
  if (!open) return null

  return (
    <div
      data-tour-prompt="true"
      role="dialog"
      aria-label={title}
      className="fixed bottom-5 right-5 z-[90] w-[min(380px,calc(100vw-2rem))] rounded-2xl border border-amber-400/45 bg-[#1B1628]/96 p-4 text-[#FAF6F0] shadow-[0_24px_64px_rgba(0,0,0,0.75)] backdrop-blur-xl"
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/35 bg-amber-400/15 px-2.5 py-0.5 text-[11px] font-bold text-amber-300">
          <Sparkles className="h-3 w-3" />
          <span>{badge}</span>
          <span className="text-amber-200/75">· {stepCount} steps</span>
        </span>
        <button
          type="button"
          aria-label="Close tour prompt"
          onClick={onDismiss}
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-white/55 transition hover:bg-white/10 hover:text-white cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <h2 className="font-display text-sm font-bold text-white leading-snug">
        {title}
      </h2>
      <p className="mt-1 text-xs leading-relaxed text-[#CFC8DF]">
        {description}
      </p>

      <div className="mt-3.5 flex flex-wrap items-center justify-between gap-2 pt-2.5 border-t border-white/10">
        <button
          type="button"
          onClick={onDontShowAgain}
          className="text-[11px] font-medium text-[#9F98B2] transition hover:text-white cursor-pointer"
        >
          Don&apos;t show again
        </button>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-full border border-white/12 bg-white/6 px-3 py-1.5 text-[11px] font-semibold text-white/80 transition hover:bg-white/12 cursor-pointer"
          >
            Not now
          </button>
          <button
            type="button"
            onClick={onStartTour}
            className="inline-flex items-center gap-1 rounded-full bg-gradient-to-b from-[#FBBF24] to-[#F59E0B] px-3.5 py-1.5 text-[11px] font-bold text-[#17110B] shadow-[0_6px_18px_rgba(245,158,11,0.4)] transition hover:brightness-105 cursor-pointer"
          >
            <span>Start Tour</span>
            <ArrowRight className="h-3 w-3" />
          </button>
        </div>
      </div>
    </div>
  )
}
