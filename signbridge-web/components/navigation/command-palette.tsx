"use client"

import * as React from "react"
import { AnimatePresence, motion, useReducedMotion } from "framer-motion"
import {
  Search,
  Video,
  Hand,
  Sparkles,
  UploadCloud,
  Settings,
  X,
  ArrowRight,
  Cpu,
  MessageSquare,
  BookOpen
} from "lucide-react"
import type { RecognitionMode, StudioWindowId } from "@/lib/signbridge-store"

export interface CommandPaletteProps {
  open: boolean
  onClose: () => void
  onOpenMode: (mode: RecognitionMode) => void
  onOpenWindow: (win: StudioWindowId) => void
  onSelectCustomGloss?: (gloss: string) => void
  onSelectDictionaryGloss?: (gloss: string) => void
}

type StudioRouteItem =
  | { kind: "mode"; id: RecognitionMode; label: string; desc: string; icon: React.ComponentType<{ className?: string }> }
  | { kind: "window"; id: StudioWindowId; label: string; desc: string; icon: React.ComponentType<{ className?: string }> }

const ROUTE_ITEMS: StudioRouteItem[] = [
  { kind: "mode", id: "Continuous", label: "Continuous Sign Studio", desc: "Natural multi-sign sentences with pause translation", icon: MessageSquare },
  { kind: "mode", id: "Word", label: "Words Sign Studio", desc: "Single ASL vocabulary sign recognition", icon: Video },
  { kind: "mode", id: "Fingerspelling", label: "Fingerspelling Studio", desc: "A–Z alphabet landmark recognition", icon: Hand },
  { kind: "window", id: "dictionary", label: "3D Sign Dictionary", desc: "Browse all 2,731 animated 3D ASL signs & step guides", icon: BookOpen },
  { kind: "window", id: "custom-learner", label: "Custom Sign Learner", desc: "Record, train & live-test your personal custom signs", icon: Sparkles },
  { kind: "window", id: "video", label: "Analyze Uploaded Video", desc: "Upload MP4/MOV/WebM clips for ASL recognition", icon: UploadCloud },
  { kind: "window", id: "settings", label: "Audio & Engine Settings", desc: "Kokoro TTS voice, confidence cutoff & custom routing", icon: Settings }
]

export function CommandPalette({
  open,
  onClose,
  onOpenMode,
  onOpenWindow,
  onSelectCustomGloss,
  onSelectDictionaryGloss
}: CommandPaletteProps) {
  const prefersReduced = useReducedMotion()
  const [query, setQuery] = React.useState("")
  const [vocab, setVocab] = React.useState<string[]>([])
  const [isLoadingVocab, setIsLoadingVocab] = React.useState(false)
  const inputRef = React.useRef<HTMLInputElement | null>(null)

  React.useEffect(() => {
    if (!open) return
    let active = true
    const t = window.setTimeout(() => {
      inputRef.current?.focus()
      if (vocab.length === 0) {
        setIsLoadingVocab(true)
        fetch("/api/vocabulary?limit=2731")
          .then((r) => (r.ok ? r.json() : null))
          .then((data) => {
            if (!active) return
            const rawList = Array.isArray(data?.items)
              ? data.items
              : Array.isArray(data?.glosses)
                ? data.glosses
                : Array.isArray(data?.vocabulary)
                  ? data.vocabulary
                  : []
            const items = rawList.map((v: string | { gloss?: string }) =>
              typeof v === "string" ? v : String(v.gloss ?? "")
            )
            setVocab(items.filter(Boolean))
          })
          .catch(() => {})
          .finally(() => {
            if (active) setIsLoadingVocab(false)
          })
      }
    }, 40)
    return () => {
      active = false
      window.clearTimeout(t)
    }
  }, [open, vocab.length])

  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        if (open) onClose()
      } else if (e.key === "Escape" && open) {
        onClose()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [open, onClose])

  const trimmed = query.trim().toLowerCase()

  const filteredRoutes = React.useMemo(() => {
    if (!trimmed) return ROUTE_ITEMS
    return ROUTE_ITEMS.filter(
      (r) =>
        r.label.toLowerCase().includes(trimmed) ||
        r.desc.toLowerCase().includes(trimmed) ||
        r.id.toLowerCase().includes(trimmed)
    )
  }, [trimmed])

  const filteredGlosses = React.useMemo(() => {
    if (!trimmed || vocab.length === 0) return []
    return vocab
      .filter((g) => g.toLowerCase().includes(trimmed))
      .slice(0, 8)
  }, [trimmed, vocab])

  return (
    <AnimatePresence>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh] bg-black/60 backdrop-blur-md"
          role="dialog"
          aria-modal="true"
          aria-label="Quick Command and ASL Vocabulary Search"
          onClick={onClose}
        >
          <motion.div
            initial={prefersReduced ? false : { opacity: 0, scale: 0.97, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={prefersReduced ? undefined : { opacity: 0, scale: 0.97, y: -6 }}
            transition={{ duration: 0.16 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-xl rounded-[26px] bg-[#1D1829] border border-white/12 shadow-[0_28px_80px_-14px_rgba(0,0,0,0.85)] overflow-hidden"
          >
            <div className="flex items-center gap-3 px-5 py-4 border-b border-white/10">
              <Search className="w-4 h-4 text-amber-400 shrink-0" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Jump to workspace or search ASL glosses (e.g. HELLO, THANKYOU)..."
                className="flex-1 bg-transparent text-sm font-medium text-[#FAF6F0] placeholder:text-[#7E7890] focus:outline-none"
              />
              <button
                type="button"
                onClick={onClose}
                aria-label="Close search"
                className="p-1.5 rounded-full hover:bg-white/10 text-[#ACA5BC] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="max-h-[420px] overflow-y-auto p-3 space-y-4">
              <div>
                <div className="px-2.5 pb-1.5 text-[11px] font-bold uppercase tracking-wider text-[#7E7890]">
                  Studio Modes &amp; Windows
                </div>
                <div className="space-y-1">
                  {filteredRoutes.map((route) => {
                    const Icon = route.icon
                    return (
                      <button
                        key={`${route.kind}:${route.id}`}
                        type="button"
                        onClick={() => {
                          if (route.kind === "mode") {
                            onOpenMode(route.id)
                          } else {
                            onOpenWindow(route.id)
                          }
                          onClose()
                        }}
                        className="w-full flex items-center justify-between px-3 py-2.5 rounded-2xl hover:bg-white/8 text-left transition cursor-pointer group"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <span className="w-8 h-8 rounded-xl bg-amber-400/15 text-amber-300 flex items-center justify-center shrink-0">
                            <Icon className="w-4 h-4" />
                          </span>
                          <div className="min-w-0">
                            <div className="text-sm font-bold text-[#FAF6F0] truncate">
                              {route.label}
                            </div>
                            <div className="text-xs text-[#ACA5BC] truncate">{route.desc}</div>
                          </div>
                        </div>
                        <ArrowRight className="w-4 h-4 text-[#7E7890] opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 transition shrink-0" />
                      </button>
                    )
                  })}
                  {filteredRoutes.length === 0 && filteredGlosses.length === 0 && (
                    <div className="px-3 py-6 text-center text-xs text-[#ACA5BC]">
                      {isLoadingVocab
                        ? "Searching vocabulary..."
                        : `No matching views or ASL glosses for "${query}".`}
                    </div>
                  )}
                </div>
              </div>

              {filteredGlosses.length > 0 && (
                <div>
                  <div className="px-2.5 pb-1.5 flex items-center justify-between text-[11px] font-bold uppercase tracking-wider text-[#7E7890]">
                    <span>ASL Vocabulary Glosses (Ensemble V2)</span>
                    <span className="inline-flex items-center gap-1 font-mono text-[10px] text-amber-300">
                      <Cpu className="w-3 h-3" /> 2,731 Classes
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5">
                    {filteredGlosses.map((gloss) => (
                      <button
                        key={gloss}
                        type="button"
                        onClick={() => {
                          if (onSelectDictionaryGloss) {
                            onSelectDictionaryGloss(gloss)
                            onOpenWindow("dictionary")
                          } else {
                            onSelectCustomGloss?.(gloss)
                            onOpenWindow("custom-learner")
                          }
                          onClose()
                        }}
                        className="flex items-center justify-between px-3 py-2 rounded-xl bg-[#261F36] hover:bg-amber-400/20 border border-white/10 text-xs font-bold text-[#FAF6F0] cursor-pointer transition"
                      >
                        <span className="truncate">{gloss}</span>
                        <span className="text-[10px] font-semibold text-amber-300 ml-2 shrink-0">
                          {onSelectDictionaryGloss ? "3D Guide" : "Customize"}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="px-5 py-2.5 border-t border-white/10 bg-[#171322] flex items-center justify-between text-[11px] text-[#ACA5BC]">
              <span>Navigate anywhere in SignBridge</span>
              <span className="font-mono">ESC to close</span>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}
