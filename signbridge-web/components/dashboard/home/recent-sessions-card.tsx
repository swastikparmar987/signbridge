"use client"

import * as React from "react"
import { motion, useReducedMotion } from "framer-motion"
import { Clock, Play, Copy, Volume2, Trash2, History } from "lucide-react"
import { toast } from "sonner"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from "@/components/ui/dialog"
import {
  SignSession,
  formatSessionTimestamp,
  useSignBridgeStore
} from "@/lib/signbridge-store"

export function RecentSessionsCard({
  onNavigateTab
}: {
  onNavigateTab: (tabId: string) => void
}) {
  const prefersReduced = useReducedMotion()
  const { sessions, clearSessions, preferences } = useSignBridgeStore()
  const [selectedSession, setSelectedSession] = React.useState<SignSession | null>(null)

  const recentThree = sessions.slice(0, 3)

  const speakText = async (text: string) => {
    if (!text) return
    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice: preferences.ttsVoice })
      })
      if (res.ok) {
        const data = await res.json()
        const src = data.audio_url || data.audio_data
        if (src) {
          const audio = new Audio(src)
          await audio.play()
          return
        }
      }
    } catch {
      // Fallback below
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))
    }
  }

  return (
    <>
      <motion.section
        initial={prefersReduced ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: prefersReduced ? 0 : 0.25, delay: 0.1 }}
        aria-label="Recent Sessions"
        className="sb-dashboard-card rounded-[24px] p-4 sm:p-5 flex flex-col justify-between gap-3"
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-7 h-7 rounded-lg bg-[#251F35] text-amber-400 flex items-center justify-center shrink-0">
              <Clock className="w-4 h-4" />
            </div>
            <h2 className="text-[16px] font-extrabold tracking-tight text-[#FAF6F0] truncate">
              Recent Sessions
            </h2>
          </div>

          <button
            type="button"
            onClick={() => onNavigateTab("progress")}
            className="text-xs font-bold text-amber-400 hover:text-amber-300 transition cursor-pointer shrink-0"
          >
            View All →
          </button>
        </div>

        {recentThree.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center py-5 px-3 rounded-2xl bg-[#231D32] border border-dashed border-white/14">
            <div className="w-9 h-9 rounded-full bg-amber-400/18 text-amber-400 flex items-center justify-center mb-2">
              <History className="w-4 h-4" />
            </div>
            <p className="text-xs font-bold text-[#FAF6F0]">
              No recognition sessions yet
            </p>
            <p className="text-[11px] text-[#ACA5BC] mt-1 max-w-[230px] leading-relaxed">
              Sign with the live webcam above or fingerspell to record real session history.
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-white/10">
            {recentThree.map((session) => (
              <li key={session.id}>
                <motion.button
                  type="button"
                  whileHover={prefersReduced ? undefined : { x: 2 }}
                  whileTap={prefersReduced ? undefined : { scale: 0.99 }}
                  onClick={() => setSelectedSession(session)}
                  className="w-full py-2.5 flex items-center justify-between gap-3 text-left group cursor-pointer rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="w-7 h-7 rounded-full bg-[#251F35] text-amber-400 group-hover:bg-[#D97706] group-hover:text-white flex items-center justify-center transition-colors shrink-0">
                      <Play className="w-3 h-3 fill-current translate-x-[0.5px]" />
                    </span>
                    <div className="min-w-0">
                      <div className="text-[12.5px] font-bold text-[#FAF6F0] truncate">
                        {session.title}
                      </div>
                      <div className="text-[10.5px] font-medium text-[#ACA5BC] truncate">
                        {session.mode}
                        {session.glosses.length > 0
                          ? ` · ${session.glosses.length} ${
                              session.glosses.length === 1 ? "token" : "tokens"
                            }`
                          : ""}
                      </div>
                    </div>
                  </div>
                  <span className="text-[11px] font-medium text-[#ACA5BC] shrink-0">
                    {formatSessionTimestamp(session.createdAt)}
                  </span>
                </motion.button>
              </li>
            ))}
          </ul>
        )}
      </motion.section>

      <Dialog
        open={Boolean(selectedSession)}
        onOpenChange={(open) => !open && setSelectedSession(null)}
      >
        <DialogContent className="bg-[#1D1829] border-white/10 text-[#FAF6F0] sm:max-w-md rounded-3xl">
          <DialogHeader>
            <DialogTitle className="text-base font-extrabold">
              {selectedSession?.title}
            </DialogTitle>
            <DialogDescription className="text-xs text-[#ACA5BC]">
              {selectedSession
                ? `${selectedSession.mode} Mode · ${formatSessionTimestamp(
                    selectedSession.createdAt
                  )}`
                : ""}
            </DialogDescription>
          </DialogHeader>

          {selectedSession && (
            <div className="space-y-4 pt-1">
              <div className="p-3.5 rounded-2xl bg-[#251F35] border border-white/10">
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#ACA5BC] block mb-1">
                  Captured Transcript
                </span>
                <p className="text-sm font-bold text-[#FAF6F0] break-words">
                  {selectedSession.text}
                </p>
              </div>

              {selectedSession.glosses.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {selectedSession.glosses.map((g, i) => (
                    <span
                      key={`${g}-${i}`}
                      className="px-2.5 py-1 rounded-full bg-amber-400/18 text-amber-300 text-[11px] font-bold"
                    >
                      {g}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex items-center justify-between gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => {
                    clearSessions()
                    setSelectedSession(null)
                    toast.info("Cleared session history")
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold text-[#ACA5BC] hover:text-red-600 hover:bg-red-50 transition cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear History</span>
                </button>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard.writeText(selectedSession.text)
                      toast.success("Copied transcript")
                    }}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-[#251F35] border border-white/10 text-xs font-bold text-[#FAF6F0] hover:bg-amber-400/18 transition cursor-pointer"
                  >
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => speakText(selectedSession.text)}
                    className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-[#FF8C1A] to-[#E85D04] text-xs font-bold text-white shadow-sm cursor-pointer"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                    <span>Replay Voice</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}

