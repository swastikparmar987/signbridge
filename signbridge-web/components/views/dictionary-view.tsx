"use client"

import * as React from "react"
import { Search, X, Camera, CameraOff, ArrowRight, Sparkles, Film } from "lucide-react"
import { Dictionary3DStage } from "@/components/studio/dictionary-3d-stage"
import {
  clearMediaPipeHandsListener,
  getSharedMediaPipeHands,
  sendVideoFrameToSharedHands,
  setMediaPipeHandsListener,
  type MediaPipeHandsResults,
  type MediaPipeLandmark
} from "@/lib/mediapipe-hands"

export interface DictionaryViewProps {
  initialGloss?: string
  onTrySigningLive?: (gloss: string) => void
}

interface VocabListItem {
  gloss: string
  video_count: number
  has_demonstration: boolean
  tier: string
  two_hand: boolean
  handshape: string
}

interface DictionarySignDetail {
  gloss: string
  tier: string
  video_count: number
  video_url: string | null
  two_hand: boolean
  handshape: string
  steps: string[]
  total_frames: number
  frames: number[][][]
}

type CategoryFilter = "all" | "one_hand" | "two_hand" | "core"

const STARTER_GLOSSES = [
  "HELLO",
  "FRIEND",
  "THANKYOU",
  "HELP",
  "WATER",
  "BOOK",
  "PLEASE",
  "YES",
  "NO",
  "WORK",
  "FAMILY",
  "LOVE"
]

const BONE_PAIRS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20]
]

function computeHandFormSimilarity(
  liveHand: MediaPipeLandmark[],
  refFrame42: number[][] | undefined,
  twoHand: boolean
): number {
  if (!liveHand || liveHand.length < 21 || !refFrame42 || refFrame42.length < 42) return 0

  const scoreAgainstOffset = (offset: number): number => {
    let refEnergy = 0
    for (let i = 0; i < 21; i++) {
      refEnergy += Math.abs(refFrame42[offset + i][0]) + Math.abs(refFrame42[offset + i][1])
    }
    if (refEnergy < 1e-3) return 0

    let dotSum = 0
    let count = 0
    for (const [a, b] of BONE_PAIRS) {
      const lx = liveHand[b].x - liveHand[a].x
      const ly = liveHand[b].y - liveHand[a].y
      const lz = (liveHand[b].z || 0) - (liveHand[a].z || 0)
      const lLen = Math.hypot(lx, ly, lz)

      const rx = refFrame42[offset + b][0] - refFrame42[offset + a][0]
      const ry = refFrame42[offset + b][1] - refFrame42[offset + a][1]
      const rz = (refFrame42[offset + b][2] || 0) - (refFrame42[offset + a][2] || 0)
      const rLen = Math.hypot(rx, ry, rz)

      if (lLen > 1e-4 && rLen > 1e-4) {
        const cos = (lx * rx + ly * ry + lz * rz) / (lLen * rLen)
        dotSum += cos
        count++
      }
    }
    if (count === 0) return 0
    const avgCos = dotSum / count
    return Math.round(Math.max(0, Math.min(100, ((avgCos + 0.25) / 1.15) * 100)))
  }

  const s0 = scoreAgainstOffset(0)
  const s1 = twoHand ? scoreAgainstOffset(21) : 0
  return Math.max(s0, s1)
}

/**
 * Standalone interactive 2,731-gloss 3D ASL Sign Dictionary workspace.
 */
export function DictionaryView({ initialGloss, onTrySigningLive }: DictionaryViewProps) {
  const [vocab, setVocab] = React.useState<VocabListItem[]>([])
  const [vocabLoading, setVocabLoading] = React.useState(true)
  const [query, setQuery] = React.useState("")
  const [filter, setFilter] = React.useState<CategoryFilter>("all")
  const [visibleLimit, setVisibleLimit] = React.useState(90)

  const [selectedGloss, setSelectedGloss] = React.useState<string>(
    initialGloss?.trim().toUpperCase() || "FRIEND"
  )
  const [prevInitialGloss, setPrevInitialGloss] = React.useState(initialGloss)
  const [signDetail, setSignDetail] = React.useState<DictionarySignDetail | null>(null)
  const [fetchedGloss, setFetchedGloss] = React.useState<string | null>(null)
  const [showRefVideo, setShowRefVideo] = React.useState(false)
  const [prevSelectedGloss, setPrevSelectedGloss] = React.useState(selectedGloss)

  if (initialGloss !== prevInitialGloss) {
    setPrevInitialGloss(initialGloss)
    if (initialGloss && initialGloss.trim()) {
      setSelectedGloss(initialGloss.trim().toUpperCase())
    }
  }
  if (selectedGloss !== prevSelectedGloss) {
    setPrevSelectedGloss(selectedGloss)
    setShowRefVideo(false)
  }
  const isLoadingSign = Boolean(selectedGloss) && fetchedGloss !== selectedGloss

  // Live practice mini-cam state
  const [camActive, setCamActive] = React.useState(false)
  const [formMatchPct, setFormMatchPct] = React.useState<number | null>(null)
  const videoRef = React.useRef<HTMLVideoElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const streamRef = React.useRef<MediaStream | null>(null)
  const rafRef = React.useRef<number | null>(null)
  const listenerRef = React.useRef<((results: MediaPipeHandsResults) => void) | null>(null)
  const activeFrameRef = React.useRef(0)
  const signDetailRef = React.useRef<DictionarySignDetail | null>(null)

  React.useEffect(() => {
    signDetailRef.current = signDetail
  }, [signDetail])

  // Fetch complete 2,731-gloss metadata list once for instant search
  React.useEffect(() => {
    let active = true
    fetch("/api/vocabulary?limit=2731")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!active || !Array.isArray(data?.items)) return
        setVocab(data.items as VocabListItem[])
      })
      .catch(() => {})
      .finally(() => {
        if (active) setVocabLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  // Fetch 32-frame 3D exemplar & anatomical steps for selected gloss
  React.useEffect(() => {
    if (!selectedGloss) return
    let active = true
    fetch(`/api/dictionary/sign/${encodeURIComponent(selectedGloss)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!active || !data) return
        setSignDetail(data as DictionarySignDetail)
      })
      .catch(() => {})
      .finally(() => {
        if (active) setFetchedGloss(selectedGloss)
      })
    return () => {
      active = false
    }
  }, [selectedGloss])

  const filteredItems = React.useMemo(() => {
    const q = query.trim().toUpperCase()
    const base = vocab.filter((item) => {
      if (q && !item.gloss.includes(q)) return false
      if (filter === "one_hand") return !item.two_hand
      if (filter === "two_hand") return item.two_hand
      if (filter === "core") return item.tier === "High Precision" || item.tier === "Verified Demo"
      return true
    })

    if (!q && filter === "all") {
      const starterOrder = new Map(STARTER_GLOSSES.map((g, idx) => [g, idx]))
      const starters = base
        .filter((i) => starterOrder.has(i.gloss))
        .sort((a, b) => (starterOrder.get(a.gloss) ?? 99) - (starterOrder.get(b.gloss) ?? 99))
      const rest = base.filter((i) => !starterOrder.has(i.gloss))
      return [...starters, ...rest]
    }
    return base
  }, [vocab, query, filter])

  const visibleItems = React.useMemo(
    () => filteredItems.slice(0, visibleLimit),
    [filteredItems, visibleLimit]
  )

  const handleListScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 220 && visibleLimit < filteredItems.length) {
      setVisibleLimit((prev) => Math.min(filteredItems.length, prev + 90))
    }
  }

  // Practice mini-webcam lifecycle
  const stopPracticeCam = React.useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (listenerRef.current) {
      clearMediaPipeHandsListener(listenerRef.current)
      listenerRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCamActive(false)
    setFormMatchPct(null)
  }, [])

  React.useEffect(() => {
    return () => {
      stopPracticeCam()
    }
  }, [stopPracticeCam])

  const startPracticeCam = async () => {
    if (camActive) {
      stopPracticeCam()
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 360 }, height: { ideal: 270 }, facingMode: "user" },
        audio: false
      })
      streamRef.current = stream
      setCamActive(true)

      await getSharedMediaPipeHands({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      })

      const handleResults = (results: MediaPipeHandsResults) => {
        const canvas = canvasRef.current
        const video = videoRef.current
        const ctx = canvas?.getContext("2d")
        if (canvas && ctx) {
          if (video && video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
          }
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          if (results.multiHandLandmarks) {
            ctx.fillStyle = "#F59E0B"
            for (const hand of results.multiHandLandmarks) {
              for (const pt of hand) {
                ctx.beginPath()
                ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, Math.PI * 2)
                ctx.fill()
              }
            }
          }
        }

        const firstHand = results.multiHandLandmarks?.[0]
        const curDetail = signDetailRef.current
        if (firstHand && curDetail?.frames?.length) {
          const refFrame = curDetail.frames[Math.min(curDetail.frames.length - 1, activeFrameRef.current)]
          setFormMatchPct(computeHandFormSimilarity(firstHand, refFrame, curDetail.two_hand))
        } else {
          setFormMatchPct(null)
        }
      }
      listenerRef.current = handleResults
      setMediaPipeHandsListener(handleResults)

      const attachAndLoop = () => {
        const video = videoRef.current
        if (video) {
          video.srcObject = stream
          void video.play().catch(() => {})
        }
        let processing = false
        const pump = async () => {
          if (!streamRef.current) return
          const v = videoRef.current
          if (v && v.readyState >= 2 && !processing) {
            processing = true
            try {
              await sendVideoFrameToSharedHands(v)
            } finally {
              processing = false
            }
          }
          rafRef.current = requestAnimationFrame(() => void pump())
        }
        rafRef.current = requestAnimationFrame(() => void pump())
      }

      window.setTimeout(attachAndLoop, 40)
    } catch {
      stopPracticeCam()
    }
  }

  const activeVocabIndex = React.useMemo(() => {
    const idx = vocab.findIndex((v) => v.gloss === selectedGloss)
    return idx >= 0 ? `#${String(idx + 1).padStart(4, "0")}` : undefined
  }, [vocab, selectedGloss])

  const steps = signDetail?.steps?.length
    ? signDetail.steps
    : [
        "Match the finger posture shown in the 3D hand rig.",
        "Follow the glowing motion path across center chest space.",
        "Hold briefly at the final target posture."
      ]

  const stepLabels = ["1. Handshape", "2. Movement", "3. Hold & Finish"]

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[276px_minmax(0,1fr)_308px] gap-4 min-h-[calc(100vh-118px)]">
      {/* Left Gloss Browser Sidebar */}
      <aside className="flex flex-col rounded-[22px] border border-white/10 bg-[#181423]/92 backdrop-blur-xl overflow-hidden max-h-[calc(100vh-120px)]">
        <div className="p-3.5 border-b border-white/8 space-y-3">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[#8F88A0] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setVisibleLimit(90)
              }}
              placeholder="Search 2,731 ASL glosses..."
              aria-label="Search ASL glosses"
              className="w-full pl-8 pr-8 py-2 rounded-xl bg-[#110E19] border border-white/10 text-xs font-medium text-[#FAF6F0] placeholder:text-[#726B84] focus:outline-none focus:border-amber-400/50"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery("")}
                aria-label="Clear search"
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#8F88A0] hover:text-white cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-1">
            {(
              [
                { id: "all", label: "All" },
                { id: "two_hand", label: "2-Hand" },
                { id: "one_hand", label: "1-Hand" },
                { id: "core", label: "Core" }
              ] as const
            ).map((pill) => (
              <button
                key={pill.id}
                type="button"
                onClick={() => {
                  setFilter(pill.id)
                  setVisibleLimit(90)
                }}
                className={`px-2.5 py-1 rounded-full text-[11px] font-semibold transition-colors whitespace-nowrap cursor-pointer ${
                  filter === pill.id
                    ? "bg-amber-400/18 border border-amber-400/40 text-amber-200"
                    : "bg-white/5 border border-white/8 text-[#9E97B0] hover:text-white"
                }`}
              >
                {pill.label}
              </button>
            ))}
          </div>
        </div>

        <div
          onScroll={handleListScroll}
          className="flex-1 overflow-y-auto p-2.5 space-y-1.5"
        >
          {vocabLoading ? (
            <div className="py-8 text-center text-xs text-[#8E87A0]">
              Loading 2,731 ASL glosses...
            </div>
          ) : visibleItems.length === 0 ? (
            <div className="py-8 text-center text-xs text-[#8E87A0]">
              No matching signs for &quot;{query}&quot;.
            </div>
          ) : (
            visibleItems.map((item) => {
              const isSelected = item.gloss === selectedGloss
              return (
                <button
                  key={item.gloss}
                  type="button"
                  onClick={() => setSelectedGloss(item.gloss)}
                  className={`w-full flex items-center justify-between gap-2 px-3.5 py-2.5 rounded-xl border text-left transition-all cursor-pointer ${
                    isSelected
                      ? "bg-amber-500/14 border-amber-400/50 text-amber-200 shadow-[0_0_24px_-10px_rgba(245,158,11,0.55)]"
                      : "bg-white/[0.025] border-white/6 text-[#EBE6F4] hover:bg-white/[0.06] hover:border-white/15"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="text-sm font-bold tracking-wide truncate">{item.gloss}</div>
                    {item.handshape && (
                      <div className="text-[10px] text-[#958EA8] truncate">{item.handshape}</div>
                    )}
                  </div>
                  <span
                    className={`shrink-0 px-2 py-0.5 rounded-full text-[10px] font-medium ${
                      isSelected
                        ? "bg-amber-400/20 text-amber-200"
                        : "bg-white/6 text-[#9B94AC]"
                    }`}
                  >
                    {item.two_hand ? "2-Hand" : "1-Hand"}
                  </span>
                </button>
              )
            })
          )}
        </div>
      </aside>

      {/* Center Interactive 3D Stage */}
      <section className="flex flex-col min-h-[500px] lg:h-[calc(100vh-120px)]">
        <Dictionary3DStage
          gloss={selectedGloss}
          glossIndexLabel={activeVocabIndex ? `Gloss ${activeVocabIndex}` : undefined}
          twoHand={signDetail?.two_hand ?? false}
          frames={signDetail?.frames ?? []}
          handshape={signDetail?.handshape}
          isLoading={isLoadingSign}
          onFrameChange={(idx) => {
            activeFrameRef.current = idx
          }}
        />
      </section>

      {/* Right Instruction Steps & Live Practice Panel */}
      <aside className="flex flex-col gap-3.5">
        <div className="rounded-[22px] border border-white/10 bg-[#181423]/92 backdrop-blur-xl p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-bold text-[#FAF6F0] truncate">
              How to Sign: {selectedGloss}
            </h3>
            {signDetail?.tier === "High Precision" && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/12 border border-emerald-400/25 text-emerald-300 shrink-0">
                <Sparkles className="w-2.5 h-2.5" /> High Precision
              </span>
            )}
          </div>

          <div className="space-y-2.5">
            {steps.slice(0, 3).map((detail, idx) => (
              <div
                key={stepLabels[idx]}
                className="rounded-xl border border-white/8 bg-[#120F1B]/80 p-3 space-y-1"
              >
                <div className="text-xs font-bold text-amber-300">{stepLabels[idx]}</div>
                <p className="text-xs leading-relaxed text-[#C4BED3]">{detail}</p>
              </div>
            ))}
          </div>

          {signDetail?.video_url && (
            <div className="pt-1">
              <button
                type="button"
                onClick={() => setShowRefVideo((v) => !v)}
                className="inline-flex items-center gap-1.5 text-[11px] font-medium text-[#ACA5BC] hover:text-amber-300 transition-colors cursor-pointer"
              >
                <Film className="w-3.5 h-3.5" />
                <span>{showRefVideo ? "Hide human video clip" : "View ASL Citizen clip"}</span>
              </button>
              {showRefVideo && (
                <video
                  src={signDetail.video_url}
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="mt-2 w-full rounded-xl border border-white/10 max-h-36 object-cover bg-black"
                />
              )}
            </div>
          )}
        </div>

        {/* Live Mirror Form Practice Card */}
        <div className="flex-1 flex flex-col justify-between rounded-[22px] border border-white/10 bg-[#181423]/92 backdrop-blur-xl p-4 gap-3">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-[#FAF6F0]">Live Mirror Practice</span>
              <button
                type="button"
                onClick={() => void startPracticeCam()}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold border transition-colors cursor-pointer ${
                  camActive
                    ? "bg-rose-500/15 border-rose-400/35 text-rose-200"
                    : "bg-white/6 border-white/12 text-[#D5CFE3] hover:bg-white/12"
                }`}
              >
                {camActive ? <CameraOff className="w-3 h-3" /> : <Camera className="w-3 h-3" />}
                <span>{camActive ? "Stop Cam" : "Start Cam"}</span>
              </button>
            </div>

            <div className="relative aspect-[4/3] w-full rounded-xl overflow-hidden border border-white/10 bg-[#100D17] flex items-center justify-center">
              {camActive ? (
                <>
                  <video
                    ref={videoRef}
                    muted
                    playsInline
                    className="w-full h-full object-cover scale-x-[-1]"
                  />
                  <canvas
                    ref={canvasRef}
                    className="pointer-events-none absolute inset-0 w-full h-full scale-x-[-1]"
                  />
                </>
              ) : (
                <div className="px-4 text-center space-y-1.5">
                  <Camera className="w-6 h-6 text-[#6D6580] mx-auto" />
                  <p className="text-[11px] text-[#8E87A0]">
                    Enable webcam to compare your handshape against the 3D model
                  </p>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between text-xs pt-0.5">
              <span className="text-[#9E97B0]">Form Match:</span>
              <span
                className={`font-bold tabular-nums ${
                  formMatchPct !== null && formMatchPct >= 75
                    ? "text-emerald-400"
                    : formMatchPct !== null
                      ? "text-amber-300"
                      : "text-[#766E88]"
                }`}
              >
                {formMatchPct !== null ? `${formMatchPct}%` : camActive ? "Show hand" : "Camera off"}
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              stopPracticeCam()
              onTrySigningLive?.(selectedGloss)
            }}
            className="w-full inline-flex items-center justify-center gap-2 rounded-full bg-gradient-to-b from-[#FBBF24] to-[#F59E0B] px-4 py-2.5 text-xs font-bold text-[#1A130D] shadow-[0_8px_24px_-6px_rgba(245,158,11,0.5)] transition hover:brightness-105 cursor-pointer"
          >
            <span className="truncate">Try Signing {selectedGloss} Live</span>
            <ArrowRight className="w-3.5 h-3.5 shrink-0" />
          </button>
        </div>
      </aside>
    </div>
  )
}
