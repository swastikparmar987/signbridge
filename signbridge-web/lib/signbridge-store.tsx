"use client"

import * as React from "react"

export type RecognitionMode = "Continuous" | "Word" | "Fingerspelling"
export type StudioWindowId = "custom-learner" | "settings" | "video" | "dictionary"

export const DEFAULT_DISPLAY_NAME = "Learner"

export function createFreshUserId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `sb_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`
  }
  const timePart = Date.now().toString(36)
  const randPart = Math.random().toString(36).slice(2, 8)
  return `sb_${timePart}${randPart}`.slice(0, 24)
}

export function getDisplayInitials(displayName?: string): string {
  const cleaned = (displayName || "").trim()
  if (!cleaned || cleaned === DEFAULT_DISPLAY_NAME) return "SB"
  const parts = cleaned.split(/\s+/).filter(Boolean)
  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase()
  }
  return cleaned.slice(0, 2).toUpperCase()
}

export function getTimeGreetingForName(displayName?: string): string {
  const cleaned = (displayName || "").trim()
  const suffix = cleaned && cleaned !== DEFAULT_DISPLAY_NAME ? `, ${cleaned}` : ""
  if (typeof window === "undefined") {
    return suffix ? `Welcome back${suffix}` : "Welcome to SignBridge"
  }
  const hour = new Date().getHours()
  if (hour < 12) return `Good Morning${suffix}`
  if (hour < 17) return `Good Afternoon${suffix}`
  return `Good Evening${suffix}`
}

export interface SignSession {
  id: string
  title: string
  mode: RecognitionMode
  text: string
  glosses: string[]
  confidence?: number
  createdAt: string
}

export interface NotificationItem {
  id: string
  title: string
  detail: string
  time: string
  unread: boolean
  kind: "model" | "session" | "system"
}

export type KokoroVoiceId = "af_sarah" | "af_bella" | "af_nicole" | "am_adam"

export interface SignBridgePreferences {
  modelVariant: "production_exp13" | "demo_50"
  ttsVoice: KokoroVoiceId
  autoSpeakTTS: boolean
  confidenceCutoff: number
  telemetryHud: boolean
  personalizationEnabled: boolean
  userId: string
  displayName: string
}

export interface PipelineServiceState {
  groq: { available: boolean; model: string; engine: string } | null
  qwen: { available: boolean; model: string; device: string; engine: string } | null
  tts: { available: boolean; engine: string; voice: string; status: string } | null
}

export interface CustomSignItem {
  gloss: string
  total_samples: number
  active_samples: number
  has_prototype: boolean
  prototype_sample_count: number
  last_updated: string | null
}

export interface ModelHealthState {
  status: "checking" | "online" | "offline"
  device: string | null
  hardwareDevice: string | null
  numClasses: number | null
  exp13Top1: number | null
  exp13Top5: number | null
  productionTop1: number | null
  productionTop5: number | null
  activeExperiment: string | null
  fingerspellingReady: boolean
  lastLatencyMs: number | null
  lastPredictionConfidence: number | null
}

interface PersistedState {
  sessions: SignSession[]
  readNotificationIds: string[]
  lastSessionNumber: number
  preferences: SignBridgePreferences
}

const STORAGE_KEY = "signbridge_dashboard_state_v1"

const DEFAULT_PREFERENCES: SignBridgePreferences = {
  modelVariant: "production_exp13",
  ttsVoice: "af_sarah",
  autoSpeakTTS: false,
  confidenceCutoff: 45,
  telemetryHud: true,
  personalizationEnabled: false,
  userId: "",
  displayName: DEFAULT_DISPLAY_NAME
}

const DEFAULT_PERSISTED: PersistedState = {
  sessions: [],
  readNotificationIds: [],
  lastSessionNumber: 0,
  preferences: DEFAULT_PREFERENCES
}

function getLocalDateKey(date: Date = new Date()): string {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, "0")
  const d = String(date.getDate()).padStart(2, "0")
  return `${y}-${m}-${d}`
}

export function formatSessionTimestamp(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ""
  const now = new Date()
  const todayKey = getLocalDateKey(now)
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  const yesterdayKey = getLocalDateKey(yesterday)
  const targetKey = getLocalDateKey(date)

  const timeStr = date.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit"
  })

  if (targetKey === todayKey) return `Today, ${timeStr}`
  if (targetKey === yesterdayKey) return `Yesterday, ${timeStr}`
  return `${date.toLocaleDateString([], { month: "short", day: "numeric" })}, ${timeStr}`
}

interface SignBridgeContextValue {
  sessions: SignSession[]
  modelHealth: ModelHealthState
  pipelineStatus: PipelineServiceState
  customSigns: CustomSignItem[]
  personalizationEnabled: boolean
  preferences: SignBridgePreferences
  notifications: NotificationItem[]
  unreadCount: number
  refreshHealth: () => Promise<void>
  refreshPipelineStatus: () => Promise<void>
  refreshPersonalization: () => Promise<void>
  setPersonalizationEnabled: (enabled: boolean) => Promise<void>
  updatePreferences: (patch: Partial<SignBridgePreferences>) => void
  resetPreferences: () => void
  startFreshProfile: (eraseRemoteCurrent?: boolean) => Promise<void>
  recordSession: (input: {
    mode: RecognitionMode
    text: string
    glosses: string[]
    confidence?: number
    title?: string
  }) => SignSession | null
  reportInferenceTelemetry: (latencyMs: number | null, confidence?: number | null) => void
  markAllNotificationsRead: () => void
  clearSessions: () => void
}

const SignBridgeContext = React.createContext<SignBridgeContextValue | null>(null)

export function SignBridgeStoreProvider({ children }: { children: React.ReactNode }) {
  const [persisted, setPersisted] = React.useState<PersistedState>(DEFAULT_PERSISTED)
  const [hydrated, setHydrated] = React.useState(false)
  const [pipelineStatus, setPipelineStatus] = React.useState<PipelineServiceState>({
    groq: null,
    qwen: null,
    tts: null
  })
  const [customSigns, setCustomSigns] = React.useState<CustomSignItem[]>([])
  const [backendPersonalizationEnabled, setBackendPersonalizationEnabled] = React.useState(false)
  const [modelHealth, setModelHealth] = React.useState<ModelHealthState>({
    status: "checking",
    device: null,
    hardwareDevice: null,
    numClasses: null,
    exp13Top1: null,
    exp13Top5: null,
    productionTop1: null,
    productionTop5: null,
    activeExperiment: null,
    fingerspellingReady: false,
    lastLatencyMs: null,
    lastPredictionConfidence: null
  })

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const params = new URLSearchParams(window.location.search)
        const forceFresh =
          params.get("fresh") === "1" || params.get("new_user") === "1"

        if (forceFresh) {
          params.delete("fresh")
          params.delete("new_user")
          const nextSearch = params.toString()
          const cleanedUrl = `${window.location.pathname}${
            nextSearch ? `?${nextSearch}` : ""
          }${window.location.hash}`
          window.history.replaceState(window.history.state, "", cleanedUrl)
          setCustomSigns([])
          setBackendPersonalizationEnabled(false)
          setPersisted({
            sessions: [],
            readNotificationIds: [],
            lastSessionNumber: 0,
            preferences: {
              ...DEFAULT_PREFERENCES,
              userId: createFreshUserId()
            }
          })
          return
        }

        const raw = window.localStorage.getItem(STORAGE_KEY)
        if (raw) {
          const parsed = JSON.parse(raw) as Partial<PersistedState>
          const storedUid =
            typeof parsed.preferences?.userId === "string" &&
            parsed.preferences.userId.trim().length > 0
              ? parsed.preferences.userId.trim()
              : createFreshUserId()
          const storedName =
            typeof parsed.preferences?.displayName === "string" &&
            parsed.preferences.displayName.trim().length > 0
              ? parsed.preferences.displayName.trim()
              : DEFAULT_DISPLAY_NAME

          setPersisted({
            sessions: Array.isArray(parsed.sessions) ? parsed.sessions : [],
            readNotificationIds: Array.isArray(parsed.readNotificationIds)
              ? parsed.readNotificationIds
              : [],
            lastSessionNumber:
              typeof parsed.lastSessionNumber === "number"
                ? parsed.lastSessionNumber
                : 0,
            preferences: {
              ...DEFAULT_PREFERENCES,
              ...(parsed.preferences ?? {}),
              userId: storedUid,
              displayName: storedName
            }
          })
        } else {
          setPersisted({
            ...DEFAULT_PERSISTED,
            preferences: {
              ...DEFAULT_PREFERENCES,
              userId: createFreshUserId()
            }
          })
        }
      } catch {
        setPersisted({
          ...DEFAULT_PERSISTED,
          preferences: {
            ...DEFAULT_PREFERENCES,
            userId: createFreshUserId()
          }
        })
      } finally {
        setHydrated(true)
      }
    }, 0)
    return () => window.clearTimeout(timer)
  }, [])

  React.useEffect(() => {
    if (!hydrated) return
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted))
    } catch {
      // Ignore storage quota errors
    }
  }, [persisted, hydrated])

  const refreshHealth = React.useCallback(async () => {
    try {
      const res = await fetch("/api/health", { cache: "no-store" })
      if (!res.ok) {
        setModelHealth((prev) => ({ ...prev, status: "offline" }))
        return
      }
      const data = await res.json()
      const exp13 = data.evaluation_metrics?.exp13
      const prod = data.evaluation_metrics?.production

      setModelHealth((prev) => ({
        ...prev,
        status: data.status === "healthy" ? "online" : "offline",
        device: typeof data.device === "string" ? data.device : null,
        hardwareDevice:
          typeof data.hardware_device === "string" && data.hardware_device.trim().length > 0
            ? data.hardware_device
            : null,
        numClasses: typeof data.num_classes === "number" ? data.num_classes : null,
        exp13Top1:
          typeof exp13?.top1_accuracy === "number"
            ? exp13.top1_accuracy
            : typeof prod?.top1_accuracy === "number"
              ? prod.top1_accuracy
              : null,
        exp13Top5:
          typeof exp13?.top5_accuracy === "number"
            ? exp13.top5_accuracy
            : typeof prod?.top5_accuracy === "number"
              ? prod.top5_accuracy
              : null,
        productionTop1: typeof prod?.top1_accuracy === "number" ? prod.top1_accuracy : null,
        productionTop5: typeof prod?.top5_accuracy === "number" ? prod.top5_accuracy : null,
        activeExperiment:
          typeof prod?.experiment === "string"
            ? prod.experiment
            : typeof exp13?.experiment === "string"
              ? "EXP13 Held-Out Test"
              : null,
        fingerspellingReady: Boolean(data.fingerspelling_ready)
      }))
    } catch {
      setModelHealth((prev) => ({ ...prev, status: "offline" }))
    }
  }, [])

  const refreshPipelineStatus = React.useCallback(async () => {
    try {
      const res = await fetch("/api/pipeline/status", { cache: "no-store" })
      if (!res.ok) return
      const data = await res.json()
      setPipelineStatus({
        groq: data.groq ?? null,
        qwen: data.qwen ?? null,
        tts: data.tts ?? null
      })
    } catch {
      // Ignore transient network errors
    }
  }, [])

  const refreshPersonalization = React.useCallback(async () => {
    if (!hydrated || !persisted.preferences.userId) return
    try {
      const uid = encodeURIComponent(persisted.preferences.userId)
      const res = await fetch(`/api/personalization/profile?user_id=${uid}`, {
        cache: "no-store"
      })
      if (!res.ok) return
      const data = await res.json()
      const signsList = Array.isArray(data.signs) ? data.signs : []
      setCustomSigns(signsList)
      const backendOn = Boolean(data.enabled)
      setBackendPersonalizationEnabled(backendOn)
      if (backendOn) {
        setPersisted((prev) =>
          prev.preferences.personalizationEnabled
            ? prev
            : { ...prev, preferences: { ...prev.preferences, personalizationEnabled: true } }
        )
      }
    } catch {
      // Ignore transient network errors
    }
  }, [hydrated, persisted.preferences.userId])

  const setPersonalizationEnabled = React.useCallback(
    async (enabled: boolean) => {
      const uid = persisted.preferences.userId
      if (!uid) return
      setPersisted((prev) => ({
        ...prev,
        preferences: {
          ...prev.preferences,
          personalizationEnabled: enabled
        }
      }))
      setBackendPersonalizationEnabled(enabled)
      await fetch(enabled ? "/api/personalization/enable" : "/api/personalization/disable", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: uid })
      }).catch(() => {})
      await refreshPersonalization()
    },
    [persisted.preferences.userId, refreshPersonalization]
  )

  React.useEffect(() => {
    const initial = window.setTimeout(() => {
      void refreshHealth()
      void refreshPipelineStatus()
      void refreshPersonalization()
    }, 0)
    const timer = window.setInterval(() => {
      void refreshHealth()
      void refreshPipelineStatus()
      void refreshPersonalization()
    }, 30000)
    return () => {
      window.clearTimeout(initial)
      window.clearInterval(timer)
    }
  }, [refreshHealth, refreshPipelineStatus, refreshPersonalization])

  const updatePreferences = React.useCallback(
    (patch: Partial<SignBridgePreferences>) => {
      setPersisted((prev) => ({
        ...prev,
        preferences: {
          ...prev.preferences,
          ...patch
        }
      }))
    },
    []
  )

  const resetPreferences = React.useCallback(() => {
    setPersisted((prev) => ({
      ...prev,
      preferences: {
        ...DEFAULT_PREFERENCES,
        userId: prev.preferences.userId || createFreshUserId(),
        displayName: prev.preferences.displayName || DEFAULT_DISPLAY_NAME
      }
    }))
  }, [])

  const startFreshProfile = React.useCallback(
    async (eraseRemoteCurrent = false) => {
      const currentUid = persisted.preferences.userId
      if (eraseRemoteCurrent && currentUid) {
        await fetch(`/api/personalization?user_id=${encodeURIComponent(currentUid)}`, {
          method: "DELETE"
        }).catch(() => {})
      }
      const nextUid = createFreshUserId()
      setCustomSigns([])
      setBackendPersonalizationEnabled(false)
      setPersisted({
        sessions: [],
        readNotificationIds: [],
        lastSessionNumber: 0,
        preferences: {
          ...DEFAULT_PREFERENCES,
          userId: nextUid
        }
      })
    },
    [persisted.preferences.userId]
  )

  const reportInferenceTelemetry = React.useCallback(
    (latencyMs: number | null, confidence?: number | null) => {
      setModelHealth((prev) => ({
        ...prev,
        lastLatencyMs:
          typeof latencyMs === "number" && Number.isFinite(latencyMs) && latencyMs >= 0
            ? Math.round(latencyMs)
            : prev.lastLatencyMs,
        lastPredictionConfidence:
          typeof confidence === "number" && Number.isFinite(confidence)
            ? Number(confidence.toFixed(1))
            : prev.lastPredictionConfidence
      }))
    },
    []
  )

  const recordSession = React.useCallback(
    (input: {
      mode: RecognitionMode
      text: string
      glosses: string[]
      confidence?: number
      title?: string
    }): SignSession | null => {
      const cleanedGlosses = input.glosses.map((g) => g.trim()).filter(Boolean)
      const cleanedText = input.text.trim()
      if (!cleanedText && cleanedGlosses.length === 0) return null

      let created: SignSession | null = null

      setPersisted((prev) => {
        const nextNum = prev.lastSessionNumber + 1
        const modeTitle =
          input.mode === "Continuous"
            ? `Continuous Signing – Session ${nextNum}`
            : input.mode === "Word"
              ? `Word Recognition – ${cleanedGlosses[0] || `Session ${nextNum}`}`
              : `Fingerspelling – "${cleanedText || `Session ${nextNum}`}"`

        const session: SignSession = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          title: input.title || modeTitle,
          mode: input.mode,
          text: cleanedText || cleanedGlosses.join(" "),
          glosses: cleanedGlosses,
          confidence: input.confidence,
          createdAt: new Date().toISOString()
        }
        created = session

        return {
          ...prev,
          lastSessionNumber: nextNum,
          sessions: [session, ...prev.sessions].slice(0, 25)
        }
      })

      return created
    },
    []
  )

  const clearSessions = React.useCallback(() => {
    setPersisted((prev) => ({
      ...prev,
      sessions: []
    }))
  }, [])

  const notifications = React.useMemo<NotificationItem[]>(() => {
    const items: NotificationItem[] = []
    if (modelHealth.status === "online") {
      items.push({
        id: "model-online",
        title: "SignBridge Model Ready",
        detail: `${modelHealth.numClasses ?? 2731} ASL classes loaded${
          modelHealth.hardwareDevice ? ` on ${modelHealth.hardwareDevice}` : ""
        }`,
        time: "Live",
        unread: !persisted.readNotificationIds.includes("model-online"),
        kind: "model"
      })
    } else if (modelHealth.status === "offline") {
      items.push({
        id: "model-offline",
        title: "Recognition Backend Offline",
        detail: "Start the FastAPI backend on port 8000 for live predictions.",
        time: "Alert",
        unread: !persisted.readNotificationIds.includes("model-offline"),
        kind: "system"
      })
    }

    if (persisted.sessions.length > 0) {
      const latest = persisted.sessions[0]
      items.push({
        id: `session-${latest.id}`,
        title: latest.title,
        detail: `"${latest.text}"`,
        time: formatSessionTimestamp(latest.createdAt),
        unread: !persisted.readNotificationIds.includes(`session-${latest.id}`),
        kind: "session"
      })
    }

    return items
  }, [modelHealth, persisted.sessions, persisted.readNotificationIds])

  const unreadCount = React.useMemo(
    () => notifications.filter((n) => n.unread).length,
    [notifications]
  )

  const markAllNotificationsRead = React.useCallback(() => {
    setPersisted((prev) => ({
      ...prev,
      readNotificationIds: Array.from(
        new Set([...prev.readNotificationIds, ...notifications.map((n) => n.id)])
      )
    }))
  }, [notifications])

  const personalizationEnabled =
    backendPersonalizationEnabled || (persisted.preferences.personalizationEnabled && customSigns.length > 0)

  const value = React.useMemo<SignBridgeContextValue>(
    () => ({
      sessions: persisted.sessions,
      modelHealth,
      pipelineStatus,
      customSigns,
      personalizationEnabled,
      preferences: persisted.preferences,
      notifications,
      unreadCount,
      refreshHealth,
      refreshPipelineStatus,
      refreshPersonalization,
      setPersonalizationEnabled,
      updatePreferences,
      resetPreferences,
      startFreshProfile,
      recordSession,
      reportInferenceTelemetry,
      markAllNotificationsRead,
      clearSessions
    }),
    [
      persisted.sessions,
      persisted.preferences,
      modelHealth,
      pipelineStatus,
      customSigns,
      personalizationEnabled,
      notifications,
      unreadCount,
      refreshHealth,
      refreshPipelineStatus,
      refreshPersonalization,
      setPersonalizationEnabled,
      updatePreferences,
      resetPreferences,
      startFreshProfile,
      recordSession,
      reportInferenceTelemetry,
      markAllNotificationsRead,
      clearSessions
    ]
  )

  return <SignBridgeContext.Provider value={value}>{children}</SignBridgeContext.Provider>
}

export function useSignBridgeStore(): SignBridgeContextValue {
  const ctx = React.useContext(SignBridgeContext)
  if (!ctx) {
    throw new Error("useSignBridgeStore must be used inside SignBridgeStoreProvider")
  }
  return ctx
}
