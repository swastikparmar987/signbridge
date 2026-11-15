"use client"

export interface MediaPipeLandmark {
  x: number
  y: number
  z?: number
}

export interface MediaPipeHandsResults {
  multiHandLandmarks?: MediaPipeLandmark[][]
  multiHandedness?: Array<{ label?: string; score?: number }>
}

interface MediaPipeHandsOptions {
  maxNumHands?: number
  modelComplexity?: number
  minDetectionConfidence?: number
  minTrackingConfidence?: number
}

interface MediaPipeHandsInstance {
  setOptions: (options: MediaPipeHandsOptions) => void
  onResults: (callback: (results: MediaPipeHandsResults) => void) => void
  send: (input: { image: HTMLVideoElement | HTMLCanvasElement }) => Promise<void>
  initialize?: () => Promise<void>
}

interface MediaPipeHandsConstructor {
  new (config: { locateFile: (file: string) => string }): MediaPipeHandsInstance
}

const LOCAL_HANDS_SCRIPT_ID = "signbridge-local-mediapipe-hands"
const LOCAL_HANDS_BASE = "/mediapipe/hands"

let scriptLoadPromise: Promise<MediaPipeHandsConstructor> | null = null
let handsSingletonPromise: Promise<MediaPipeHandsInstance> | null = null
let activeListener: ((results: MediaPipeHandsResults) => void) | null = null
let isSendingFrame = false
let lastOptionsKey = ""

function getWindowHands(): MediaPipeHandsConstructor | undefined {
  if (typeof window === "undefined") return undefined
  return (window as unknown as { Hands?: MediaPipeHandsConstructor }).Hands
}

function loadLocalMediaPipeScript(): Promise<MediaPipeHandsConstructor> {
  const existingCtor = getWindowHands()
  if (existingCtor) {
    return Promise.resolve(existingCtor)
  }

  if (scriptLoadPromise) {
    return scriptLoadPromise
  }

  scriptLoadPromise = new Promise<MediaPipeHandsConstructor>((resolve, reject) => {
    if (typeof window === "undefined" || typeof document === "undefined") {
      reject(new Error("MediaPipe Hands requires a browser environment"))
      return
    }

    let script = document.getElementById(LOCAL_HANDS_SCRIPT_ID) as HTMLScriptElement | null
    const deadline = Date.now() + 12000

    const pollReady = () => {
      const ctor = getWindowHands()
      if (ctor) {
        resolve(ctor)
        return
      }
      if (Date.now() > deadline) {
        scriptLoadPromise = null
        reject(new Error("Timed out loading local MediaPipe Hands bundle"))
        return
      }
      setTimeout(pollReady, 50)
    }

    if (!script) {
      script = document.createElement("script")
      script.id = LOCAL_HANDS_SCRIPT_ID
      script.src = `${LOCAL_HANDS_BASE}/hands.js`
      script.async = true
      script.onload = pollReady
      script.onerror = () => {
        scriptLoadPromise = null
        reject(new Error("Failed to load /mediapipe/hands/hands.js"))
      }
      document.head.appendChild(script)
    } else {
      pollReady()
    }
  })

  return scriptLoadPromise
}

export async function getSharedMediaPipeHands(
  options: MediaPipeHandsOptions = {}
): Promise<MediaPipeHandsInstance> {
  const merged: Required<MediaPipeHandsOptions> = {
    maxNumHands: options.maxNumHands ?? 2,
    modelComplexity: options.modelComplexity ?? 1,
    minDetectionConfidence: options.minDetectionConfidence ?? 0.5,
    minTrackingConfidence: options.minTrackingConfidence ?? 0.5
  }

  if (!handsSingletonPromise) {
    handsSingletonPromise = (async () => {
      const HandsCtor = await loadLocalMediaPipeScript()
      const instance = new HandsCtor({
        locateFile: (file: string) => `${LOCAL_HANDS_BASE}/${file}`
      })
      instance.setOptions(merged)
      lastOptionsKey = JSON.stringify(merged)
      instance.onResults((results: MediaPipeHandsResults) => {
        if (activeListener) {
          activeListener(results)
        }
      })
      if (typeof instance.initialize === "function") {
        await instance.initialize()
      }
      return instance
    })().catch((err) => {
      handsSingletonPromise = null
      throw err
    })
  }

  const instance = await handsSingletonPromise
  const optionsKey = JSON.stringify(merged)
  if (optionsKey !== lastOptionsKey) {
    instance.setOptions(merged)
    lastOptionsKey = optionsKey
  }
  return instance
}

export function setMediaPipeHandsListener(
  listener: ((results: MediaPipeHandsResults) => void) | null
): void {
  activeListener = listener
}

export async function sendVideoFrameToSharedHands(
  video: HTMLVideoElement,
  options?: MediaPipeHandsOptions
): Promise<boolean> {
  if (
    isSendingFrame ||
    !video ||
    video.readyState < 2 ||
    video.videoWidth <= 0 ||
    video.videoHeight <= 0
  ) {
    return false
  }

  isSendingFrame = true
  try {
    const hands = await getSharedMediaPipeHands(options)
    await hands.send({ image: video })
    return true
  } finally {
    isSendingFrame = false
  }
}

export function clearMediaPipeHandsListener(
  listener: (results: MediaPipeHandsResults) => void
): void {
  if (activeListener === listener) {
    activeListener = null
  }
}

