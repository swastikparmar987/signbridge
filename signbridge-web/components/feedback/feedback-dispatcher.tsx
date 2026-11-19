"use client"

import * as React from "react"
import { onFeedback } from "@/lib/feedback/bus"
import { vibrateFor } from "@/lib/feedback/haptics"
import { playFor, unlockSound } from "@/lib/feedback/sound"
import { useSignBridgeStore } from "@/lib/signbridge-store"

/**
 * Turns feedback events into vibration and (optional) sound, app-wide.
 * Visual feedback is handled separately by <FeedbackLayer />.
 */
export function FeedbackDispatcher() {
  const { preferences } = useSignBridgeStore()
  const prefsRef = React.useRef(preferences)
  React.useEffect(() => {
    prefsRef.current = preferences
  }, [preferences])

  React.useEffect(() => {
    return onFeedback((event) => {
      if (event.type === "captureProgress") return
      const p = prefsRef.current
      if (p.haptics) vibrateFor(event)
      if (p.soundEffects) playFor(event)
    })
  }, [])

  // Browsers only allow audio after a user gesture; unlock on the first one.
  React.useEffect(() => {
    const unlock = () => {
      if (prefsRef.current.soundEffects) unlockSound()
    }
    window.addEventListener("pointerdown", unlock, { passive: true })
    window.addEventListener("keydown", unlock)
    return () => {
      window.removeEventListener("pointerdown", unlock)
      window.removeEventListener("keydown", unlock)
    }
  }, [])

  return null
}
