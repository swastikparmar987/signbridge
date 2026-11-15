import * as THREE from "three"
import type { FingerAngles, HandPoseSpec } from "@/components/studio/hand-3d-rig"

export type HandSide = "right" | "left"

export interface DictionarySequenceAnalysis {
  primarySlot: 0 | 1
  slot0Side: HandSide
  slot1Side: HandSide
  twoHand: boolean
  center: [number, number, number]
  palmSpan: number
  wristTrail0: [number, number, number][]
  wristTrail1: [number, number, number][]
}

export interface ExtractedHandPose {
  active: boolean
  side: HandSide
  worldPos: THREE.Vector3
  pivotQuat: THREE.Quaternion
  pose: HandPoseSpec
}

// Precomputed orthonormal rest bases B_rest = [c0 (pinky->index), c1 (wrist->middle), c2 (c0 x c1)]
// from the GLB rig bones after alignGroup Euler(0, -PI/2, PI), inverted (B_rest^T).
const RIGHT_REST_INV = new THREE.Matrix4().set(
  0.993137, -0.015486, -0.115926, 0,
  0.018822, 0.999438, 0.027734, 0,
  0.115432, -0.029726, 0.992871, 0,
  0, 0, 0, 1
)

const LEFT_REST_INV = new THREE.Matrix4().set(
  0.992934, -0.015434, 0.117661, 0,
  0.018822, 0.999438, -0.027734, 0,
  -0.117167, 0.029753, 0.992667, 0,
  0, 0, 0, 1
)

const NEUTRAL_FINGER: FingerAngles = { spread: 0, mcp: 0.08, pip: 0.1, dip: 0.06 }

const NEUTRAL_POSE: HandPoseSpec = {
  rotation: [0, 0, 0],
  thumb: { spread: 0.28, mcp: 0.2, pip: 0.14, dip: 0.08 },
  index: NEUTRAL_FINGER,
  middle: NEUTRAL_FINGER,
  ring: NEUTRAL_FINGER,
  pinky: NEUTRAL_FINGER
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

function isHandActive(hand21: number[][] | null | undefined): boolean {
  if (!hand21 || hand21.length < 21) return false
  let energy = 0
  for (let i = 0; i < 21; i++) {
    const p = hand21[i]
    energy += Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2])
  }
  return energy > 1e-3
}

function toWorldVec(pt: number[]): THREE.Vector3 {
  return new THREE.Vector3(pt[0] - 0.5, -(pt[1] - 0.5), -(pt[2] || 0))
}

function segmentAngle(a: THREE.Vector3, b: THREE.Vector3): number {
  const la = a.length()
  const lb = b.length()
  if (la < 1e-5 || lb < 1e-5) return 0
  const cos = clamp(a.dot(b) / (la * lb), -1, 1)
  return Math.acos(cos)
}

function computeCurlChirality(hand21: number[][]): number {
  const idx = toWorldVec(hand21[5])
  const pnk = toWorldVec(hand21[17])
  const si = idx.sub(pnk)
  const acc = new THREE.Vector3()
  const v1 = new THREE.Vector3()
  const v2 = new THREE.Vector3()
  const cr = new THREE.Vector3()
  for (const base of [5, 9, 13, 17]) {
    v1.copy(toWorldVec(hand21[base + 1])).sub(toWorldVec(hand21[base]))
    v2.copy(toWorldVec(hand21[base + 3])).sub(toWorldVec(hand21[base + 1]))
    cr.crossVectors(v1, v2)
    acc.add(cr)
  }
  return acc.dot(si)
}

function extractSlotSlice(frame42: number[][], slot: 0 | 1): number[][] | null {
  if (!frame42 || frame42.length < 42) return null
  const offset = slot === 0 ? 0 : 21
  const slice = frame42.slice(offset, offset + 21)
  return isHandActive(slice) ? slice : null
}

/**
 * Inspects a 32-frame normalized sequence to determine hand chirality, centroid,
 * palm scale, and smoothed 3D wrist trails for both hands.
 */
export function analyzeDictionarySequence(
  frames: number[][][],
  twoHand: boolean
): DictionarySequenceAnalysis {
  let count0 = 0
  let count1 = 0
  let sumX0 = 0
  let sumX1 = 0
  let chir0 = 0
  let chir1 = 0
  let centerX = 0
  let centerY = 0
  let centerZ = 0
  let centerCount = 0
  const spans: number[] = []

  const rawTrail0: ([number, number, number] | null)[] = []
  const rawTrail1: ([number, number, number] | null)[] = []

  for (const frame of frames) {
    const h0 = extractSlotSlice(frame, 0)
    const h1 = extractSlotSlice(frame, 1)
    if (h0) {
      count0++
      sumX0 += h0[9][0]
      chir0 += computeCurlChirality(h0)
      const w = toWorldVec(h0[0])
      const mid = toWorldVec(h0[9])
      spans.push(w.distanceTo(mid))
      centerX += h0[9][0]
      centerY += h0[9][1]
      centerZ += h0[9][2] || 0
      centerCount++
      rawTrail0.push([h0[9][0], h0[9][1], h0[9][2] || 0])
    } else {
      rawTrail0.push(null)
    }

    if (h1) {
      count1++
      sumX1 += h1[9][0]
      chir1 += computeCurlChirality(h1)
      const w = toWorldVec(h1[0])
      const mid = toWorldVec(h1[9])
      spans.push(w.distanceTo(mid))
      if (twoHand || !h0) {
        centerX += h1[9][0]
        centerY += h1[9][1]
        centerZ += h1[9][2] || 0
        centerCount++
      }
      rawTrail1.push([h1[9][0], h1[9][1], h1[9][2] || 0])
    } else {
      rawTrail1.push(null)
    }
  }

  const primarySlot: 0 | 1 = count0 >= count1 ? 0 : 1
  const meanX0 = count0 > 0 ? sumX0 / count0 : 0.4
  const meanX1 = count1 > 0 ? sumX1 / count1 : 0.6

  let slot0Side: HandSide = meanX0 <= 0.5 ? "right" : "left"
  let slot1Side: HandSide = meanX1 > 0.5 ? "left" : "right"

  if (twoHand) {
    // Two-hand ASL signs always show one right hand and one left hand
    if (Math.abs(chir0 - chir1) > 0.002) {
      slot0Side = chir0 >= chir1 ? "right" : "left"
      slot1Side = slot0Side === "right" ? "left" : "right"
    } else {
      slot0Side = meanX0 <= meanX1 ? "right" : "left"
      slot1Side = slot0Side === "right" ? "left" : "right"
    }
  } else {
    const primaryChir = primarySlot === 0 ? chir0 : chir1
    const primaryMeanX = primarySlot === 0 ? meanX0 : meanX1
    const chosen: HandSide =
      Math.abs(primaryChir) >= 0.0012
        ? primaryChir > 0
          ? "right"
          : "left"
        : primaryMeanX <= 0.52
          ? "right"
          : "left"
    slot0Side = chosen
    slot1Side = chosen === "right" ? "left" : "right"
  }

  spans.sort((a, b) => a - b)
  const medianSpan = spans.length > 0 ? spans[Math.floor(spans.length / 2)] : 0.16
  const palmSpan = clamp(medianSpan, 0.11, 0.3)

  const avgX = centerCount > 0 ? centerX / centerCount : 0.5
  const avgY = centerCount > 0 ? centerY / centerCount : 0.55
  const avgZ = centerCount > 0 ? centerZ / centerCount : 0
  // Full palm-centroid anchor keeps every sign centered cleanly inside the 3D spotlight
  const center: [number, number, number] = [avgX, avgY, avgZ]

  const wristTrail0 = rawTrail0.filter((p): p is [number, number, number] => p !== null)
  const wristTrail1 = twoHand
    ? rawTrail1.filter((p): p is [number, number, number] => p !== null)
    : []

  return {
    primarySlot,
    slot0Side,
    slot1Side,
    twoHand,
    center,
    palmSpan,
    wristTrail0,
    wristTrail1
  }
}

function resolveSingleHandAtFrame(
  frame: number[][],
  analysis: DictionarySequenceAnalysis
): { slot0: number[][] | null; slot1: number[][] | null } {
  const s0 = extractSlotSlice(frame, 0)
  const s1 = extractSlotSlice(frame, 1)
  if (analysis.twoHand) {
    return { slot0: s0, slot1: s1 }
  }
  const primary = analysis.primarySlot === 0 ? s0 || s1 : s1 || s0
  return { slot0: primary, slot1: null }
}

function lerpHand21(a: number[][] | null, b: number[][] | null, alpha: number): number[][] | null {
  if (a && b) {
    const out: number[][] = new Array(21)
    for (let i = 0; i < 21; i++) {
      out[i] = [
        a[i][0] + (b[i][0] - a[i][0]) * alpha,
        a[i][1] + (b[i][1] - a[i][1]) * alpha,
        (a[i][2] || 0) + ((b[i][2] || 0) - (a[i][2] || 0)) * alpha
      ]
    }
    return out
  }
  return a || b || null
}

/**
 * Interpolates 3D landmarks cleanly at a fractional frame cursor without
 * ever blending valid hands toward zero-padded dropout frames.
 */
export function interpolateFrameLandmarks(
  frames: number[][][],
  fractionalFrame: number,
  analysis: DictionarySequenceAnalysis
): { slot0: number[][] | null; slot1: number[][] | null } {
  if (frames.length === 0) return { slot0: null, slot1: null }
  const maxIdx = frames.length - 1
  const clamped = clamp(fractionalFrame, 0, maxIdx)
  const i0 = Math.floor(clamped)
  const i1 = Math.min(maxIdx, i0 + 1)
  const alpha = clamped - i0

  const f0 = resolveSingleHandAtFrame(frames[i0], analysis)
  const f1 = resolveSingleHandAtFrame(frames[i1], analysis)

  return {
    slot0: lerpHand21(f0.slot0, f1.slot0, alpha),
    slot1: lerpHand21(f0.slot1, f1.slot1, alpha)
  }
}

/**
 * Converts a 21-point MediaPipe hand into a quaternion-aligned 3D pose for the GLB rig.
 */
export function landmarks21ToHandPose(
  hand21: number[][] | null,
  side: HandSide,
  analysis: DictionarySequenceAnalysis,
  worldScale: number
): ExtractedHandPose {
  if (!hand21 || hand21.length < 21) {
    return {
      active: false,
      side,
      worldPos: new THREE.Vector3(),
      pivotQuat: new THREE.Quaternion(),
      pose: NEUTRAL_POSE
    }
  }

  const pts = hand21.map(toWorldVec)
  const wrist = pts[0]
  const idxMcp = pts[5]
  const midMcp = pts[9]
  const pnkMcp = pts[17]

  // Orthonormal hand frame [c0 (pinky->index), c1 (wrist->middle), c2 (c0 x c1)]
  const c1 = new THREE.Vector3().subVectors(midMcp, wrist)
  if (c1.lengthSq() < 1e-8) c1.set(0, 1, 0)
  c1.normalize()

  const sRaw = new THREE.Vector3().subVectors(idxMcp, pnkMcp)
  const c0 = sRaw.sub(c1.clone().multiplyScalar(sRaw.dot(c1)))
  if (c0.lengthSq() < 1e-8) c0.set(1, 0, 0)
  c0.normalize()

  const c2 = new THREE.Vector3().crossVectors(c0, c1).normalize()

  const targetBasis = new THREE.Matrix4().makeBasis(c0, c1, c2)
  const restInv = side === "right" ? RIGHT_REST_INV : LEFT_REST_INV
  const rotMat = new THREE.Matrix4().multiplyMatrices(targetBasis, restInv)
  const pivotQuat = new THREE.Quaternion().setFromRotationMatrix(rotMat)

  const [cx, cy, cz] = analysis.center
  const worldPos = new THREE.Vector3(
    (hand21[9][0] - cx) * worldScale,
    -(hand21[9][1] - cy) * worldScale + 0.05,
    -((hand21[9][2] || 0) - cz) * worldScale * 0.55
  )

  const computeFinger = (mcpIdx: number, neutralSpread: number): FingerAngles => {
    const vPalm = new THREE.Vector3().subVectors(pts[mcpIdx], wrist)
    const vProx = new THREE.Vector3().subVectors(pts[mcpIdx + 1], pts[mcpIdx])
    const vMid = new THREE.Vector3().subVectors(pts[mcpIdx + 2], pts[mcpIdx + 1])
    const vDist = new THREE.Vector3().subVectors(pts[mcpIdx + 3], pts[mcpIdx + 2])

    const mcpBend = clamp(segmentAngle(vPalm, vProx) * 1.05, 0.02, 1.52)
    const pipBend = clamp(segmentAngle(vProx, vMid) * 1.08, 0.02, 1.72)
    const dipBend = clamp(segmentAngle(vMid, vDist) * 0.95, 0.02, 1.32)

    const pNorm = vProx.clone().normalize()
    const rawSpread = clamp(pNorm.dot(c0) * 0.72, -0.32, 0.32)
    const spread = clamp(rawSpread * 0.55 + neutralSpread * 0.45, -0.35, 0.35)

    return { spread, mcp: mcpBend, pip: pipBend, dip: dipBend }
  }

  const vT0 = new THREE.Vector3().subVectors(pts[1], wrist)
  const vT1 = new THREE.Vector3().subVectors(pts[2], pts[1])
  const vT2 = new THREE.Vector3().subVectors(pts[3], pts[2])
  const vT3 = new THREE.Vector3().subVectors(pts[4], pts[3])
  const thumb: FingerAngles = {
    spread: clamp(vT1.clone().normalize().dot(c0) * 0.55, 0.06, 0.56),
    mcp: clamp(segmentAngle(vT0, vT1) * 1.05, 0.05, 1.15),
    pip: clamp(segmentAngle(vT1, vT2) * 1.05, 0.04, 1.35),
    dip: clamp(segmentAngle(vT2, vT3) * 0.95, 0.02, 1.05)
  }

  return {
    active: true,
    side,
    worldPos,
    pivotQuat,
    pose: {
      rotation: [0, 0, 0],
      thumb,
      index: computeFinger(5, 0.08),
      middle: computeFinger(9, 0.01),
      ring: computeFinger(13, -0.06),
      pinky: computeFinger(17, -0.14)
    }
  }
}

/**
 * Converts normalized trajectory coordinates into Three.js world space points.
 */
export function trailToWorldPoints(
  trail: [number, number, number][],
  analysis: DictionarySequenceAnalysis,
  worldScale: number
): THREE.Vector3[] {
  const [cx, cy, cz] = analysis.center
  return trail.map(
    ([x, y, z]) =>
      new THREE.Vector3(
        (x - cx) * worldScale,
        -(y - cy) * worldScale + 0.05,
        -((z || 0) - cz) * worldScale * 0.55 - 0.02
      )
  )
}
