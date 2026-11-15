import * as THREE from "three"

export interface FingerAngles {
  /** Base spread (abduction/adduction around Z) and curl (flexion around X) */
  spread: number
  mcp: number
  pip: number
  dip: number
}

export interface HandPoseSpec {
  rotation: [number, number, number]
  position?: [number, number, number]
  thumb: FingerAngles
  index: FingerAngles
  middle: FingerAngles
  ring: FingerAngles
  pinky: FingerAngles
}

export type HandGuidePreset =
  | "OPEN"
  | "A"
  | "B"
  | "C"
  | "D"
  | "E"
  | "F"
  | "I"
  | "L"
  | "O"
  | "S"
  | "U"
  | "V"
  | "W"
  | "Y"
  | "HELLO"
  | "LOVE"

const F = (mcp: number, pip: number, dip: number, spread = 0): FingerAngles => ({
  spread,
  mcp,
  pip,
  dip,
})

export const HAND_POSES: Record<HandGuidePreset, HandPoseSpec> = {
  OPEN: {
    rotation: [0.06, 0.08, 0.02],
    thumb: F(0.24, 0.14, 0.08, 0.36),
    index: F(0.07, 0.1, 0.07, 0.11),
    middle: F(0.05, 0.09, 0.06, 0.02),
    ring: F(0.07, 0.11, 0.08, -0.06),
    pinky: F(0.1, 0.13, 0.09, -0.15),
  },
  HELLO: {
    rotation: [0.04, 0.15, -0.1],
    thumb: F(0.22, 0.18, 0.1, 0.32),
    index: F(0.09, 0.14, 0.08, 0.05),
    middle: F(0.07, 0.12, 0.08, 0.0),
    ring: F(0.09, 0.14, 0.09, -0.04),
    pinky: F(0.12, 0.16, 0.1, -0.09),
  },
  A: {
    rotation: [0.08, 0.1, 0.0],
    thumb: F(0.22, 0.28, 0.18, 0.34),
    index: F(1.48, 1.56, 1.15, 0.02),
    middle: F(1.52, 1.58, 1.18, 0.0),
    ring: F(1.52, 1.58, 1.18, -0.02),
    pinky: F(1.48, 1.52, 1.12, -0.05),
  },
  B: {
    rotation: [0.04, 0.1, 0.0],
    thumb: F(0.95, 1.15, 0.8, 0.12),
    index: F(0.04, 0.06, 0.04, 0.02),
    middle: F(0.03, 0.05, 0.04, 0.0),
    ring: F(0.04, 0.06, 0.05, -0.02),
    pinky: F(0.06, 0.08, 0.06, -0.05),
  },
  C: {
    rotation: [0.08, -0.45, 0.08],
    thumb: F(0.55, 0.55, 0.45, 0.42),
    index: F(0.68, 0.85, 0.65, 0.04),
    middle: F(0.7, 0.88, 0.68, 0.01),
    ring: F(0.72, 0.88, 0.68, -0.02),
    pinky: F(0.75, 0.88, 0.68, -0.05),
  },
  D: {
    rotation: [0.08, -0.18, -0.02],
    thumb: F(0.88, 0.92, 0.7, 0.2),
    index: F(0.03, 0.05, 0.04, 0.03),
    middle: F(1.18, 1.32, 0.95, 0.02),
    ring: F(1.26, 1.38, 1.0, -0.02),
    pinky: F(1.3, 1.38, 1.02, -0.06),
  },
  E: {
    rotation: [0.06, 0.08, 0.0],
    thumb: F(1.05, 1.25, 0.95, 0.16),
    index: F(1.25, 1.5, 1.25, 0.03),
    middle: F(1.28, 1.52, 1.28, 0.0),
    ring: F(1.28, 1.52, 1.28, -0.02),
    pinky: F(1.25, 1.48, 1.22, -0.05),
  },
  F: {
    rotation: [0.04, 0.18, 0.02],
    thumb: F(0.85, 0.85, 0.65, 0.25),
    index: F(1.12, 1.2, 0.95, 0.04),
    middle: F(0.04, 0.08, 0.05, 0.06),
    ring: F(0.05, 0.09, 0.06, -0.05),
    pinky: F(0.08, 0.12, 0.08, -0.16),
  },
  I: {
    rotation: [0.05, 0.12, 0.0],
    thumb: F(0.92, 1.05, 0.82, 0.18),
    index: F(1.48, 1.55, 1.15, 0.02),
    middle: F(1.52, 1.58, 1.18, 0.0),
    ring: F(1.48, 1.55, 1.15, -0.02),
    pinky: F(0.05, 0.08, 0.05, -0.16),
  },
  L: {
    rotation: [0.05, 0.08, 0.0],
    thumb: F(0.15, 0.1, 0.06, 0.78),
    index: F(0.03, 0.05, 0.04, 0.04),
    middle: F(1.48, 1.56, 1.18, 0.0),
    ring: F(1.5, 1.58, 1.18, -0.02),
    pinky: F(1.48, 1.54, 1.15, -0.05),
  },
  O: {
    rotation: [0.08, -0.52, 0.05],
    thumb: F(0.75, 0.78, 0.62, 0.34),
    index: F(0.98, 1.12, 0.9, 0.02),
    middle: F(1.02, 1.14, 0.92, 0.0),
    ring: F(1.04, 1.14, 0.92, -0.02),
    pinky: F(1.05, 1.14, 0.92, -0.05),
  },
  S: {
    rotation: [0.06, 0.1, 0.0],
    thumb: F(1.08, 1.18, 0.92, 0.1),
    index: F(1.54, 1.62, 1.24, 0.02),
    middle: F(1.58, 1.64, 1.26, 0.0),
    ring: F(1.58, 1.64, 1.26, -0.02),
    pinky: F(1.54, 1.6, 1.22, -0.04),
  },
  U: {
    rotation: [0.04, 0.1, 0.0],
    thumb: F(0.98, 1.12, 0.85, 0.18),
    index: F(0.03, 0.05, 0.04, 0.02),
    middle: F(0.03, 0.05, 0.04, -0.02),
    ring: F(1.48, 1.56, 1.16, -0.02),
    pinky: F(1.48, 1.54, 1.14, -0.06),
  },
  V: {
    rotation: [0.04, 0.12, 0.0],
    thumb: F(0.98, 1.12, 0.85, 0.18),
    index: F(0.03, 0.06, 0.04, 0.18),
    middle: F(0.03, 0.06, 0.04, -0.14),
    ring: F(1.48, 1.56, 1.16, -0.02),
    pinky: F(1.48, 1.54, 1.14, -0.06),
  },
  W: {
    rotation: [0.04, 0.12, 0.0],
    thumb: F(1.0, 1.14, 0.85, 0.18),
    index: F(0.04, 0.06, 0.04, 0.18),
    middle: F(0.03, 0.06, 0.04, 0.0),
    ring: F(0.05, 0.08, 0.05, -0.18),
    pinky: F(1.48, 1.54, 1.14, -0.06),
  },
  Y: {
    rotation: [0.05, 0.1, 0.04],
    thumb: F(0.14, 0.12, 0.08, 0.72),
    index: F(1.48, 1.56, 1.16, 0.02),
    middle: F(1.52, 1.58, 1.18, 0.0),
    ring: F(1.48, 1.55, 1.15, -0.02),
    pinky: F(0.05, 0.08, 0.05, -0.22),
  },
  LOVE: {
    rotation: [0.04, 0.1, 0.0],
    thumb: F(0.16, 0.12, 0.08, 0.7),
    index: F(0.03, 0.06, 0.04, 0.08),
    middle: F(1.48, 1.56, 1.18, 0.0),
    ring: F(1.48, 1.56, 1.18, -0.02),
    pinky: F(0.05, 0.08, 0.05, -0.2),
  },
}

interface FingerRig {
  basePivot: THREE.Group
  proximalPivot: THREE.Group
  middlePivot: THREE.Group
  distalPivot: THREE.Group
  isThumb: boolean
  sideSign: number
}

export interface ArticulatedHandRig {
  root: THREE.Group
  wristGroup: THREE.Group
  fingers: {
    thumb: FingerRig
    index: FingerRig
    middle: FingerRig
    ring: FingerRig
    pinky: FingerRig
  }
  dispose: () => void
}

export interface SculptedHandOptions {
  /** Base skin/clay color (default warm golden peach matching mockup) */
  skinColor?: number
  sheenColor?: number
  mirrorLeft?: boolean
  scale?: number
}

/**
 * Builds a smooth matte clay-sculpted 3D articulated hand matching the aesthetic
 * of the SignBridge mode menu and fingerspelling.xyz-style 3D guidance.
 */
export function createSculptedHandRig(options: SculptedHandOptions = {}): ArticulatedHandRig {
  const {
    skinColor = 0xeab47f,
    sheenColor = 0xffdec0,
    mirrorLeft = false,
    scale = 1.0,
  } = options

  const sideSign = mirrorLeft ? -1 : 1
  const geometries: THREE.BufferGeometry[] = []

  const track = <T extends THREE.BufferGeometry>(g: T): T => {
    geometries.push(g)
    return g
  }

  const clayMaterial = new THREE.MeshPhysicalMaterial({
    color: skinColor,
    roughness: 0.52,
    metalness: 0.02,
    sheen: 0.42,
    sheenRoughness: 0.65,
    sheenColor: new THREE.Color(sheenColor),
    clearcoat: 0.08,
    clearcoatRoughness: 0.55,
  })

  const padMaterial = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(skinColor).lerp(new THREE.Color(0xffd2b3), 0.26),
    roughness: 0.58,
    metalness: 0.0,
    sheen: 0.3,
    sheenColor: new THREE.Color(sheenColor),
  })

  const root = new THREE.Group()
  root.scale.set(scale * sideSign, scale, scale)

  const wristGroup = new THREE.Group()
  root.add(wristGroup)

  // Wrist base rounded cuff
  const wristGeo = track(new THREE.CapsuleGeometry(0.32, 0.24, 12, 24))
  const wristMesh = new THREE.Mesh(wristGeo, clayMaterial)
  wristMesh.position.set(0, -0.78, -0.02)
  wristMesh.scale.set(1.22, 1.0, 0.72)
  wristGroup.add(wristMesh)

  // Sculpted organic palm using overlapping smooth ellipsoids
  const palmCoreGeo = track(new THREE.SphereGeometry(0.58, 28, 24))
  const palmCore = new THREE.Mesh(palmCoreGeo, clayMaterial)
  palmCore.scale.set(0.98, 1.16, 0.45)
  palmCore.position.set(0.0, 0.02, 0.0)
  wristGroup.add(palmCore)

  // Recessed upper knuckle transition seamlessly merging fingers into palm
  const knuckleBridgeGeo = track(new THREE.CapsuleGeometry(0.19, 0.56, 12, 24))
  const knuckleBridge = new THREE.Mesh(knuckleBridgeGeo, clayMaterial)
  knuckleBridge.rotation.z = Math.PI / 2 - 0.04
  knuckleBridge.scale.set(0.88, 0.85, 0.72)
  knuckleBridge.position.set(0.0, 0.43, 0.0)
  wristGroup.add(knuckleBridge)

  // Soft thenar muscle cushion under thumb for organic silhouette
  const thenarGeo = track(new THREE.SphereGeometry(0.32, 20, 18))
  const thenar = new THREE.Mesh(thenarGeo, padMaterial)
  thenar.scale.set(0.92, 1.24, 0.78)
  thenar.position.set(0.33, -0.2, 0.08)
  thenar.rotation.z = -0.32
  wristGroup.add(thenar)

  const createPhalanx = (
    radiusTop: number,
    radiusBottom: number,
    length: number,
    withTip = false
  ): THREE.Group => {
    const seg = new THREE.Group()

    const cylGeo = track(
      new THREE.CylinderGeometry(radiusTop, radiusBottom, length, 20, 1, false)
    )
    const cyl = new THREE.Mesh(cylGeo, clayMaterial)
    cyl.position.y = length / 2
    seg.add(cyl)

    const baseCapGeo = track(new THREE.SphereGeometry(radiusBottom * 1.01, 18, 16))
    const baseCap = new THREE.Mesh(baseCapGeo, clayMaterial)
    seg.add(baseCap)

    const topCapGeo = track(new THREE.SphereGeometry(radiusTop * 1.01, 18, 16))
    const topCap = new THREE.Mesh(topCapGeo, withTip ? padMaterial : clayMaterial)
    topCap.position.y = length
    if (withTip) {
      topCap.scale.set(1.0, 1.12, 0.96)
    }
    seg.add(topCap)

    return seg
  }

  const buildFinger = (
    anchor: [number, number, number],
    radius: number,
    lengths: [number, number, number],
    restSpread: number,
    isThumb = false
  ): FingerRig => {
    const basePivot = new THREE.Group()
    basePivot.position.set(...anchor)
    basePivot.rotation.z = restSpread
    if (isThumb) {
      basePivot.rotation.y = -0.52
      basePivot.rotation.z = -0.58
    }
    wristGroup.add(basePivot)

    const proximalPivot = new THREE.Group()
    basePivot.add(proximalPivot)
    proximalPivot.add(createPhalanx(radius * 0.93, radius, lengths[0], false))

    const middlePivot = new THREE.Group()
    middlePivot.position.y = lengths[0]
    proximalPivot.add(middlePivot)
    middlePivot.add(createPhalanx(radius * 0.85, radius * 0.92, lengths[1], false))

    const distalPivot = new THREE.Group()
    distalPivot.position.y = lengths[1]
    middlePivot.add(distalPivot)
    distalPivot.add(createPhalanx(radius * 0.72, radius * 0.84, lengths[2], true))

    return {
      basePivot,
      proximalPivot,
      middlePivot,
      distalPivot,
      isThumb,
      sideSign: 1,
    }
  }

  const fingers = {
    thumb: buildFinger([0.41, -0.14, 0.12], 0.142, [0.35, 0.28, 0.24], -0.34, true),
    index: buildFinger([0.33, 0.5, 0.01], 0.13, [0.43, 0.3, 0.24], -0.05),
    middle: buildFinger([0.1, 0.55, 0.01], 0.134, [0.48, 0.34, 0.25], 0.01),
    ring: buildFinger([-0.14, 0.52, 0.0], 0.126, [0.44, 0.31, 0.23], 0.06),
    pinky: buildFinger([-0.35, 0.44, -0.01], 0.11, [0.35, 0.24, 0.2], 0.12),
  }

  const dispose = () => {
    geometries.forEach((g) => g.dispose())
    clayMaterial.dispose()
    padMaterial.dispose()
  }

  return { root, wristGroup, fingers, dispose }
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Smoothly applies (or interpolates toward) a HandPoseSpec on the articulated rig.
 */
export function applyHandPose(
  rig: ArticulatedHandRig,
  pose: HandPoseSpec,
  blend = 0.18,
  extraRot: [number, number, number] = [0, 0, 0]
): void {
  const { wristGroup, fingers } = rig

  wristGroup.rotation.x = lerp(wristGroup.rotation.x, pose.rotation[0] + extraRot[0], blend)
  wristGroup.rotation.y = lerp(wristGroup.rotation.y, pose.rotation[1] + extraRot[1], blend)
  wristGroup.rotation.z = lerp(wristGroup.rotation.z, pose.rotation[2] + extraRot[2], blend)

  const applyOne = (f: FingerRig, spec: FingerAngles) => {
    if (f.isThumb) {
      f.basePivot.rotation.z = lerp(f.basePivot.rotation.z, -0.15 - spec.spread, blend)
      f.basePivot.rotation.y = lerp(f.basePivot.rotation.y, -0.58 + spec.mcp * 0.45, blend)
      f.proximalPivot.rotation.x = lerp(f.proximalPivot.rotation.x, spec.mcp * 0.72, blend)
      f.middlePivot.rotation.x = lerp(f.middlePivot.rotation.x, spec.pip * 0.85, blend)
      f.distalPivot.rotation.x = lerp(f.distalPivot.rotation.x, spec.dip, blend)
      return
    }
    f.basePivot.rotation.z = lerp(f.basePivot.rotation.z, -spec.spread, blend)
    f.proximalPivot.rotation.x = lerp(f.proximalPivot.rotation.x, spec.mcp, blend)
    f.middlePivot.rotation.x = lerp(f.middlePivot.rotation.x, spec.pip, blend)
    f.distalPivot.rotation.x = lerp(f.distalPivot.rotation.x, spec.dip, blend)
  }

  applyOne(fingers.thumb, pose.thumb)
  applyOne(fingers.index, pose.index)
  applyOne(fingers.middle, pose.middle)
  applyOne(fingers.ring, pose.ring)
  applyOne(fingers.pinky, pose.pinky)
}

/**
 * Adds soft directional + rim studio lighting matching the warm SignBridge palette.
 */
export function addWarmStudioLights(scene: THREE.Scene): void {
  scene.add(new THREE.AmbientLight(0xffe9d6, 1.35))

  const keyLight = new THREE.DirectionalLight(0xfff1de, 2.15)
  keyLight.position.set(2.8, 3.6, 4.8)
  scene.add(keyLight)

  const warmFill = new THREE.DirectionalLight(0xff9b42, 1.1)
  warmFill.position.set(-3.5, -1.2, 3.0)
  scene.add(warmFill)

  const rimLight = new THREE.DirectionalLight(0xfbbf24, 1.35)
  rimLight.position.set(0.2, 3.2, -3.5)
  scene.add(rimLight)
}
