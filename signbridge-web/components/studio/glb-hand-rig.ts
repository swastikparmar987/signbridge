import * as THREE from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import type { HandGuidePreset, HandPoseSpec, FingerAngles } from "./hand-3d-rig"

interface BoneRestState {
  bone: THREE.Object3D
  restPos: THREE.Vector3
  restQuat: THREE.Quaternion
  parentName: string | null
}

export interface GlbHandRig {
  root: THREE.Group
  pivot: THREE.Group
  bones: Map<string, BoneRestState>
  dispose: () => void
}

const BONE_ORDER: Array<[string, string | null]> = [
  ["wrist", null],
  ["thumb-metacarpal", "wrist"],
  ["thumb-phalanx-proximal", "thumb-metacarpal"],
  ["thumb-phalanx-distal", "thumb-phalanx-proximal"],
  ["thumb-tip", "thumb-phalanx-distal"],
  ["index-finger-metacarpal", "wrist"],
  ["index-finger-phalanx-proximal", "index-finger-metacarpal"],
  ["index-finger-phalanx-intermediate", "index-finger-phalanx-proximal"],
  ["index-finger-phalanx-distal", "index-finger-phalanx-intermediate"],
  ["index-finger-tip", "index-finger-phalanx-distal"],
  ["middle-finger-metacarpal", "wrist"],
  ["middle-finger-phalanx-proximal", "middle-finger-metacarpal"],
  ["middle-finger-phalanx-intermediate", "middle-finger-phalanx-proximal"],
  ["middle-finger-phalanx-distal", "middle-finger-phalanx-intermediate"],
  ["middle-finger-tip", "middle-finger-phalanx-distal"],
  ["ring-finger-metacarpal", "wrist"],
  ["ring-finger-phalanx-proximal", "ring-finger-metacarpal"],
  ["ring-finger-phalanx-intermediate", "ring-finger-phalanx-proximal"],
  ["ring-finger-phalanx-distal", "ring-finger-phalanx-intermediate"],
  ["ring-finger-tip", "ring-finger-phalanx-distal"],
  ["pinky-finger-metacarpal", "wrist"],
  ["pinky-finger-phalanx-proximal", "pinky-finger-metacarpal"],
  ["pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-proximal"],
  ["pinky-finger-phalanx-distal", "pinky-finger-phalanx-intermediate"],
  ["pinky-finger-tip", "pinky-finger-phalanx-distal"],
]

const loader = new GLTFLoader()

export async function loadGlbHandRig(options: {
  url?: string
  skinColor?: number
  sheenColor?: number
} = {}): Promise<GlbHandRig> {
  const {
    url = "/models/hand-right.glb",
    skinColor = 0xe9b482,
    sheenColor = 0xffe4cc,
  } = options

  const gltf = await loader.loadAsync(url)
  const sceneRoot = gltf.scene

  const clayMat = new THREE.MeshPhysicalMaterial({
    color: skinColor,
    roughness: 0.45,
    metalness: 0.01,
    sheen: 0.42,
    sheenRoughness: 0.6,
    sheenColor: new THREE.Color(sheenColor),
    clearcoat: 0.08,
    clearcoatRoughness: 0.48,
  })

  sceneRoot.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) {
      const mesh = obj as THREE.Mesh
      mesh.geometry.computeVertexNormals()
      mesh.material = clayMat
      mesh.frustumCulled = false
    }
  })

  // Record bind-pose transforms cleanly before normalizing wrapper transforms
  const bones = new Map<string, BoneRestState>()
  for (const [name, parentName] of BONE_ORDER) {
    const obj = sceneRoot.getObjectByName(name)
    if (obj) {
      bones.set(name, {
        bone: obj,
        restPos: obj.position.clone(),
        restQuat: obj.quaternion.clone(),
        parentName,
      })
    }
  }

  // Compute bounding box to center wrist-to-fingertips neatly upright in camera view
  const box = new THREE.Box3().setFromObject(sceneRoot)
  const size = new THREE.Vector3()
  const center = new THREE.Vector3()
  box.getSize(size)
  box.getCenter(center)

  const pivot = new THREE.Group()
  // Right-hand GLB points fingertips -Y, back-of-hand +X; flip upright facing palm forward
  sceneRoot.position.set(-center.x, -center.y, -center.z)
  const alignGroup = new THREE.Group()
  alignGroup.add(sceneRoot)
  alignGroup.rotation.set(0, -Math.PI / 2, Math.PI)
  const maxDim = Math.max(size.x, size.y, size.z, 0.001)
  alignGroup.scale.setScalar(1.9 / maxDim)
  pivot.add(alignGroup)

  const root = new THREE.Group()
  root.add(pivot)

  const dispose = () => {
    sceneRoot.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) {
        (obj as THREE.Mesh).geometry.dispose()
      }
    })
    clayMat.dispose()
  }

  return { root, pivot, bones, dispose }
}

function getLocalBend(
  boneName: string,
  pose: HandPoseSpec
): { curlX: number; spreadY: number } {
  const fingerMap: Record<string, FingerAngles> = {
    thumb: pose.thumb,
    index: pose.index,
    middle: pose.middle,
    ring: pose.ring,
    pinky: pose.pinky,
  }

  const key = boneName.startsWith("thumb")
    ? "thumb"
    : boneName.startsWith("index")
      ? "index"
      : boneName.startsWith("middle")
        ? "middle"
        : boneName.startsWith("ring")
          ? "ring"
          : boneName.startsWith("pinky")
            ? "pinky"
            : null

  if (!key) return { curlX: 0, spreadY: 0 }
  const spec = fingerMap[key]

  if (key === "thumb") {
    if (boneName.endsWith("metacarpal")) {
      return { curlX: -spec.mcp * 0.45, spreadY: spec.spread * 0.45 }
    }
    if (boneName.endsWith("proximal")) {
      return { curlX: -spec.mcp * 0.72, spreadY: spec.spread * 0.25 }
    }
    if (boneName.endsWith("distal")) {
      return { curlX: -spec.pip * 0.85, spreadY: 0 }
    }
    return { curlX: 0, spreadY: 0 }
  }

  if (boneName.endsWith("proximal")) {
    return { curlX: -spec.mcp, spreadY: spec.spread }
  }
  if (boneName.endsWith("intermediate")) {
    return { curlX: -spec.pip, spreadY: 0 }
  }
  if (boneName.endsWith("distal")) {
    return { curlX: -spec.dip, spreadY: 0 }
  }
  return { curlX: 0, spreadY: 0 }
}

const _tmpDelta = new THREE.Quaternion()
const _tmpEuler = new THREE.Euler()
const _tmpBend = new THREE.Quaternion()
const _tmpOffset = new THREE.Vector3()
const _invRest = new THREE.Quaternion()

/**
 * Forward-kinematic pose solver for the continuous WebXR SkinnedMesh hand.
 */
export function applyGlbHandPose(
  rig: GlbHandRig,
  pose: HandPoseSpec,
  blend = 0.2,
  extraRot: [number, number, number] = [0, 0, 0]
): void {
  rig.pivot.rotation.x += (pose.rotation[0] + extraRot[0] - rig.pivot.rotation.x) * blend
  rig.pivot.rotation.y += (pose.rotation[1] + extraRot[1] - rig.pivot.rotation.y) * blend
  rig.pivot.rotation.z += (pose.rotation[2] + extraRot[2] - rig.pivot.rotation.z) * blend

  const worldPos = new Map<string, THREE.Vector3>()
  const worldQuat = new Map<string, THREE.Quaternion>()
  const worldDelta = new Map<string, THREE.Quaternion>()

  for (const [name, parentName] of BONE_ORDER) {
    const state = rig.bones.get(name)
    if (!state) continue

    const targetPos = state.restPos.clone()
    const targetQuat = state.restQuat.clone()

    if (parentName) {
      const pState = rig.bones.get(parentName)
      const pCurPos = worldPos.get(parentName)
      const pDelta = worldDelta.get(parentName)
      if (pState && pCurPos && pDelta) {
        _tmpOffset.copy(state.restPos).sub(pState.restPos).applyQuaternion(pDelta)
        targetPos.copy(pCurPos).add(_tmpOffset)
        targetQuat.copy(pDelta).multiply(state.restQuat)
      }
    }

    const { curlX, spreadY } = getLocalBend(name, pose)
    if (curlX !== 0 || spreadY !== 0) {
      _tmpEuler.set(curlX, spreadY, 0, "YXZ")
      _tmpBend.setFromEuler(_tmpEuler)
      targetQuat.multiply(_tmpBend)
    }

    state.bone.position.lerp(targetPos, blend)
    state.bone.quaternion.slerp(targetQuat, blend)

    _invRest.copy(state.restQuat).invert()
    _tmpDelta.copy(state.bone.quaternion).multiply(_invRest)

    worldPos.set(name, state.bone.position.clone())
    worldQuat.set(name, state.bone.quaternion.clone())
    worldDelta.set(name, _tmpDelta.clone())
  }
}

export type { HandGuidePreset }
