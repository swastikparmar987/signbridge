"use client"

import * as React from "react"
import { useReducedMotion } from "framer-motion"
import * as THREE from "three"

/**
 * Interactive Signature SignBridge 3D Connection Arch:
 * Visualizes ASL gesture nodes crossing a glowing warm-orange bridge arch to speech output.
 */
export function BridgeScene3D({ className = "" }: { className?: string }) {
  const prefersReduced = useReducedMotion()
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const [webglOk, setWebglOk] = React.useState(true)
  const hasWebgl = !prefersReduced && webglOk

  React.useEffect(() => {
    if (prefersReduced) return
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    let renderer: THREE.WebGLRenderer | null = null
    let frameId = 0
    let disposed = false

    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
        powerPreference: "low-power"
      })
    } catch {
      window.setTimeout(() => setWebglOk(false), 0)
      return
    }

    const onLost = (e: Event) => {
      e.preventDefault()
      setWebglOk(false)
    }
    canvas.addEventListener("webglcontextlost", onLost)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 50)
    camera.position.set(0, 0.35, 5.1)

    scene.add(new THREE.AmbientLight(0xfff4ea, 1.45))
    const keyLight = new THREE.DirectionalLight(0xff9a3c, 2.3)
    keyLight.position.set(3, 5, 4)
    scene.add(keyLight)
    const rimLight = new THREE.DirectionalLight(0x34d399, 0.85)
    rimLight.position.set(-3, 2, -2)
    scene.add(rimLight)

    const group = new THREE.Group()
    scene.add(group)

    // Wide sweeping bridge arch tube
    const archCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-2.65, -0.58, 0),
      new THREE.Vector3(-1.35, 0.55, 0.22),
      new THREE.Vector3(0, 0.86, 0),
      new THREE.Vector3(1.35, 0.55, -0.22),
      new THREE.Vector3(2.65, -0.58, 0)
    ])
    const archGeo = new THREE.TubeGeometry(archCurve, 56, 0.085, 16, false)
    const archMat = new THREE.MeshStandardMaterial({
      color: 0xff7900,
      emissive: 0xe85d04,
      emissiveIntensity: 0.34,
      roughness: 0.22,
      metalness: 0.2
    })
    group.add(new THREE.Mesh(archGeo, archMat))

    // Lower curved deck & subtle structural struts
    const deckCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-2.55, -0.58, 0.08),
      new THREE.Vector3(0, -0.34, -0.06),
      new THREE.Vector3(2.55, -0.58, 0.08)
    ])
    const deckGeo = new THREE.TubeGeometry(deckCurve, 40, 0.042, 12, false)
    const deckMat = new THREE.MeshStandardMaterial({
      color: 0xffb36b,
      roughness: 0.3,
      transparent: true,
      opacity: 0.88
    })
    group.add(new THREE.Mesh(deckGeo, deckMat))

    const cableMat = new THREE.MeshBasicMaterial({
      color: 0xf4a261,
      transparent: true,
      opacity: 0.38
    })
    const cableGeos: THREE.CylinderGeometry[] = []
    for (const u of [0.22, 0.36, 0.5, 0.64, 0.78]) {
      const topPt = archCurve.getPointAt(u)
      const botPt = deckCurve.getPointAt(u)
      const height = Math.max(0.08, topPt.distanceTo(botPt))
      const cg = new THREE.CylinderGeometry(0.014, 0.014, height, 8)
      cableGeos.push(cg)
      const cable = new THREE.Mesh(cg, cableMat)
      cable.position.lerpVectors(topPt, botPt, 0.5)
      group.add(cable)
    }

    // Landmark node spheres (Sign side -> Speech side)
    const nodeGeo = new THREE.SphereGeometry(0.16, 20, 20)
    const leftMat = new THREE.MeshStandardMaterial({
      color: 0xff7900,
      emissive: 0xff5a00,
      emissiveIntensity: 0.4,
      roughness: 0.2
    })
    const rightMat = new THREE.MeshStandardMaterial({
      color: 0x1fa968,
      emissive: 0x10b981,
      emissiveIntensity: 0.3,
      roughness: 0.25
    })
    const leftNode = new THREE.Mesh(nodeGeo, leftMat)
    leftNode.position.set(-2.65, -0.58, 0)
    const rightNode = new THREE.Mesh(nodeGeo, rightMat)
    rightNode.position.set(2.65, -0.58, 0)
    group.add(leftNode, rightNode)

    // Traveling glowing pulse spheres along the arch
    const pulseGeo = new THREE.SphereGeometry(0.085, 16, 16)
    const pulseMat = new THREE.MeshStandardMaterial({
      color: 0xfff8f1,
      emissive: 0xffb347,
      emissiveIntensity: 0.9,
      roughness: 0.15
    })
    const pulses = [0, 0.33, 0.66].map(() => {
      const m = new THREE.Mesh(pulseGeo, pulseMat)
      group.add(m)
      return m
    })

    let hoverTargetY = 0
    const onPointerMove = (e: PointerEvent) => {
      const rect = container.getBoundingClientRect()
      if (!rect.width) return
      const nx = (e.clientX - rect.left) / rect.width - 0.5
      hoverTargetY = nx * 0.42
    }
    const onPointerLeave = () => {
      hoverTargetY = 0
    }
    container.addEventListener("pointermove", onPointerMove)
    container.addEventListener("pointerleave", onPointerLeave)

    const resize = () => {
      if (!renderer || !container || disposed) return
      const w = container.clientWidth || 320
      const h = container.clientHeight || 120
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      renderer.setSize(w, h, false)
      camera.aspect = w / Math.max(h, 1)
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    const clock = new THREE.Clock()
    const animate = () => {
      if (disposed || !renderer) return
      if (!document.hidden) {
        const t = clock.getElapsedTime()
        group.rotation.y += (Math.sin(t * 0.55) * 0.18 + hoverTargetY - group.rotation.y) * 0.06
        group.position.y = Math.sin(t * 1.1) * 0.06

        pulses.forEach((mesh, idx) => {
          const u = (t * 0.32 + idx * 0.333) % 1
          const pt = archCurve.getPointAt(u)
          mesh.position.copy(pt)
        })
        renderer.render(scene, camera)
      }
      frameId = window.requestAnimationFrame(animate)
    }
    frameId = window.requestAnimationFrame(animate)

    return () => {
      disposed = true
      window.cancelAnimationFrame(frameId)
      ro.disconnect()
      container.removeEventListener("pointermove", onPointerMove)
      container.removeEventListener("pointerleave", onPointerLeave)
      canvas.removeEventListener("webglcontextlost", onLost)
      archGeo.dispose()
      archMat.dispose()
      deckGeo.dispose()
      deckMat.dispose()
      cableGeos.forEach((g) => g.dispose())
      cableMat.dispose()
      nodeGeo.dispose()
      leftMat.dispose()
      rightMat.dispose()
      pulseGeo.dispose()
      pulseMat.dispose()
      renderer?.dispose()
    }
  }, [prefersReduced])

  return (
    <div ref={containerRef} className={`relative w-full h-24 overflow-hidden ${className}`}>
      {hasWebgl ? (
        <canvas ref={canvasRef} className="w-full h-full block" />
      ) : (
        <svg viewBox="0 0 280 96" className="w-full h-full" aria-hidden="true">
          <path
            d="M34 72 C85 18 195 18 246 72"
            fill="none"
            stroke="#FF7900"
            strokeWidth="6"
            strokeLinecap="round"
          />
          <circle cx="34" cy="72" r="7" fill="#FF7900" />
          <circle cx="246" cy="72" r="7" fill="#1FA968" />
        </svg>
      )}
    </div>
  )
}
