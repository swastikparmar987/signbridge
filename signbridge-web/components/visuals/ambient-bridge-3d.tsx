"use client"

import * as React from "react"
import { useReducedMotion } from "framer-motion"
import * as THREE from "three"

/**
 * Lightweight GPU-accelerated Three.js ambient sculptural ribbon backdrop.
 * Renders warm translucent amber glass waves (inspired by the SignBridge brand artwork)
 * with graceful SVG fallback if WebGL is unavailable or reduced motion is preferred.
 */
export function AmbientBridge3D() {
  const prefersReduced = useReducedMotion()
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const [webglReady, setWebglReady] = React.useState(false)

  React.useEffect(() => {
    if (prefersReduced) return
    const canvas = canvasRef.current
    if (!canvas) return

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
      return
    }

    const handleContextLost = (e: Event) => {
      e.preventDefault()
      setWebglReady(false)
    }
    canvas.addEventListener("webglcontextlost", handleContextLost)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100)
    camera.position.set(0, 0.2, 8.6)

    const ambientLight = new THREE.AmbientLight(0xfff1e3, 1.35)
    scene.add(ambientLight)

    const warmKeyLight = new THREE.DirectionalLight(0xffb470, 2.2)
    warmKeyLight.position.set(4, 6, 5)
    scene.add(warmKeyLight)

    const rimLight = new THREE.DirectionalLight(0xff7900, 1.45)
    rimLight.position.set(-5, -3, 4)
    scene.add(rimLight)

    const buildRibbonGeometry = (segX: number, segY: number, width: number, height: number) => {
      return new THREE.PlaneGeometry(width, height, segX, segY)
    }

    const primaryGeo = buildRibbonGeometry(88, 18, 19.5, 2.35)
    const primaryMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color("#FF892B"),
      emissive: new THREE.Color("#FF6D00"),
      emissiveIntensity: 0.14,
      roughness: 0.24,
      metalness: 0.05,
      transmission: 0.42,
      transparent: true,
      opacity: 0.34,
      side: THREE.DoubleSide
    })
    const primaryRibbon = new THREE.Mesh(primaryGeo, primaryMat)
    primaryRibbon.position.set(0.4, 2.15, -0.5)
    primaryRibbon.rotation.set(0.28, -0.1, 0.14)
    scene.add(primaryRibbon)

    const secondaryGeo = buildRibbonGeometry(76, 14, 18.5, 1.6)
    const secondaryMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color("#FFB978"),
      emissive: new THREE.Color("#FF9233"),
      emissiveIntensity: 0.12,
      roughness: 0.35,
      metalness: 0.04,
      transparent: true,
      opacity: 0.24,
      side: THREE.DoubleSide
    })
    const secondaryRibbon = new THREE.Mesh(secondaryGeo, secondaryMat)
    secondaryRibbon.position.set(-0.6, 1.65, -1.6)
    secondaryRibbon.rotation.set(-0.22, 0.12, -0.1)
    scene.add(secondaryRibbon)

    // Store base positions for clean sinusoidal animation
    const primaryBase = Float32Array.from(primaryGeo.attributes.position.array as Float32Array)
    const secondaryBase = Float32Array.from(secondaryGeo.attributes.position.array as Float32Array)

    let pointerX = 0
    let pointerY = 0
    const onPointerMove = (e: PointerEvent) => {
      if (window.innerWidth === 0 || window.innerHeight === 0) return
      pointerX = (e.clientX / window.innerWidth - 0.5) * 0.4
      pointerY = (e.clientY / window.innerHeight - 0.5) * 0.3
    }
    window.addEventListener("pointermove", onPointerMove, { passive: true })

    const resize = () => {
      if (!renderer || disposed) return
      const w = window.innerWidth
      const h = Math.min(window.innerHeight, 640)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75))
      renderer.setSize(w, h, false)
      camera.aspect = w / Math.max(h, 1)
      camera.updateProjectionMatrix()
    }
    resize()
    window.addEventListener("resize", resize)

    const clock = new THREE.Clock()
    let isFirstFrame = true

    const animate = () => {
      if (disposed || !renderer) return
      if (document.hidden) {
        frameId = window.requestAnimationFrame(animate)
        return
      }

      const t = clock.getElapsedTime() * 0.55

      const pPos = primaryGeo.attributes.position as THREE.BufferAttribute
      for (let i = 0; i < pPos.count; i++) {
        const bx = primaryBase[i * 3]
        const by = primaryBase[i * 3 + 1]
        const wave = Math.sin(bx * 0.42 + t) * 0.65 + Math.cos(bx * 0.22 - t * 0.8) * 0.35
        const twist = Math.sin(bx * 0.3 + by * 0.5 + t * 0.7) * 0.45
        pPos.setXYZ(i, bx, by + wave * 0.48, twist)
      }
      pPos.needsUpdate = true
      primaryGeo.computeVertexNormals()

      const sPos = secondaryGeo.attributes.position as THREE.BufferAttribute
      for (let i = 0; i < sPos.count; i++) {
        const bx = secondaryBase[i * 3]
        const by = secondaryBase[i * 3 + 1]
        const wave = Math.cos(bx * 0.38 - t * 0.85) * 0.55
        const twist = Math.sin(bx * 0.25 - by * 0.45 + t * 0.6) * 0.4
        sPos.setXYZ(i, bx, by + wave * 0.45, twist)
      }
      sPos.needsUpdate = true
      secondaryGeo.computeVertexNormals()

      camera.position.x += (pointerX - camera.position.x) * 0.04
      camera.position.y += (0.25 - pointerY - camera.position.y) * 0.04
      camera.lookAt(0, 0.8, 0)

      renderer.render(scene, camera)
      if (isFirstFrame) {
        isFirstFrame = false
        setWebglReady(true)
      }
      frameId = window.requestAnimationFrame(animate)
    }

    frameId = window.requestAnimationFrame(animate)

    return () => {
      disposed = true
      window.cancelAnimationFrame(frameId)
      window.removeEventListener("resize", resize)
      window.removeEventListener("pointermove", onPointerMove)
      canvas.removeEventListener("webglcontextlost", handleContextLost)
      primaryGeo.dispose()
      primaryMat.dispose()
      secondaryGeo.dispose()
      secondaryMat.dispose()
      renderer?.dispose()
    }
  }, [prefersReduced])

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-x-0 top-0 h-[560px] z-0 overflow-hidden select-none"
    >
      {/* Subtle sculptural SVG fallback always present beneath WebGL */}
      <svg
        viewBox="0 0 1440 420"
        preserveAspectRatio="none"
        className={`absolute inset-0 w-full h-full transition-opacity duration-700 ${
          webglReady ? "opacity-25" : "opacity-45"
        }`}
      >
        <defs>
          <linearGradient id="sbAmbientRibbonA" x1="0" y1="0" x2="1" y2="0.4">
            <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.16" />
            <stop offset="48%" stopColor="#D97706" stopOpacity="0.24" />
            <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0.14" />
          </linearGradient>
          <linearGradient id="sbAmbientRibbonB" x1="1" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.18" />
            <stop offset="55%" stopColor="#FBBF24" stopOpacity="0.14" />
            <stop offset="100%" stopColor="#15121E" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path
          d="M-60,245 C220,175 430,300 720,165 C980,45 1210,120 1500,30 L1500,145 C1210,205 975,140 725,245 C430,365 215,250 -60,320 Z"
          fill="url(#sbAmbientRibbonA)"
        />
        <path
          d="M-40,120 C260,185 520,85 830,185 C1090,265 1290,195 1490,95 L1490,155 C1270,260 1055,310 810,225 C510,125 250,235 -40,175 Z"
          fill="url(#sbAmbientRibbonB)"
        />
      </svg>

      {!prefersReduced && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full opacity-55"
        />
      )}

      {/* Velvet dark bottom veil so main workspace stays crisp and legible */}
      <div className="absolute inset-x-0 bottom-0 h-56 bg-gradient-to-t from-[#15121E] via-[#15121E]/90 to-transparent" />
    </div>
  )
}

