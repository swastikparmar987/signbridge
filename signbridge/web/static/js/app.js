/**
 * SignBridge Frontend Application v3.0
 *
 * WEBCAM PIPELINE:
 *  Browser webcam → getUserMedia() → MediaPipe Hands → 42 landmarks/frame
 *  → 2.0-second gesture capture & rolling history buffer (~60 frames)
 *  → Uniform 32-frame downsampling: np.linspace(0, total - 1, 32)
 *  → Preserves natural lead-in and lead-out resting zeros (matches training data)
 *  → POST /api/predict/sequence [32, 42, 3]
 *  → FastAPI → normalize_sequence() → GRU → 2,731-class prediction → UI
 *
 * MODES:
 *  1. Interactive "Sign Now" (Spacebar / Click): 2.0-second timed gesture capture with progress bar
 *  2. Continuous Live Stream: Resamples last 2.0s rolling history every 600ms with temporal debouncing
 *  3. Diagnostic HUD: Toggle with 'D' key or '🛠️ Debug' button
 */

document.addEventListener('DOMContentLoaded', () => {

  // --------------------------------------------------------------------------
  // DOM Elements
  // --------------------------------------------------------------------------
  const navRecognize            = document.getElementById('nav-recognize');
  const navLearn                = document.getElementById('nav-learn');
  const btnTeachMeTop           = document.getElementById('btn-teach-me-top');
  const btnTeachMeChip          = document.getElementById('btn-teach-me-chip');
  const viewRecognize           = document.getElementById('view-recognize');
  const viewLearn               = document.getElementById('view-learn');

  const tabCamera               = document.getElementById('tab-camera');
  const tabUpload               = document.getElementById('tab-upload');
  const panelCamera             = document.getElementById('panel-camera');
  const panelUpload             = document.getElementById('panel-upload');

  const webcamFeed              = document.getElementById('webcam-feed');
  const landmarkCanvas          = document.getElementById('landmark-canvas');
  const canvasCtx               = landmarkCanvas ? landmarkCanvas.getContext('2d') : null;

  const btnStartCamera          = document.getElementById('btn-start-camera');
  const btnToggleCamera         = document.getElementById('btn-toggle-camera');
  const btnCaptureSign          = document.getElementById('btn-capture-sign');
  const btnToggleDebug          = document.getElementById('btn-toggle-debug');
  const cameraPromptOverlay     = document.getElementById('camera-prompt-overlay');
  const cameraStatusBadge       = document.getElementById('camera-status-badge');
  const cameraStatusText        = document.getElementById('camera-status-text');

  const captureProgressContainer= document.getElementById('capture-progress-container');
  const captureProgressFill     = document.getElementById('capture-progress-fill');
  const captureStatusText       = document.getElementById('capture-status-text');
  const captureTimeText         = document.getElementById('capture-time-text');

  const dropzone                = document.getElementById('dropzone');
  const videoFileInput          = document.getElementById('video-file-input');
  const uploadPreviewContainer  = document.getElementById('upload-preview-container');
  const uploadVideoPlayer       = document.getElementById('upload-video-player');
  const uploadFileName          = document.getElementById('upload-file-name');
  const btnReupload             = document.getElementById('btn-reupload');

  const primaryGloss            = document.getElementById('primary-gloss');
  const primaryConfidence       = document.getElementById('primary-confidence');
  const confidenceAssessment    = document.getElementById('confidence-assessment');
  const confidenceBadge         = document.getElementById('confidence-badge');
  const top5List                = document.getElementById('top5-list');

  const learnSearchInput        = document.getElementById('learn-search-input');
  const vocabList               = document.getElementById('vocab-list');
  const vocabCountBadge         = document.getElementById('vocab-count-badge');
  const demoGlossTitle          = document.getElementById('demo-gloss-title');
  const demoTierBadge           = document.getElementById('demo-tier-badge');
  const demoVideoPlayer         = document.getElementById('demo-video-player');
  const demoVideoPlaceholder    = document.getElementById('demo-video-placeholder');
  const demoPlaceholderText     = document.getElementById('demo-placeholder-text');
  const btnTrySign              = document.getElementById('btn-try-sign');

  const practiceTargetBanner    = document.getElementById('practice-target-banner');
  const practiceTargetGloss     = document.getElementById('practice-target-gloss');
  const btnCancelPractice       = document.getElementById('btn-cancel-practice');

  // Sentence Builder DOM Elements
  const sbWordCount             = document.getElementById('sb-word-count');
  const sbAutoAddToggle         = document.getElementById('sb-auto-add-toggle');
  const sbEmptyState            = document.getElementById('sb-empty-state');
  const sbWordsContainer        = document.getElementById('sb-words-container');
  const sbBtnSpeak              = document.getElementById('sb-btn-speak');
  const sbBtnCopy               = document.getElementById('sb-btn-copy');
  const sbCopyText              = document.getElementById('sb-copy-text');
  const sbBtnAddCurrent         = document.getElementById('sb-btn-add-current');
  const sbBtnBackspace          = document.getElementById('sb-btn-backspace');
  const sbBtnClear              = document.getElementById('sb-btn-clear');
  const sbQuickChips            = document.querySelectorAll('.sb-quick-chip');

  // --------------------------------------------------------------------------
  // Pipeline Parameters & State
  // --------------------------------------------------------------------------
  const SEQUENCE_LENGTH         = 32;       // Target temporal length for GRU model
  const HISTORY_WINDOW_FRAMES   = 60;       // ~2.0 seconds at 30 FPS
  const GESTURE_CAPTURE_MS      = 2000;     // 2.0s capture window for "Sign Now"
  const CONTINUOUS_INTERVAL_MS  = 650;      // Continuous streaming inference interval

  let isCameraRunning           = false;
  let mediaStream               = null;
  let handsDetector             = null;
  let animationFrameId          = null;

  // Buffering
  let rollingHistory            = [];       // Rolling buffer of last ~60 raw frames (including zeros)
  let isRecordingGesture        = false;    // True during 2-second "Sign Now" capture
  let gestureRecordedFrames     = [];       // Accumulated frames during interactive capture
  let gestureStartTime          = 0;
  let gestureProgressInterval   = null;

  let lastContinuousInference   = 0;
  let isInferencing             = false;

  // Smoothing & Stability
  let lastStableGloss           = '—';
  let consecutiveMatches        = 0;
  const CONSECUTIVE_REQUIRED    = 2;

  // Target sign for practice mode
  let targetPracticeGloss       = null;
  let currentLearnGloss         = 'APPLE';
  let searchDebounceTimeout     = null;

  // Diagnostic State
  let debugVisible              = false;
  const DS = {
    cameraReady:        false,
    mediaPipeReady:     false,
    mediaPipeError:     '',
    handsDetected:      0,
    historyLength:      0,
    activeFramesInHist: 0,
    apiStatus:          'IDLE',
    lastPrediction:     '—',
    apiLatencyMs:       0,
    totalFrames:        0,
    samplingMode:       'IDLE',
  };

  // --------------------------------------------------------------------------
  // Diagnostic HUD Overlay
  // --------------------------------------------------------------------------
  (function initDebugHUD() {
    const panel = document.getElementById('panel-camera');
    if (!panel) return;

    const wrap = document.createElement('div');
    wrap.id = 'sb-debug-hud';
    wrap.style.cssText = `
      display: none;
      margin-top: 10px;
      background: #0f1117;
      border: 1px solid #2d3748;
      border-radius: 8px;
      padding: 12px 16px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      color: #cbd5e1;
      line-height: 1.7;
    `;

    wrap.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid #1e293b; padding-bottom:6px;">
        <span style="font-weight:700; color:#818cf8; font-size:12px;">🛠️ PIPELINE TELEMETRY</span>
        <span style="color:#64748b; font-size:10px;">Toggle with 'D' key</span>
      </div>
      <div id="sb-debug-metrics" style="display:grid; grid-template-columns: 1fr 1fr; gap:6px 16px;"></div>
    `;

    panel.appendChild(wrap);
  })();

  function toggleDebugHUD() {
    debugVisible = !debugVisible;
    const hud = document.getElementById('sb-debug-hud');
    if (hud) hud.style.display = debugVisible ? 'block' : 'none';
  }

  function updateDebugHUD() {
    if (!debugVisible) return;
    const metrics = document.getElementById('sb-debug-metrics');
    if (!metrics) return;

    const row = (label, val, col = '#e2e8f0') =>
      `<div style="display:flex; justify-content:space-between;">
        <span style="color:#64748b;">${label}:</span>
        <span style="color:${col}; font-weight:600;">${val}</span>
      </div>`;

    metrics.innerHTML = [
      row('Camera', DS.cameraReady ? 'READY' : 'OFF', DS.cameraReady ? '#4ade80' : '#f87171'),
      row('MediaPipe', DS.mediaPipeReady ? 'READY' : (DS.mediaPipeError || 'INIT'), DS.mediaPipeReady ? '#4ade80' : '#f87171'),
      row('Hands Detected', `${DS.handsDetected}`, DS.handsDetected > 0 ? '#38bdf8' : '#64748b'),
      row('History Window', `${DS.historyLength}/${HISTORY_WINDOW_FRAMES}`, '#fbbf24'),
      row('Active Hand Frames', `${DS.activeFramesInHist}`, '#a78bfa'),
      row('Capture Mode', DS.samplingMode, '#c084fc'),
      row('API Status', DS.apiStatus, DS.apiStatus === 'SUCCESS' ? '#4ade80' : '#f87171'),
      row('Latency', DS.apiLatencyMs ? `${DS.apiLatencyMs}ms` : '—', '#94a3b8'),
      row('Last Prediction', DS.lastPrediction, '#f1f5f9'),
      row('Frames Processed', `${DS.totalFrames}`, '#64748b'),
    ].join('');
  }

  // Keyboard shortcut 'D' to toggle debug panel
  window.addEventListener('keydown', (e) => {
    if (e.key === 'd' || e.key === 'D') {
      if (document.activeElement?.tagName !== 'INPUT') {
        toggleDebugHUD();
      }
    }
    // Spacebar to trigger "Sign Now" if camera is running
    if (e.code === 'Space' && isCameraRunning && !isRecordingGesture) {
      if (document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        startGestureCapture();
      }
    }
  });

  if (btnToggleDebug) {
    btnToggleDebug.addEventListener('click', toggleDebugHUD);
  }

  // --------------------------------------------------------------------------
  // Navigation
  // --------------------------------------------------------------------------
  navRecognize.addEventListener('click', () => switchMainView('recognize'));
  navLearn.addEventListener('click',     () => switchMainView('learn'));
  if (btnTeachMeTop) btnTeachMeTop.addEventListener('click', triggerTeachMeSomething);
  if (btnTeachMeChip) btnTeachMeChip.addEventListener('click', triggerTeachMeSomething);

  function switchMainView(viewName) {
    if (viewName === 'recognize') {
      navRecognize.classList.add('active');
      navRecognize.setAttribute('aria-current', 'page');
      navLearn.classList.remove('active');
      navLearn.removeAttribute('aria-current');
      viewRecognize.classList.add('active');
      viewLearn.classList.remove('active');
    } else {
      navLearn.classList.add('active');
      navLearn.setAttribute('aria-current', 'page');
      navRecognize.classList.remove('active');
      navRecognize.removeAttribute('aria-current');
      viewLearn.classList.add('active');
      viewRecognize.classList.remove('active');
      if (isCameraRunning && !targetPracticeGloss) stopCamera();
      if (vocabList.children.length === 0) {
        fetchVocabulary();
        loadDemonstration('APPLE');
      }
    }
  }

  tabCamera.addEventListener('click', () => switchRecognizeTab('camera'));
  tabUpload.addEventListener('click', () => switchRecognizeTab('upload'));

  function switchRecognizeTab(mode) {
    if (mode === 'camera') {
      tabCamera.classList.add('active');    tabCamera.setAttribute('aria-selected', 'true');
      tabUpload.classList.remove('active'); tabUpload.setAttribute('aria-selected', 'false');
      panelCamera.classList.add('active');  panelUpload.classList.remove('active');
    } else {
      tabUpload.classList.add('active');    tabUpload.setAttribute('aria-selected', 'true');
      tabCamera.classList.remove('active'); tabCamera.setAttribute('aria-selected', 'false');
      panelUpload.classList.add('active');  panelCamera.classList.remove('active');
      if (isCameraRunning) stopCamera();
    }
  }

  // --------------------------------------------------------------------------
  // Camera Management
  // --------------------------------------------------------------------------
  btnStartCamera.addEventListener('click', startCamera);
  btnToggleCamera.addEventListener('click', () => isCameraRunning ? stopCamera() : startCamera());
  btnCaptureSign.addEventListener('click', () => {
    if (isCameraRunning && !isRecordingGesture) startGestureCapture();
  });

  async function startCamera() {
    try {
      updateCameraStatus('Connecting camera…', 'neutral');
      DS.cameraReady = false;

      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      webcamFeed.srcObject = mediaStream;
      await webcamFeed.play();

      landmarkCanvas.width  = webcamFeed.videoWidth  || 640;
      landmarkCanvas.height = webcamFeed.videoHeight || 480;

      DS.cameraReady = true;
      console.log('[SB] Camera connected:', landmarkCanvas.width, 'x', landmarkCanvas.height);

      if (!handsDetector) initMediaPipeHands();

      isCameraRunning        = true;
      rollingHistory         = [];
      gestureRecordedFrames  = [];
      isRecordingGesture     = false;
      DS.totalFrames         = 0;

      cameraPromptOverlay.classList.add('hidden');
      btnToggleCamera.disabled = false;
      btnCaptureSign.disabled  = false;
      btnToggleCamera.querySelector('span').textContent = 'Stop Camera';
      updateCameraStatus('Ready to sign — click Sign Now or hold Space', 'active');

      processCameraLoop();

    } catch (err) {
      DS.cameraReady = false;
      console.error('[SB] Camera error:', err);
      updateCameraStatus('Camera error: ' + err.message, 'error');
      alert('Camera access failed: ' + err.message);
    }
  }

  function stopCamera() {
    isCameraRunning    = false;
    isRecordingGesture = false;
    if (gestureProgressInterval) { clearInterval(gestureProgressInterval); gestureProgressInterval = null; }
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    webcamFeed.srcObject = null;
    if (canvasCtx) canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    rollingHistory        = [];
    gestureRecordedFrames = [];
    Object.assign(DS, {
      cameraReady: false, handsDetected: 0, historyLength: 0,
      activeFramesInHist: 0, apiStatus: 'IDLE', samplingMode: 'IDLE'
    });
    updateDebugHUD();

    captureProgressContainer.classList.add('hidden');
    btnCaptureSign.classList.remove('recording');
    btnCaptureSign.disabled  = true;
    cameraPromptOverlay.classList.remove('hidden');
    btnToggleCamera.disabled = true;
    btnToggleCamera.querySelector('span').textContent = 'Start Camera';
    updateCameraStatus('Camera off', 'neutral');
    resetPredictionUI();
  }

  function updateCameraStatus(text, state = 'neutral') {
    cameraStatusText.textContent = text;
    state === 'active'
      ? cameraStatusBadge.classList.add('active')
      : cameraStatusBadge.classList.remove('active');
  }

  // --------------------------------------------------------------------------
  // MediaPipe Hands Initialization
  // --------------------------------------------------------------------------
  function initMediaPipeHands() {
    if (typeof Hands === 'undefined') {
      DS.mediaPipeError = 'CDN not loaded';
      DS.mediaPipeReady = false;
      updateDebugHUD();
      console.error('[SB] MediaPipe Hands script not loaded from CDN.');
      return;
    }

    try {
      handsDetector = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });
      handsDetector.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      handsDetector.onResults(onHandResults);
      DS.mediaPipeReady = true;
      DS.mediaPipeError = '';
      updateDebugHUD();
      console.log('[SB] MediaPipe Hands initialized successfully.');
    } catch (e) {
      DS.mediaPipeError = e.message;
      DS.mediaPipeReady = false;
      updateDebugHUD();
      console.error('[SB] MediaPipe init failed:', e);
    }
  }

  async function processCameraLoop() {
    if (!isCameraRunning) return;
    if (webcamFeed.readyState >= 2 && handsDetector) {
      try {
        await handsDetector.send({ image: webcamFeed });
      } catch (e) {
        // Suppress frame drop errors during resize/teardown
      }
    }
    animationFrameId = requestAnimationFrame(processCameraLoop);
  }

  // --------------------------------------------------------------------------
  // MediaPipe Frame Callback
  // --------------------------------------------------------------------------
  /**
   * Processes each camera frame:
   * 1. Extracts landmarks [42, 3] (21 landmarks x 2 hands, sorted left-to-right by wrist X)
   * 2. Preserves zeros when hands are not present (vital for GRU recurrent state)
   * 3. Buffers into gestureRecordedFrames (if interactive capture active)
   * 4. Buffers into rollingHistory (continuous sliding window of ~60 frames)
   * 5. Triggers continuous streaming inference every 650ms if active motion detected
   */
  function onHandResults(results) {
    if (!canvasCtx) return;
    canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    DS.totalFrames++;

    // Default zero-initialized frame [42, 3]
    const frameLandmarks = Array.from({ length: 42 }, () => [0.0, 0.0, 0.0]);
    let hasHands = false;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      hasHands = true;

      // Extract and sort detected hands by wrist X (ascending: leftmost hand first)
      const detectedHands = results.multiHandLandmarks.map((handLms) => ({
        wristX: handLms[0].x,
        coords: handLms.map(lm => [lm.x, lm.y, lm.z]),
        rawHand: handLms,
      }));

      detectedHands.sort((a, b) => a.wristX - b.wristX);

      detectedHands.slice(0, 2).forEach((hand, idx) => {
        const offset = idx * 21;
        hand.coords.forEach((coord, lmIdx) => {
          frameLandmarks[offset + lmIdx] = coord;
        });
        drawHandOverlay(hand.rawHand);
      });

      DS.handsDetected = Math.min(detectedHands.length, 2);
    } else {
      DS.handsDetected = 0;
    }

    // ── Mode 1: Interactive Timed Gesture Capture ────────────────────────────
    if (isRecordingGesture) {
      gestureRecordedFrames.push(frameLandmarks);
    }

    // ── Mode 2: Continuous Rolling History Buffer (~60 frames / 2.0s) ────────
    // We intentionally push ALL frames (including zeros) to capture natural rest/sign/rest motion
    rollingHistory.push(frameLandmarks);
    if (rollingHistory.length > HISTORY_WINDOW_FRAMES) {
      rollingHistory.shift();
    }

    // Update telemetry counters
    DS.historyLength = rollingHistory.length;
    DS.activeFramesInHist = rollingHistory.filter(f => !isZeroFrame(f)).length;
    updateDebugHUD();

    // ── Autonomous Continuous Stream Inference ──────────────────────────────
    const now = Date.now();
    if (
      !isRecordingGesture &&
      !isInferencing &&
      rollingHistory.length >= 32 &&
      now - lastContinuousInference >= CONTINUOUS_INTERVAL_MS
    ) {
      // Check if there is meaningful signing activity in the window (at least 6 frames with hands)
      if (DS.activeFramesInHist >= 6) {
        lastContinuousInference = now;
        DS.samplingMode = 'CONTINUOUS (60f→32f)';
        const sampledSequence = resampleTo32Frames(rollingHistory);
        executeSequenceInference(sampledSequence, false);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Uniform Temporal Resampling (matches np.linspace(0, total-1, 32))
  // --------------------------------------------------------------------------
  /**
   * Resamples an arbitrary sequence of frames (e.g. 50-70 frames from a 2-second capture)
   * into exactly 32 frames uniformly distributed across time.
   * This identically replicates Python's get_sampled_frame_indices() used during training.
   */
  function resampleTo32Frames(frames) {
    if (!frames || frames.length === 0) {
      return Array.from({ length: SEQUENCE_LENGTH }, () => Array.from({ length: 42 }, () => [0, 0, 0]));
    }
    if (frames.length === 1) {
      return Array.from({ length: SEQUENCE_LENGTH }, () => frames[0]);
    }

    const result = [];
    const total = frames.length;
    for (let i = 0; i < SEQUENCE_LENGTH; i++) {
      const idx = Math.min(Math.round((i * (total - 1)) / (SEQUENCE_LENGTH - 1)), total - 1);
      result.push(frames[idx]);
    }
    return result;
  }

  function isZeroFrame(frame) {
    // Check if both hand slots are all zeros
    return frame[0][0] === 0 && frame[0][1] === 0 && frame[21][0] === 0 && frame[21][1] === 0;
  }

  // --------------------------------------------------------------------------
  // Interactive Sign Gesture Capture (2.0-Second Timed Capture)
  // --------------------------------------------------------------------------
  function startGestureCapture() {
    if (!isCameraRunning || isRecordingGesture) return;

    isRecordingGesture    = true;
    gestureRecordedFrames = [];
    gestureStartTime      = Date.now();
    DS.samplingMode       = 'RECORDING (2.0s)';

    // Update UI elements
    btnCaptureSign.classList.add('recording');
    btnCaptureSign.querySelector('span').textContent = 'Recording…';
    captureProgressContainer.classList.remove('hidden');
    captureProgressFill.style.width = '0%';
    captureStatusText.textContent   = 'Perform sign clearly…';
    updateCameraStatus('Recording your sign gesture (2.0s)…', 'active');
    setProcessingUI('Capturing sign gesture…');

    // Smooth visual progress bar animation
    const intervalMs = 20;
    gestureProgressInterval = setInterval(() => {
      const elapsed = Date.now() - gestureStartTime;
      const progress = Math.min(100, (elapsed / GESTURE_CAPTURE_MS) * 100);
      captureProgressFill.style.width = `${progress}%`;
      const remainingSec = Math.max(0, ((GESTURE_CAPTURE_MS - elapsed) / 1000)).toFixed(1);
      captureTimeText.textContent = `${remainingSec}s`;

      if (elapsed >= GESTURE_CAPTURE_MS) {
        clearInterval(gestureProgressInterval);
        gestureProgressInterval = null;
        finishGestureCapture();
      }
    }, intervalMs);
  }

  function finishGestureCapture() {
    isRecordingGesture = false;
    btnCaptureSign.classList.remove('recording');
    btnCaptureSign.querySelector('span').textContent = '⚡ Sign Now (Space)';
    captureProgressContainer.classList.add('hidden');
    updateCameraStatus('Analyzing sign gesture…', 'active');
    setProcessingUI('Analyzing captured gesture…');

    const totalCaptured = gestureRecordedFrames.length;
    console.log(`[SB] Gesture capture completed: ${totalCaptured} frames recorded.`);

    // Resample the full 2.0-second gesture into exactly 32 frames
    const sampledSequence = resampleTo32Frames(gestureRecordedFrames);
    DS.samplingMode = `GESTURE (${totalCaptured}f→32f)`;
    executeSequenceInference(sampledSequence, true);
  }

  // --------------------------------------------------------------------------
  // Inference API Execution
  // --------------------------------------------------------------------------
  async function executeSequenceInference(sequence32, isExplicitCapture = false) {
    if (sequence32.length !== 32 || sequence32[0].length !== 42) {
      console.error('[SB] Invalid sequence dimensions before API dispatch:', sequence32.length, sequence32[0]?.length);
      return;
    }

    isInferencing = true;
    DS.apiStatus = 'SENDING';
    updateDebugHUD();

    const t0 = Date.now();
    try {
      const res = await fetch('/api/predict/sequence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: sequence32, top_k: 5 }),
      });

      DS.apiLatencyMs = Date.now() - t0;

      if (!res.ok) {
        DS.apiStatus = `ERR_${res.status}`;
        updateDebugHUD();
        console.error('[SB] Inference error:', res.status);
        return;
      }

      const data = await res.json();
      DS.apiStatus      = 'SUCCESS';
      DS.lastPrediction = `${data.predicted_gloss} (${data.confidence.toFixed(1)}%)`;
      updateDebugHUD();

      if (isExplicitCapture) {
        // Direct capture: update immediately with full confidence presentation
        updatePredictionUI(data);
        updateCameraStatus(`Identified: ${data.predicted_gloss} (${data.confidence.toFixed(1)}%)`, 'active');
      } else {
        // Continuous stream: apply temporal smoothing to prevent UI flicker
        applyTemporalSmoothing(data);
      }

    } catch (err) {
      DS.apiLatencyMs = Date.now() - t0;
      DS.apiStatus = 'FETCH_ERR';
      updateDebugHUD();
      console.error('[SB] Network inference error:', err);
    } finally {
      isInferencing = false;
    }
  }

  // --------------------------------------------------------------------------
  // Temporal Smoothing (for continuous live streaming mode)
  // --------------------------------------------------------------------------
  function applyTemporalSmoothing(result) {
    const rawTop1 = result.predicted_gloss;
    const rawConf = result.confidence;

    if (rawTop1 === lastStableGloss) {
      consecutiveMatches++;
    } else {
      consecutiveMatches = 1;
      lastStableGloss    = rawTop1;
    }

    // Update hero prediction if confirmed consecutively OR if model is very confident
    if (consecutiveMatches >= CONSECUTIVE_REQUIRED || rawConf >= 60.0) {
      updatePredictionUI(result);
    } else {
      renderTop5List(result.top_k);
    }
  }

  // --------------------------------------------------------------------------
  // Canvas Hand Skeleton Overlay
  // --------------------------------------------------------------------------
  function drawHandOverlay(landmarks) {
    const W = landmarkCanvas.width, H = landmarkCanvas.height;
    const CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],
      [9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],[0,17],
    ];

    canvasCtx.strokeStyle = 'rgba(99, 102, 241, 0.65)';
    canvasCtx.lineWidth = 2.5;

    CONNECTIONS.forEach(([i, j]) => {
      canvasCtx.beginPath();
      canvasCtx.moveTo(landmarks[i].x * W, landmarks[i].y * H);
      canvasCtx.lineTo(landmarks[j].x * W, landmarks[j].y * H);
      canvasCtx.stroke();
    });

    landmarks.forEach((lm, idx) => {
      canvasCtx.beginPath();
      canvasCtx.arc(lm.x * W, lm.y * H, idx === 0 ? 4.5 : 3.0, 0, 2 * Math.PI);
      canvasCtx.fillStyle = idx === 0 ? '#ec4899' : '#6366f1';
      canvasCtx.fill();
    });
  }

  // --------------------------------------------------------------------------
  // Video Upload Handlers
  // --------------------------------------------------------------------------
  dropzone.addEventListener('click',  () => videoFileInput.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files?.[0]) handleVideoFile(e.dataTransfer.files[0]);
  });
  videoFileInput.addEventListener('change', e => { if (e.target.files?.[0]) handleVideoFile(e.target.files[0]); });
  btnReupload.addEventListener('click', () => {
    uploadPreviewContainer.classList.add('hidden');
    dropzone.classList.remove('hidden');
    videoFileInput.value = '';
    resetPredictionUI();
  });

  async function handleVideoFile(file) {
    uploadVideoPlayer.src = URL.createObjectURL(file);
    uploadFileName.textContent = file.name;
    dropzone.classList.add('hidden');
    uploadPreviewContainer.classList.remove('hidden');
    setProcessingUI('Extracting landmarks & analyzing sign…');

    const fd = new FormData();
    fd.append('file', file);
    fd.append('top_k', 5);

    try {
      const res = await fetch('/api/predict/video', { method: 'POST', body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Inference failed'); }
      updatePredictionUI(await res.json());
    } catch (err) {
      console.error('[SB] Video processing error:', err);
      alert('Video processing failed: ' + err.message);
      resetPredictionUI();
    }
  }

  // --------------------------------------------------------------------------
  // Learn & Practice Module
  // --------------------------------------------------------------------------
  learnSearchInput.addEventListener('input', e => {
    clearTimeout(searchDebounceTimeout);
    searchDebounceTimeout = setTimeout(() => fetchVocabulary(e.target.value), 200);
  });

  document.querySelectorAll('.chip-btn[data-gloss]').forEach(btn => {
    btn.addEventListener('click', () => loadDemonstration(btn.getAttribute('data-gloss')));
  });

  async function fetchVocabulary(query = '') {
    try {
      const url = query ? `/api/vocabulary?q=${encodeURIComponent(query)}&limit=60` : `/api/vocabulary?limit=60`;
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      renderVocabList(data.items);
      vocabCountBadge.textContent = `${data.total_matches} Signs`;
    } catch (err) { console.warn('[SB] Vocab fetch error:', err); }
  }

  function renderVocabList(items) {
    if (!items?.length) {
      vocabList.innerHTML = `<p style="padding:16px; color:var(--text-muted); text-align:center;">No signs match your search.</p>`;
      return;
    }
    vocabList.innerHTML = items.map(item => `
      <div class="vocab-item ${item.gloss === currentLearnGloss ? 'selected' : ''}"
           data-gloss="${item.gloss}" role="button" tabindex="0">
        <span class="vocab-name">${item.gloss}</span>
        <span class="vocab-tier-tag">${item.tier}</span>
      </div>`).join('');
    vocabList.querySelectorAll('.vocab-item').forEach(el =>
      el.addEventListener('click', () => loadDemonstration(el.getAttribute('data-gloss')))
    );
  }

  async function loadDemonstration(gloss) {
    currentLearnGloss = gloss;
    vocabList.querySelectorAll('.vocab-item').forEach(el =>
      el.classList.toggle('selected', el.getAttribute('data-gloss') === gloss)
    );
    demoGlossTitle.textContent = gloss;
    btnTrySign.querySelector('span').textContent = `Try Signing ${gloss}`;
    try {
      const res = await fetch(`/api/demonstration/${encodeURIComponent(gloss)}`);
      if (!res.ok) return;
      const data = await res.json();
      demoTierBadge.textContent = data.tier || 'Standard';
      if (data.has_demonstration && data.video_url) {
        demoVideoPlayer.src = data.video_url;
        demoVideoPlayer.classList.remove('hidden');
        demoVideoPlaceholder.classList.add('hidden');
        demoVideoPlayer.play().catch(() => {});
      } else {
        demoVideoPlayer.pause();
        demoVideoPlayer.classList.add('hidden');
        demoVideoPlaceholder.classList.remove('hidden');
        demoPlaceholderText.textContent = data.message || 'No demonstration video available.';
      }
    } catch (err) { console.warn('[SB] Demo load error:', err); }
  }

  async function triggerTeachMeSomething() {
    try {
      const res = await fetch('/api/teach-me');
      if (!res.ok) return;
      const data = await res.json();
      switchMainView('learn');
      loadDemonstration(data.gloss);
    } catch (err) { console.warn('[SB] Teach-me error:', err); }
  }

  btnTrySign.addEventListener('click', () => {
    targetPracticeGloss = currentLearnGloss;
    practiceTargetGloss.textContent = targetPracticeGloss;
    practiceTargetBanner.classList.remove('hidden');
    switchMainView('recognize');
    switchRecognizeTab('camera');
    if (!isCameraRunning) startCamera();
  });

  if (btnCancelPractice) {
    btnCancelPractice.addEventListener('click', () => {
      targetPracticeGloss = null;
      practiceTargetBanner.classList.add('hidden');
      resetPredictionUI();
    });
  }

  // --------------------------------------------------------------------------
  // Prediction Presentation UI
  // --------------------------------------------------------------------------
  function updatePredictionUI(result) {
    const gloss = result.predicted_gloss;
    const conf  = result.confidence;

    if (primaryGloss.textContent !== gloss) {
      primaryGloss.classList.remove('update-pulse');
      void primaryGloss.offsetWidth;
      primaryGloss.classList.add('update-pulse');
    }
    primaryGloss.textContent      = gloss;
    primaryConfidence.textContent = `${conf.toFixed(1)}%`;

    if (targetPracticeGloss) {
      const isTop1 = gloss.toUpperCase() === targetPracticeGloss.toUpperCase();
      const isTop5 = result.top_k.some(i => i.gloss.toUpperCase() === targetPracticeGloss.toUpperCase());
      if (isTop1 && conf >= 70) {
        confidenceBadge.className = 'confidence-badge high';
        confidenceBadge.textContent = 'Great Job!';
        confidenceAssessment.textContent = `🎉 Excellent! That looks like ${targetPracticeGloss}.`;
      } else if (isTop1 || isTop5) {
        confidenceBadge.className = 'confidence-badge medium';
        confidenceBadge.textContent = 'Getting Close!';
        confidenceAssessment.textContent = `💡 Good effort! ${targetPracticeGloss} detected in top candidates.`;
      } else {
        confidenceBadge.className = 'confidence-badge neutral';
        confidenceBadge.textContent = 'Try Again';
        confidenceAssessment.textContent = `🔁 Not recognized as ${targetPracticeGloss}. Try watching the demo again!`;
      }
    } else {
      if (conf >= 60) {
        confidenceBadge.className = 'confidence-badge high';
        confidenceBadge.textContent = 'High confidence';
        confidenceAssessment.textContent = 'Very strong recognition match';
      } else if (conf >= 25) {
        confidenceBadge.className = 'confidence-badge medium';
        confidenceBadge.textContent = 'Moderate confidence';
        confidenceAssessment.textContent = 'Good match — check alternatives';
      } else {
        confidenceBadge.className = 'confidence-badge neutral';
        confidenceBadge.textContent = 'Low confidence';
        confidenceAssessment.textContent = 'Sign clearly and complete the full motion';
      }
    }

    // Auto-add high-confidence signs to Sentence Builder
    if (sbAutoAddToggle && sbAutoAddToggle.checked && conf >= 45.0) {
      addWordToSentence(gloss, true);
    }

    renderTop5List(result.top_k);
  }

  function renderTop5List(items) {
    if (!items?.length) return;
    top5List.innerHTML = items.map((item, i) => {
      const prob   = item.confidence.toFixed(1);
      const rank   = String(i + 1).padStart(2, '0');
      const target = targetPracticeGloss && item.gloss.toUpperCase() === targetPracticeGloss.toUpperCase();
      return `
        <div class="ranking-item ${target ? 'target-match' : ''}" role="listitem" data-gloss="${item.gloss}" title="Click to add '${item.gloss}' to sentence">
          <span class="rank-num">${rank}</span>
          <span class="rank-gloss" title="${item.gloss}">${item.gloss}</span>
          <div class="rank-bar-bg" aria-hidden="true">
            <div class="rank-bar-fill" style="width:${Math.max(prob, 2)}%"></div>
          </div>
          <span class="rank-prob">${prob}%</span>
        </div>`;
    }).join('');

    // Clicking any candidate in Top-5 list immediately selects it and adds to sentence
    top5List.querySelectorAll('.ranking-item:not(.empty)').forEach(el => {
      el.addEventListener('click', () => {
        const selectedGloss = el.getAttribute('data-gloss');
        if (selectedGloss) {
          primaryGloss.textContent = selectedGloss;
          primaryGloss.classList.remove('update-pulse');
          void primaryGloss.offsetWidth;
          primaryGloss.classList.add('update-pulse');
          addWordToSentence(selectedGloss, false);
          updateCameraStatus(`Selected '${selectedGloss}' → Added to sentence`, 'active');
        }
      });
    });
  }

  function setProcessingUI(msg) {
    primaryGloss.textContent       = 'Analyzing…';
    primaryConfidence.textContent  = '—';
    confidenceBadge.className      = 'confidence-badge medium';
    confidenceBadge.textContent    = 'Processing';
    confidenceAssessment.textContent = msg;
  }

  function resetPredictionUI() {
    primaryGloss.textContent       = '—';
    primaryConfidence.textContent  = '0.0%';
    confidenceBadge.className      = 'confidence-badge neutral';
    confidenceBadge.textContent    = 'Awaiting Input';
    confidenceAssessment.textContent = targetPracticeGloss
      ? `Perform ${targetPracticeGloss} for the camera`
      : 'Show a sign or click Sign Now';
    top5List.innerHTML = [1,2,3,4,5].map((n, i) => `
      <div class="ranking-item empty"><span class="rank-num">0${n}</span>
        <span class="rank-gloss">${i === 0 ? 'Waiting for sign input…' : '—'}</span>
        <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:0%"></div></div>
        <span class="rank-prob">0.0%</span></div>`).join('');
    lastStableGloss    = '—';
    consecutiveMatches = 0;
  }

  // --------------------------------------------------------------------------
  // Sentence Builder & Speech Synthesis Logic
  // --------------------------------------------------------------------------
  let sentenceWords     = [];
  let lastAutoAddedWord = '';
  let lastAutoAddedTime = 0;

  function addWordToSentence(word, isAuto = false) {
    if (!word || word === '—' || word === 'Analyzing…' || word.startsWith('Waiting')) return;
    const cleanWord = word.trim().toUpperCase();
    const now = Date.now();

    // Prevent adding same sign continuously in live streaming within 2.5 seconds
    if (isAuto) {
      if (cleanWord === lastAutoAddedWord && (now - lastAutoAddedTime < 2500)) {
        return;
      }
    }

    sentenceWords.push(cleanWord);
    lastAutoAddedWord = cleanWord;
    lastAutoAddedTime = now;
    renderSentenceUI();
  }

  function removeWordFromSentence(index) {
    if (index >= 0 && index < sentenceWords.length) {
      sentenceWords.splice(index, 1);
      renderSentenceUI();
    }
  }

  function backspaceSentence() {
    if (sentenceWords.length > 0) {
      sentenceWords.pop();
      renderSentenceUI();
    }
  }

  function clearSentence() {
    sentenceWords = [];
    lastAutoAddedWord = '';
    renderSentenceUI();
  }

  function renderSentenceUI() {
    if (!sbWordsContainer) return;
    const count = sentenceWords.length;
    if (sbWordCount) sbWordCount.textContent = `${count} word${count === 1 ? '' : 's'}`;

    if (count === 0) {
      if (sbEmptyState) sbEmptyState.classList.remove('hidden');
      sbWordsContainer.innerHTML = '';
    } else {
      if (sbEmptyState) sbEmptyState.classList.add('hidden');
      sbWordsContainer.innerHTML = sentenceWords.map((word, idx) => `
        <span class="sb-word-chip" data-idx="${idx}">
          <span>${word}</span>
          <button class="sb-word-remove" title="Remove word" aria-label="Remove ${word}">&times;</button>
        </span>
      `).join('');

      sbWordsContainer.querySelectorAll('.sb-word-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const chip = e.target.closest('.sb-word-chip');
          const idx = parseInt(chip.getAttribute('data-idx'), 10);
          removeWordFromSentence(idx);
        });
      });
    }
  }

  function speakSentence() {
    if (sentenceWords.length === 0) {
      updateCameraStatus('Sentence is empty — sign words first!', 'neutral');
      return;
    }
    const text = sentenceWords.join(' ').toLowerCase();
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      utterance.lang = 'en-US';

      if (sbBtnSpeak) sbBtnSpeak.classList.add('speaking');
      utterance.onend = () => { if (sbBtnSpeak) sbBtnSpeak.classList.remove('speaking'); };
      utterance.onerror = () => { if (sbBtnSpeak) sbBtnSpeak.classList.remove('speaking'); };

      window.speechSynthesis.speak(utterance);
      updateCameraStatus(`Speaking: "${text}"`, 'active');
    } else {
      alert('Text-to-speech is not supported in this browser.');
    }
  }

  function copySentence() {
    if (sentenceWords.length === 0) return;
    const text = sentenceWords.join(' ');
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(() => {
        if (sbCopyText) sbCopyText.textContent = 'Copied!';
        setTimeout(() => { if (sbCopyText) sbCopyText.textContent = 'Copy Text'; }, 1800);
      }).catch(() => {});
    }
  }

  // Sentence Builder event listeners
  if (sbBtnSpeak)      sbBtnSpeak.addEventListener('click', speakSentence);
  if (sbBtnCopy)       sbBtnCopy.addEventListener('click', copySentence);
  if (sbBtnBackspace)  sbBtnBackspace.addEventListener('click', backspaceSentence);
  if (sbBtnClear)      sbBtnClear.addEventListener('click', clearSentence);
  if (sbBtnAddCurrent) {
    sbBtnAddCurrent.addEventListener('click', () => {
      const current = primaryGloss?.textContent;
      if (current && current !== '—' && current !== 'Analyzing…') {
        addWordToSentence(current, false);
      }
    });
  }

  // Quick high-accuracy vocabulary chips
  if (sbQuickChips) {
    sbQuickChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const gloss = chip.getAttribute('data-gloss');
        if (gloss) {
          addWordToSentence(gloss, false);
          updateCameraStatus(`Added '${gloss}' to sentence`, 'active');
        }
      });
    });
  }

  // Initial Sentence Builder render
  renderSentenceUI();

}); // end DOMContentLoaded
