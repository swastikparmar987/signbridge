/**
 * SignBridge v2.2 — Webcam Pipeline with Diagnostic Debug Panel
 *
 * DATA FLOW:
 *   getUserMedia → MediaPipe Hands → [42,3] landmarks/frame
 *   → rolling buffer (real frames only, no zero-fill)
 *   → POST /api/predict/sequence { sequence:[32][42][3] }
 *   → FastAPI → normalize → reshape → GRU → prediction
 *
 * DEBUG PANEL (always visible when camera is running):
 *   Camera / MediaPipe / Hands / Buffer / API / Prediction / Latency
 *
 * ONE-SHOT DIAGNOSTIC:
 *   When buffer first reaches 32 frames, also POST to /api/debug/sequence
 *   and render the full backend diagnostic in the debug panel.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const navRecognize        = document.getElementById('nav-recognize');
  const navLearn            = document.getElementById('nav-learn');
  const btnTeachMeTop       = document.getElementById('btn-teach-me-top');
  const btnTeachMeChip      = document.getElementById('btn-teach-me-chip');
  const viewRecognize       = document.getElementById('view-recognize');
  const viewLearn           = document.getElementById('view-learn');
  const tabCamera           = document.getElementById('tab-camera');
  const tabUpload           = document.getElementById('tab-upload');
  const panelCamera         = document.getElementById('panel-camera');
  const panelUpload         = document.getElementById('panel-upload');
  const webcamFeed          = document.getElementById('webcam-feed');
  const landmarkCanvas      = document.getElementById('landmark-canvas');
  const canvasCtx           = landmarkCanvas ? landmarkCanvas.getContext('2d') : null;
  const btnStartCamera      = document.getElementById('btn-start-camera');
  const btnToggleCamera     = document.getElementById('btn-toggle-camera');
  const cameraPromptOverlay = document.getElementById('camera-prompt-overlay');
  const cameraStatusBadge   = document.getElementById('camera-status-badge');
  const cameraStatusText    = document.getElementById('camera-status-text');
  const dropzone            = document.getElementById('dropzone');
  const videoFileInput      = document.getElementById('video-file-input');
  const uploadPreviewContainer = document.getElementById('upload-preview-container');
  const uploadVideoPlayer   = document.getElementById('upload-video-player');
  const uploadFileName      = document.getElementById('upload-file-name');
  const btnReupload         = document.getElementById('btn-reupload');
  const primaryGloss        = document.getElementById('primary-gloss');
  const primaryConfidence   = document.getElementById('primary-confidence');
  const confidenceAssessment= document.getElementById('confidence-assessment');
  const confidenceBadge     = document.getElementById('confidence-badge');
  const top5List            = document.getElementById('top5-list');
  const learnSearchInput    = document.getElementById('learn-search-input');
  const vocabList           = document.getElementById('vocab-list');
  const vocabCountBadge     = document.getElementById('vocab-count-badge');
  const demoGlossTitle      = document.getElementById('demo-gloss-title');
  const demoTierBadge       = document.getElementById('demo-tier-badge');
  const demoVideoPlayer     = document.getElementById('demo-video-player');
  const demoVideoPlaceholder= document.getElementById('demo-video-placeholder');
  const demoPlaceholderText = document.getElementById('demo-placeholder-text');
  const btnTrySign          = document.getElementById('btn-try-sign');
  const practiceTargetBanner= document.getElementById('practice-target-banner');
  const practiceTargetGloss = document.getElementById('practice-target-gloss');
  const btnCancelPractice   = document.getElementById('btn-cancel-practice');

  // ── Pipeline state ────────────────────────────────────────────────────────
  let isCameraRunning   = false;
  let mediaStream       = null;
  let handsDetector     = null;
  let animationFrameId  = null;
  let targetPracticeGloss = null;
  let currentLearnGloss   = 'APPLE';
  let searchDebounceTimeout = null;

  // Landmark buffer — real frames only
  const SEQUENCE_LENGTH     = 32;
  let   landmarkBuffer      = [];
  let   lastInferenceTime   = 0;
  const INFERENCE_INTERVAL_MS = 500;   // 2 API calls/sec max

  // Smoothing
  let lastStableGloss     = '—';
  let consecutiveMatches  = 0;
  const CONSECUTIVE_REQUIRED = 2;

  // One-shot diagnostic flag
  let diagnosticFired = false;

  // ── Debug panel state ─────────────────────────────────────────────────────
  const DS = {
    cameraReady:     false,
    mediaPipeReady:  false,
    mediaPipeError:  '',
    handsDetected:   0,
    bufferSize:      0,
    apiStatus:       'IDLE',
    lastPrediction:  '—',
    apiLatencyMs:    0,
    frameCount:      0,       // total frames processed
    lastPayloadShape: '—',
    diagReport:      null,    // full diagnostic from /api/debug/sequence
  };

  // ── Inject debug panel into DOM ───────────────────────────────────────────
  (function buildDebugPanel() {
    // Panel below the camera card
    const panelCameraEl = document.getElementById('panel-camera');
    if (!panelCameraEl) return;

    const wrap = document.createElement('div');
    wrap.id = 'sb-debug-wrap';
    wrap.style.cssText = `
      margin-top: 8px;
      background: #0d0d14;
      border: 1px solid #333;
      border-radius: 10px;
      padding: 10px 14px;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
      font-size: 11.5px;
      line-height: 1.8;
      color: #ccc;
    `;

    const header = document.createElement('div');
    header.style.cssText = 'display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;';
    header.innerHTML = `
      <span style="color:#7c6fff; font-weight:700; font-size:12px; letter-spacing:1px;">
        🛠 PIPELINE DEBUG
      </span>
      <button id="btn-clear-diag" style="
        background:none; border:1px solid #444; border-radius:4px;
        color:#888; font-size:10px; padding:2px 8px; cursor:pointer;">
        Clear
      </button>
    `;
    wrap.appendChild(header);

    const table = document.createElement('div');
    table.id = 'sb-debug-table';
    wrap.appendChild(table);

    const diagBlock = document.createElement('pre');
    diagBlock.id = 'sb-debug-diag';
    diagBlock.style.cssText = `
      display:none;
      margin-top:10px;
      padding:8px;
      background:#0a0a18;
      border-radius:6px;
      color:#00ff99;
      font-size:10px;
      max-height:300px;
      overflow-y:auto;
      white-space:pre-wrap;
      word-break:break-all;
    `;
    wrap.appendChild(diagBlock);

    panelCameraEl.appendChild(wrap);

    document.getElementById('btn-clear-diag').addEventListener('click', () => {
      diagnosticFired = false;
      DS.diagReport = null;
      diagBlock.style.display = 'none';
      diagBlock.textContent = '';
      refreshDebug();
    });
  })();

  function refreshDebug() {
    const table = document.getElementById('sb-debug-table');
    if (!table) return;

    const row = (label, value, color) =>
      `<div style="display:flex; gap:10px;">
        <span style="color:#666; min-width:120px;">${label}</span>
        <span style="color:${color || '#eee'}; font-weight:600;">${value}</span>
      </div>`;

    const camColor   = DS.cameraReady    ? '#00ff99' : '#ff4444';
    const mpColor    = DS.mediaPipeReady ? '#00ff99' : '#ff4444';
    const bufColor   = DS.bufferSize >= SEQUENCE_LENGTH ? '#00ff99' : '#ffcc00';
    const apiColors  = { IDLE:'#888', SENDING:'#ffcc00', SUCCESS:'#00ff99', ERROR:'#ff4444' };
    const apiColor   = apiColors[DS.apiStatus] || '#888';

    table.innerHTML = [
      row('Camera',      DS.cameraReady    ? '✅ READY' : '❌ ' + (DS.cameraReady === false && !isCameraRunning ? 'OFF' : 'ERROR'), camColor),
      row('MediaPipe',   DS.mediaPipeReady ? '✅ READY' : '❌ ' + (DS.mediaPipeError || 'NOT INIT'), mpColor),
      row('Hands',       `${DS.handsDetected} detected`, DS.handsDetected > 0 ? '#00ff99' : '#888'),
      row('Buffer',      `${DS.bufferSize} / ${SEQUENCE_LENGTH}`, bufColor),
      row('Frames seen', String(DS.frameCount), '#aaa'),
      row('Payload',     DS.lastPayloadShape, '#aaa'),
      row('API',         DS.apiStatus, apiColor),
      row('Latency',     DS.apiLatencyMs ? `${DS.apiLatencyMs} ms` : '—', '#aaa'),
      row('Prediction',  DS.lastPrediction, '#fff'),
    ].join('');
  }

  function showDiagReport(data) {
    const block = document.getElementById('sb-debug-diag');
    if (!block) return;
    block.style.display = 'block';
    block.textContent = JSON.stringify(data, null, 2);
  }

  // Initial render
  refreshDebug();

  // ── Navigation ────────────────────────────────────────────────────────────
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

  // ── Tab switching ─────────────────────────────────────────────────────────
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

  // ── Camera start/stop ──────────────────────────────────────────────────────
  btnStartCamera.addEventListener('click', startCamera);
  btnToggleCamera.addEventListener('click', () => isCameraRunning ? stopCamera() : startCamera());

  async function startCamera() {
    try {
      updateCameraStatus('Connecting camera…', 'neutral');
      DS.cameraReady = false;
      refreshDebug();

      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      webcamFeed.srcObject = mediaStream;
      await webcamFeed.play();

      landmarkCanvas.width  = webcamFeed.videoWidth  || 640;
      landmarkCanvas.height = webcamFeed.videoHeight || 480;

      DS.cameraReady = true;
      refreshDebug();
      console.log('[SB] Camera started. Resolution:', landmarkCanvas.width, 'x', landmarkCanvas.height);

      if (!handsDetector) initMediaPipeHands();

      isCameraRunning = true;
      landmarkBuffer  = [];
      DS.bufferSize   = 0;
      DS.frameCount   = 0;
      diagnosticFired = false;

      cameraPromptOverlay.classList.add('hidden');
      btnToggleCamera.disabled = false;
      btnToggleCamera.querySelector('span').textContent = 'Stop Camera';
      updateCameraStatus('Looking for a sign…', 'active');

      processCameraLoop();

    } catch (err) {
      DS.cameraReady = false;
      refreshDebug();
      console.error('[SB] Camera error:', err.name, err.message);
      updateCameraStatus('Camera error: ' + err.message, 'error');
      alert('Camera access failed: ' + err.message);
    }
  }

  function stopCamera() {
    isCameraRunning = false;
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    webcamFeed.srcObject = null;
    if (canvasCtx) canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    landmarkBuffer = [];
    Object.assign(DS, {
      cameraReady: false, mediaPipeReady: false, handsDetected: 0,
      bufferSize: 0, apiStatus: 'IDLE', frameCount: 0,
    });
    refreshDebug();

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

  // ── MediaPipe Hands ───────────────────────────────────────────────────────
  function initMediaPipeHands() {
    if (typeof Hands === 'undefined') {
      DS.mediaPipeError = 'CDN script not loaded';
      DS.mediaPipeReady = false;
      refreshDebug();
      console.error('[SB] MediaPipe Hands CDN script NOT loaded. Check <script> tag in index.html.');
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
      refreshDebug();
      console.log('[SB] MediaPipe Hands initialised OK');
    } catch (e) {
      DS.mediaPipeError = e.message;
      DS.mediaPipeReady = false;
      refreshDebug();
      console.error('[SB] MediaPipe init failed:', e);
    }
  }

  async function processCameraLoop() {
    if (!isCameraRunning) return;
    if (webcamFeed.readyState >= 2 && handsDetector) {
      try {
        await handsDetector.send({ image: webcamFeed });
      } catch (e) {
        console.error('[SB] handsDetector.send() error:', e.message);
      }
    }
    animationFrameId = requestAnimationFrame(processCameraLoop);
  }

  // ── MediaPipe callback ────────────────────────────────────────────────────
  /**
   * onHandResults — fired once per video frame by MediaPipe.
   *
   * Produces frameLandmarks [42, 3]:
   *   slots  0-20  → leftmost hand (smallest wrist X) or zeros
   *   slots 21-41  → rightmost hand (largest wrist X) or zeros
   *
   * Matches Python extract_hand_sequence() ordering exactly.
   * Only real frames (hasHands) are added to the buffer.
   */
  function onHandResults(results) {
    if (!canvasCtx) return;
    canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    DS.frameCount++;

    // Build zero-initialised [42, 3] frame
    const frameLandmarks = Array.from({ length: 42 }, () => [0.0, 0.0, 0.0]);
    let hasHands = false;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      hasHands = true;

      // Pair each hand with its wrist X for sorting
      const detectedHands = results.multiHandLandmarks.map((handLms, rawIdx) => ({
        wristX: handLms[0].x,
        coords: handLms.map(lm => [lm.x, lm.y, lm.z]),
        rawHand: handLms,
        rawIdx,
      }));

      // Sort left-to-right (ascending wrist X) — matches Python pipeline
      detectedHands.sort((a, b) => a.wristX - b.wristX);

      detectedHands.slice(0, 2).forEach((hand, idx) => {
        const offset = idx * 21;   // 0 or 21
        hand.coords.forEach((coord, lmIdx) => {
          frameLandmarks[offset + lmIdx] = coord;
        });
        drawHandOverlay(hand.rawHand);
      });

      DS.handsDetected = Math.min(detectedHands.length, 2);

      // Log every 60 frames to avoid console spam
      if (DS.frameCount % 60 === 1) {
        const h = detectedHands[0];
        console.log(
          `[SB] Frame #${DS.frameCount} | hands=${DS.handsDetected}` +
          ` | hand0.wristX=${h.wristX.toFixed(4)}` +
          ` | hand0.x=${frameLandmarks[0][0].toFixed(4)}` +
          ` y=${frameLandmarks[0][1].toFixed(4)}` +
          ` z=${frameLandmarks[0][2].toFixed(4)}`
        );
      }
    } else {
      DS.handsDetected = 0;
    }

    // ── Buffer management ─────────────────────────────────────────────────
    // ONLY push when a hand is actually detected.
    // Never fill buffer with zero-frames.
    if (hasHands) {
      landmarkBuffer.push(frameLandmarks);
      if (landmarkBuffer.length > SEQUENCE_LENGTH) {
        landmarkBuffer.shift();
      }
    }

    DS.bufferSize = landmarkBuffer.length;
    refreshDebug();

    // ── Status text ───────────────────────────────────────────────────────
    if (hasHands) {
      updateCameraStatus(
        landmarkBuffer.length < SEQUENCE_LENGTH
          ? `Collecting… ${landmarkBuffer.length}/${SEQUENCE_LENGTH} frames`
          : 'Analysing sign…',
        'active'
      );
    } else {
      updateCameraStatus('Show your signing hand…', 'neutral');
    }

    // ── Fire ONE diagnostic capture ───────────────────────────────────────
    // When buffer first hits 32, send to /api/debug/sequence and dump full
    // diagnostic report. This happens only once per camera session.
    if (!diagnosticFired && hasHands && landmarkBuffer.length === SEQUENCE_LENGTH) {
      diagnosticFired = true;
      const snapshot = landmarkBuffer.map(f => f.map(c => [...c])); // deep copy
      console.log('[SB] === DIAGNOSTIC CAPTURE TRIGGERED ===');
      console.log('[SB] Buffer snapshot shape:', snapshot.length, 'x', snapshot[0].length, 'x', snapshot[0][0].length);
      console.log('[SB] Frame[0] landmark[0] (hand0 wrist):', snapshot[0][0]);
      console.log('[SB] Frame[0] landmark[21] (hand1 wrist):', snapshot[0][21]);
      console.log('[SB] Frame[31] landmark[0] (hand0 wrist):', snapshot[31][0]);
      fireDiagnostic(snapshot);
    }

    // ── Regular inference ─────────────────────────────────────────────────
    const now = Date.now();
    if (
      hasHands &&
      landmarkBuffer.length === SEQUENCE_LENGTH &&
      now - lastInferenceTime >= INFERENCE_INTERVAL_MS
    ) {
      lastInferenceTime = now;
      const snapshot = landmarkBuffer.map(f => f.map(c => [...c]));
      sendInference(snapshot);
    }
  }

  // ── Draw hand skeleton overlay ────────────────────────────────────────────
  function drawHandOverlay(landmarks) {
    const W = landmarkCanvas.width, H = landmarkCanvas.height;
    const CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],
      [9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],[0,17],
    ];
    canvasCtx.strokeStyle = 'rgba(79,70,229,0.5)';
    canvasCtx.lineWidth = 2.5;
    CONNECTIONS.forEach(([i, j]) => {
      canvasCtx.beginPath();
      canvasCtx.moveTo(landmarks[i].x * W, landmarks[i].y * H);
      canvasCtx.lineTo(landmarks[j].x * W, landmarks[j].y * H);
      canvasCtx.stroke();
    });
    landmarks.forEach(lm => {
      canvasCtx.beginPath();
      canvasCtx.arc(lm.x * W, lm.y * H, 3.5, 0, 2 * Math.PI);
      canvasCtx.fillStyle = '#4F46E5';
      canvasCtx.fill();
    });
  }

  // ── /api/debug/sequence — one-shot diagnostic ─────────────────────────────
  async function fireDiagnostic(buffer) {
    console.log('[SB] Firing /api/debug/sequence diagnostic…');
    DS.apiStatus = 'SENDING (diag)';
    DS.lastPayloadShape = `[${buffer.length}, ${buffer[0].length}, ${buffer[0][0].length}]`;
    refreshDebug();

    try {
      const t0 = Date.now();
      const res = await fetch('/api/debug/sequence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: buffer, top_k: 5 }),
      });
      const latency = Date.now() - t0;

      const data = await res.json();
      DS.apiStatus = res.ok ? 'SUCCESS (diag)' : 'ERROR (diag)';
      DS.apiLatencyMs = latency;
      DS.diagReport = data;
      refreshDebug();
      showDiagReport(data);

      // Log EVERYTHING to console
      console.log('[SB] === DIAGNOSTIC REPORT FROM BACKEND ===');
      console.log('[SB] shape_received    :', data.shape_received);
      console.log('[SB] shape_note        :', data.shape_note);
      console.log('[SB] real_frames       :', data.real_frames, '(non-zero)');
      console.log('[SB] hand0_zero_frames :', data.hand0_zero_frames);
      console.log('[SB] hand1_zero_frames :', data.hand1_zero_frames);
      console.log('[SB] x_range           :', data.x_range);
      console.log('[SB] y_range           :', data.y_range);
      console.log('[SB] z_range           :', data.z_range);
      console.log('[SB] hand_ordering_check:', data.hand_ordering_check);
      if (data.first_frame) {
        console.log('[SB] first_frame.hand0_wrist:', data.first_frame.hand0_wrist_xyz);
        console.log('[SB] first_frame.hand1_wrist:', data.first_frame.hand1_wrist_xyz);
      }
      if (data.normalization) {
        console.log('[SB] norm output_shape    :', data.normalization.output_shape);
        console.log('[SB] norm hand0_wrist_after:', data.normalization.hand0_wrist_after_norm);
        console.log('[SB] norm note            :', data.normalization.note);
      }
      console.log('[SB] === END DIAGNOSTIC ===');

    } catch (err) {
      DS.apiStatus = 'ERROR (diag)';
      refreshDebug();
      console.error('[SB] Diagnostic request failed:', err);
    }
  }

  // ── Regular inference call ────────────────────────────────────────────────
  async function sendInference(buffer) {
    // Validate shape contract before sending
    if (buffer.length !== SEQUENCE_LENGTH || buffer[0].length !== 42 || buffer[0][0].length !== 3) {
      console.error('[SB] SHAPE MISMATCH before API call:',
        buffer.length, 'x', buffer[0]?.length, 'x', buffer[0]?.[0]?.length,
        '— expected 32 x 42 x 3');
      return;
    }

    DS.apiStatus    = 'SENDING';
    DS.lastPayloadShape = `[${buffer.length}, ${buffer[0].length}, ${buffer[0][0].length}]`;
    refreshDebug();

    const t0 = Date.now();
    try {
      const res = await fetch('/api/predict/sequence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: buffer, top_k: 5 }),
      });

      DS.apiLatencyMs = Date.now() - t0;

      if (!res.ok) {
        const errText = await res.text();
        DS.apiStatus = `ERROR ${res.status}`;
        refreshDebug();
        console.error('[SB] /api/predict/sequence error', res.status, errText);
        return;
      }

      const data = await res.json();
      DS.apiStatus      = 'SUCCESS';
      DS.lastPrediction = `${data.predicted_gloss} (${data.confidence.toFixed(1)}%)`;
      refreshDebug();

      console.log('[SB] Prediction:', data.predicted_gloss, '@', data.confidence.toFixed(2) + '%');
      applyTemporalSmoothing(data);

    } catch (err) {
      DS.apiLatencyMs = Date.now() - t0;
      DS.apiStatus = 'FETCH ERROR';
      refreshDebug();
      console.error('[SB] fetch /api/predict/sequence failed:', err);
    }
  }

  // ── Temporal smoothing ────────────────────────────────────────────────────
  function applyTemporalSmoothing(result) {
    const rawTop1      = result.predicted_gloss;
    const rawConf      = result.confidence;

    if (rawTop1 === lastStableGloss) {
      consecutiveMatches++;
    } else {
      consecutiveMatches = 1;
      lastStableGloss    = rawTop1;
    }

    if (consecutiveMatches >= CONSECUTIVE_REQUIRED || rawConf >= 75.0) {
      updatePredictionUI(result);
    } else {
      renderTop5List(result.top_k);
    }
  }

  // ── Upload handling ───────────────────────────────────────────────────────
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
    setProcessingUI('Understanding your sign…');

    const fd = new FormData();
    fd.append('file', file);
    fd.append('top_k', 5);

    try {
      const res = await fetch('/api/predict/video', { method: 'POST', body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Failed'); }
      updatePredictionUI(await res.json());
    } catch (err) {
      console.error('[SB] Video upload error:', err);
      alert('Video processing failed: ' + err.message);
      resetPredictionUI();
    }
  }

  // ── Learn / Practice module ───────────────────────────────────────────────
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
    } catch (err) { console.warn('[SB] vocab fetch error:', err); }
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
    } catch (err) { console.warn('[SB] demo load error:', err); }
  }

  async function triggerTeachMeSomething() {
    try {
      const res = await fetch('/api/teach-me');
      if (!res.ok) return;
      const data = await res.json();
      switchMainView('learn');
      loadDemonstration(data.gloss);
    } catch (err) { console.warn('[SB] teach-me error:', err); }
  }

  // ── Practice mode ─────────────────────────────────────────────────────────
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

  // ── Prediction UI ─────────────────────────────────────────────────────────
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
        confidenceAssessment.textContent = `💡 Good effort! ${targetPracticeGloss} detected in candidates.`;
      } else {
        confidenceBadge.className = 'confidence-badge neutral';
        confidenceBadge.textContent = 'Try Again';
        confidenceAssessment.textContent = '🔁 Not recognised yet. Watch the demo again!';
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
        confidenceBadge.textContent = 'Not sure';
        confidenceAssessment.textContent = 'Sign clearly and hold the pose';
      }
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
        <div class="ranking-item ${target ? 'target-match' : ''}" role="listitem">
          <span class="rank-num">${rank}</span>
          <span class="rank-gloss" title="${item.gloss}">${item.gloss}</span>
          <div class="rank-bar-bg" aria-hidden="true">
            <div class="rank-bar-fill" style="width:${Math.max(prob, 2)}%"></div>
          </div>
          <span class="rank-prob">${prob}%</span>
        </div>`;
    }).join('');
  }

  function setProcessingUI(msg) {
    primaryGloss.textContent       = 'Analysing…';
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
      : 'Show a sign to get started';
    top5List.innerHTML = [1,2,3,4,5].map((n, i) => `
      <div class="ranking-item empty"><span class="rank-num">0${n}</span>
        <span class="rank-gloss">${i === 0 ? 'Waiting for sign input…' : '—'}</span>
        <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:0%"></div></div>
        <span class="rank-prob">0.0%</span></div>`).join('');
    lastStableGloss    = '—';
    consecutiveMatches = 0;
  }

}); // end DOMContentLoaded
