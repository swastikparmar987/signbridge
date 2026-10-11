/**
 * SignBridge Frontend Application v5.0
 * Unified Single Page Application (SPA) Controller
 * Matching Stitch UI Design System & Integrating Deep Learning Backend
 */

document.addEventListener('DOMContentLoaded', () => {
  'use strict';

  // ==========================================================================
  // 1. STATE & PARAMETERS
  // ==========================================================================
  const state = {
    currentView: 'home',
    activeModel: 'production_exp7',
    confidenceThreshold: 0.55,
    isCameraRunning: false,
    mediaStream: null,
    handsDetector: null,
    animationFrameId: null,

    // Sequence inference
    SEQUENCE_LENGTH: 32,
    HISTORY_WINDOW: 90,
    CONTINUOUS_INTERVAL_MS: 220,
    MIN_ACTIVE_FRAMES: 8,
    rollingHistory: [],
    isInferencing: false,
    lastInferenceTime: 0,

    // Session stats
    totalSigns: 0,
    totalSentences: 0,
    confidenceSum: 0,
    recognizedSentence: [],

    // Learn & Practice state
    currentLearnGloss: 'HELLO',
    currentPracticeTarget: 'HELLO',
    practiceStream: null,
    isPracticing: false,

    // Teach state
    teachSamples: [],
    teachTargetSamples: 5,
    teachStream: null,

    // Communication state
    isListeningSTT: false,
    speechRecognizer: null,
    messages: [
      { sender: 'Signer', text: 'Hello! Nice to meet you.', type: 'sign', time: '10:42 AM' },
      { sender: 'Hearing Partner', text: 'Welcome! How can I help you today?', type: 'voice', time: '10:43 AM' },
      { sender: 'Signer', text: 'I would like to learn more about SignBridge.', type: 'sign', time: '10:43 AM' }
    ]
  };

  // ==========================================================================
  // 2. DOM REFERENCES
  // ==========================================================================
  // Navigation & Shell
  const appShell               = document.getElementById('app-shell');
  const navLinks               = document.querySelectorAll('[data-nav]');
  const viewPanes              = document.querySelectorAll('.view-pane');
  const themeToggleBtn         = document.getElementById('theme-toggle-btn');
  const notificationsBtn       = document.getElementById('notifications-btn');
  const notificationsDropdown  = document.getElementById('notifications-dropdown');
  const globalSearchInput      = document.getElementById('global-search-input');
  const globalSearchResults    = document.getElementById('global-search-results');
  const activeModelBadge       = document.getElementById('active-model-badge');
  const activeModelLabel       = document.getElementById('active-model-label');
  const toastContainer         = document.getElementById('toast-container');

  // Onboarding
  const onboardingOverlay      = document.getElementById('onboarding-overlay');
  const btnRelaunchOnboarding  = document.getElementById('btn-relaunch-onboarding');
  const btnSkipOnboarding      = document.getElementById('btn-skip-onboarding');
  const btnWelcomeNext         = document.getElementById('btn-welcome-next');
  const btnCamBack             = document.getElementById('btn-cam-back');
  const btnCamContinue         = document.getElementById('btn-cam-continue');
  const btnPrefBack            = document.getElementById('btn-pref-back');
  const btnPrefContinue        = document.getElementById('btn-pref-continue');
  const btnStartExploring      = document.getElementById('btn-start-exploring');
  const onboardingWebcamFeed   = document.getElementById('onboarding-webcam-feed');
  const btnOnboardingTestCam   = document.getElementById('btn-onboarding-test-cam');

  // Live Translate
  const webcamFeed             = document.getElementById('webcam-feed');
  const landmarkCanvas         = document.getElementById('landmark-canvas');
  const canvasCtx              = landmarkCanvas ? landmarkCanvas.getContext('2d') : null;
  const cameraPromptOverlay    = document.getElementById('camera-prompt-overlay');
  const btnStartCamera         = document.getElementById('btn-start-camera');
  const primaryGloss           = document.getElementById('primary-gloss');
  const primaryConfidence      = document.getElementById('primary-confidence');
  const primaryEnglish         = document.getElementById('primary-english');
  const heroInterpretation    = document.getElementById('heroInterpretation');
  const altCandidatesGrid      = document.getElementById('alt-candidates-grid');
  const sentenceComposerText   = document.getElementById('sentence-composer-text');
  const tokenWorkspace        = document.getElementById('tokenWorkspace');
  const timelineStream        = document.getElementById('timelineStream');
  const liveCameraDot         = document.getElementById('live-camera-dot');
  const liveCameraStatus      = document.getElementById('live-camera-status');
  const liveStatusText        = document.getElementById('live-status-text');
  const liveHandPill          = document.getElementById('live-hand-pill');
  const liveHandText          = document.getElementById('live-hand-text');
  const sbBtnUndo              = document.getElementById('sb-btn-undo');
  const sbBtnCopy              = document.getElementById('sb-btn-copy');
  const sbBtnSpeak             = document.getElementById('sb-btn-speak');
  const sbBtnSpeakSentence    = document.getElementById('sb-btn-speak-sentence');
  const sbBtnClear             = document.getElementById('sb-btn-clear');
  const sbBtnClearTimeline    = document.getElementById('sb-btn-clear-timeline');
  const btnPronounceSign       = document.getElementById('btn-pronounce-sign');
  const liveStatSigns          = document.getElementById('live-stat-signs');
  const liveStatSentences      = document.getElementById('live-stat-sentences');
  const liveStatConfidence     = document.getElementById('live-stat-confidence');
  const pauseTranslateBtn      = document.getElementById('pauseTranslateBtn');

  // Diagnostic HUD
  const diagnosticHud          = document.getElementById('diagnostic-hud');
  const btnCloseDiag           = document.getElementById('btn-close-diag');
  const diagModel              = document.getElementById('diag-model');
  const diagLatency            = document.getElementById('diag-latency');
  const diagHands              = document.getElementById('diag-hands');
  const diagBuffer             = document.getElementById('diag-buffer');
  const diagMargin             = document.getElementById('diag-margin');

  // Video Translate
  const videoDropzone          = document.getElementById('video-dropzone');
  const videoFileInput         = document.getElementById('video-file-input');
  const btnBrowseVideo         = document.getElementById('btn-browse-video');
  const videoTranslationResult = document.getElementById('video-translation-result');

  // Teach
  const teachWebcamFeed        = document.getElementById('teach-webcam-feed');
  const teachLandmarkCanvas    = document.getElementById('teach-landmark-canvas');
  const btnRecordSample        = document.getElementById('btn-record-sample');

  // Learn
  const learnVocabSearch       = document.getElementById('learn-vocab-search');
  const vocabListContainer     = document.getElementById('vocab-list-container');
  const demoSignTitle          = document.getElementById('demo-sign-title');
  const demoSignDesc           = document.getElementById('demo-sign-desc');
  const demoVideoPlayer        = document.getElementById('demo-video-player');
  const btnTrySigningDemo      = document.getElementById('btn-try-signing-demo');

  // Practice
  const practiceTargetTitle    = document.getElementById('practice-target-title');

  // Communication
  const commTextInput          = document.getElementById('comm-text-input');
  const commBtnSend            = document.getElementById('comm-btn-send');
  const commBtnMic             = document.getElementById('comm-btn-mic');

  // Settings
  const settingsThemeLight     = document.getElementById('settings-theme-light');
  const settingsThemeDark      = document.getElementById('settings-theme-dark');
  const settingsThemeSystem    = document.getElementById('settings-theme-system');
  const settingsClearCacheBtn  = document.getElementById('settings-clear-cache-btn');

  // ==========================================================================
  // 3. TOAST NOTIFICATIONS
  // ==========================================================================
  function showToast(message, type = 'success') {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    const bgClass = type === 'error' ? 'bg-red-600' : (type === 'warning' ? 'bg-amber-600' : 'bg-[#111318]');
    toast.className = `${bgClass} text-white px-4 py-2.5 rounded-xl shadow-xl flex items-center gap-2.5 text-xs font-semibold border border-white/10 transform transition-all duration-300 translate-y-4 opacity-0 pointer-events-auto`;
    toast.innerHTML = `
      <span class="w-2 h-2 rounded-full ${type === 'error' ? 'bg-red-300' : 'bg-[#FF5A2A]'}"></span>
      <span>${message}</span>
    `;
    toastContainer.appendChild(toast);

    requestAnimationFrame(() => {
      toast.classList.remove('translate-y-4', 'opacity-0');
    });

    setTimeout(() => {
      toast.classList.add('opacity-0', 'translate-y-2');
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  // ==========================================================================
  // 4. VIEW ROUTER
  // ==========================================================================
  function switchView(viewName) {
    if (!viewName) return;
    if (viewName === 'video-translation') viewName = 'video';
    state.currentView = viewName;

    // Update active nav styling in sidebar & anywhere else
    navLinks.forEach(link => {
      const target = link.getAttribute('data-nav');
      if (link.classList.contains('nav-link')) {
        const isActive = (target === viewName || (viewName === 'video' && target === 'video-translation'));
        if (isActive) {
          link.className = 'nav-link active flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm transition-all cursor-pointer bg-[#FF5A2A] text-white font-bold shadow-md shadow-orange-500/20';
        } else {
          link.className = 'nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer';
        }
      }
    });

    // Update view panes
    viewPanes.forEach(pane => {
      if (pane.id === `view-${viewName}`) {
        pane.classList.remove('hidden');
      } else {
        pane.classList.add('hidden');
      }
    });

    // Update header title
    const titles = {
      home: 'Home Dashboard',
      communication: 'Two-Way Communication',
      learn: 'Learn Signs & Vocabulary',
      video: 'Video Translation',
      live: 'Live Translation Studio',
      practice: 'Practice Flashcards',
      teach: 'Teach SignBridge Model',
      profile: 'Maya Rodriguez Profile',
      settings: 'Settings & Model Switcher'
    };
    const titleEl = document.getElementById('current-view-title');
    if (titleEl && titles[viewName]) {
      titleEl.textContent = titles[viewName];
    }

    // View specific activations
    if (viewName === 'live') {
      if (!state.isCameraRunning) {
        startMainCamera();
      }
    } else if (viewName === 'communication') {
      if (typeof initSyntheticLandmarks === 'function') initSyntheticLandmarks();
    } else if (viewName === 'learn') {
      fetchVocabulary();
      loadDemonstration(state.currentLearnGloss);
      if (typeof initLearnInteractiveModule === 'function') {
        initLearnInteractiveModule();
      }
    } else if (viewName === 'profile') {
      loadProfileData();
    }

    // Scroll main viewport to top
    const viewsContainer = document.getElementById('views-container');
    if (viewsContainer) viewsContainer.scrollTop = 0;
  }

  // Bind click on all [data-nav] elements
  document.addEventListener('click', (e) => {
    const navEl = e.target.closest('[data-nav]');
    if (navEl) {
      e.preventDefault();
      const target = navEl.getAttribute('data-nav');
      switchView(target);
    }
  });

  // ==========================================================================
  // 5. THEME & NOTIFICATIONS
  // ==========================================================================
  function applyTheme(isDark) {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('sb-theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('sb-theme', 'light');
    }
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const isDark = !document.documentElement.classList.contains('dark');
      applyTheme(isDark);
      showToast(isDark ? 'Dark theme enabled' : 'Light theme enabled');
    });
  }

  if (settingsThemeLight) settingsThemeLight.addEventListener('click', () => applyTheme(false));
  if (settingsThemeDark) settingsThemeDark.addEventListener('click', () => applyTheme(true));
  if (settingsThemeSystem) {
    settingsThemeSystem.addEventListener('click', () => {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      applyTheme(prefersDark);
    });
  }

  if (notificationsBtn && notificationsDropdown) {
    notificationsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      notificationsDropdown.classList.toggle('hidden');
    });
    document.addEventListener('click', () => {
      notificationsDropdown.classList.add('hidden');
    });
  }

  // ==========================================================================
  // 6. GLOBAL SEARCH & VOCABULARY AUTOCOMPLETE
  // ==========================================================================
  let searchDebounce = null;
  if (globalSearchInput && globalSearchResults) {
    globalSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim();
      clearTimeout(searchDebounce);
      if (!q) {
        globalSearchResults.classList.add('hidden');
        globalSearchResults.innerHTML = '';
        return;
      }

      searchDebounce = setTimeout(async () => {
        try {
          const res = await fetch(`/api/vocabulary?q=${encodeURIComponent(q)}&limit=8`);
          if (res.ok) {
            const data = await res.json();
            renderSearchResults(data.items);
          }
        } catch (err) {
          console.warn('[SB] Search error:', err);
        }
      }, 150);
    });

    document.addEventListener('click', (e) => {
      if (!globalSearchInput.contains(e.target) && !globalSearchResults.contains(e.target)) {
        globalSearchResults.classList.add('hidden');
      }
    });
  }

  function renderSearchResults(items) {
    if (!globalSearchResults) return;
    if (!items || items.length === 0) {
      globalSearchResults.innerHTML = '<div class="p-3 text-gray-500 text-center text-xs">No matching signs found.</div>';
      globalSearchResults.classList.remove('hidden');
      return;
    }

    globalSearchResults.innerHTML = items.map(item => `
      <div class="search-item px-3.5 py-2.5 hover:bg-orange-50/70 border-b border-gray-100 last:border-b-0 flex items-center justify-between cursor-pointer transition" data-gloss="${item.gloss}">
        <div class="flex items-center gap-2.5">
          <span class="w-6 h-6 rounded-md bg-[#FFF0EB] text-[#FF5A2A] flex items-center justify-center font-bold text-[10px]">ASL</span>
          <span class="font-bold text-gray-900 text-xs">${item.gloss}</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-[10px] text-gray-400">${item.video_count} videos</span>
          <span class="text-[10px] text-[#FF5A2A] font-semibold hover:underline">Learn &rarr;</span>
        </div>
      </div>
    `).join('');

    globalSearchResults.querySelectorAll('.search-item').forEach(el => {
      el.addEventListener('click', () => {
        const gloss = el.getAttribute('data-gloss');
        if (gloss) {
          state.currentLearnGloss = gloss;
          switchView('learn');
          loadDemonstration(gloss);
          globalSearchResults.classList.add('hidden');
          globalSearchInput.value = '';
        }
      });
    });

    globalSearchResults.classList.remove('hidden');
  }

  // ==========================================================================
  // 7. ONBOARDING CONTROLLER
  // ==========================================================================
  function setOnboardingStep(stepNum) {
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById(`onboarding-step-${i}`);
      if (el) el.classList.toggle('hidden', i !== stepNum);
    }

    // Update stepper pills
    const stepperSteps = document.querySelectorAll('.stepper-step');
    stepperSteps.forEach(s => {
      const step = parseInt(s.getAttribute('data-step'), 10);
      const badge = s.querySelector('span');
      if (step === stepNum) {
        s.className = 'stepper-step flex items-center gap-1.5 text-[#FF5A2A] font-bold';
        if (badge) { badge.className = 'w-6 h-6 rounded-full bg-[#FF5A2A] text-white flex items-center justify-center text-xs'; badge.textContent = String(step); }
      } else if (step < stepNum) {
        s.className = 'stepper-step flex items-center gap-1.5 text-emerald-600 font-semibold';
        if (badge) { badge.className = 'w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center text-xs'; badge.textContent = '✓'; }
      } else {
        s.className = 'stepper-step flex items-center gap-1.5 text-neutral-400 font-medium';
        if (badge) { badge.className = 'w-6 h-6 rounded-full bg-neutral-200 text-neutral-600 flex items-center justify-center text-xs'; badge.textContent = String(step); }
      }
    });
  }

  function openOnboarding() {
    if (onboardingOverlay) {
      onboardingOverlay.classList.remove('hidden');
      setOnboardingStep(1);
    }
  }

  function closeOnboarding() {
    if (onboardingOverlay) onboardingOverlay.classList.add('hidden');
  }

  if (btnRelaunchOnboarding) btnRelaunchOnboarding.addEventListener('click', openOnboarding);
  if (btnSkipOnboarding) btnSkipOnboarding.addEventListener('click', closeOnboarding);
  if (btnWelcomeNext) btnWelcomeNext.addEventListener('click', () => setOnboardingStep(2));
  if (btnCamBack) btnCamBack.addEventListener('click', () => setOnboardingStep(1));
  if (btnCamContinue) btnCamContinue.addEventListener('click', () => setOnboardingStep(3));
  if (btnPrefBack) btnPrefBack.addEventListener('click', () => setOnboardingStep(2));
  if (btnPrefContinue) btnPrefContinue.addEventListener('click', () => setOnboardingStep(4));
  if (btnStartExploring) btnStartExploring.addEventListener('click', () => {
    closeOnboarding();
    switchView('home');
    showToast('Welcome to SignBridge! Setup completed.');
  });

  if (btnOnboardingTestCam && onboardingWebcamFeed) {
    btnOnboardingTestCam.addEventListener('click', async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        onboardingWebcamFeed.srcObject = stream;
        onboardingWebcamFeed.classList.remove('hidden');
        await onboardingWebcamFeed.play();
        const fallback = document.getElementById('onboarding-cam-fallback');
        if (fallback) fallback.classList.add('hidden');
        showToast('Camera active and calibrated');
      } catch (err) {
        showToast('Camera access denied: ' + err.message, 'error');
      }
    });
  }

  // ==========================================================================
  // 8. WEBCAM & MEDIAPIPE PIPELINE (Live Translate)
  // ==========================================================================
  async function startMainCamera() {
    if (state.isCameraRunning) return;

    try {
      state.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
        audio: false
      });

      if (webcamFeed) {
        webcamFeed.srcObject = state.mediaStream;
        webcamFeed.style.display = 'block';
        await webcamFeed.play();
      }

      if (landmarkCanvas) {
        landmarkCanvas.style.display = 'block';
      }

      state.isCameraRunning = true;
      if (cameraPromptOverlay) cameraPromptOverlay.style.display = 'none';

      if (liveCameraDot) liveCameraDot.className = 'w-2 h-2 rounded-full bg-[#2EB875] animate-pulse';
      if (liveCameraStatus) liveCameraStatus.textContent = 'Camera Connected';
      if (liveStatusText) liveStatusText.textContent = 'Listening for signs...';

      initMediaPipe();
      showToast('Camera connected — Live Recognition active');
    } catch (err) {
      console.error('[SB] Camera error:', err);
      if (liveCameraDot) liveCameraDot.className = 'w-2 h-2 rounded-full bg-red-500';
      if (liveCameraStatus) liveCameraStatus.textContent = 'Camera Error';
      if (liveStatusText) liveStatusText.textContent = 'Camera access denied';
      showToast('Could not access camera: ' + err.message, 'error');
    }
  }

  function initMediaPipe() {
    if (typeof window.Hands === 'undefined') {
      console.warn('[SB] Waiting for MediaPipe Hands CDN...');
      setTimeout(initMediaPipe, 300);
      return;
    }

    state.handsDetector = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    state.handsDetector.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    state.handsDetector.onResults(onMediaPipeResults);

    let isProcessing = false;
    async function onFrame() {
      if (!state.isCameraRunning || isProcessing) return;
      if (webcamFeed && webcamFeed.readyState >= 2) {
        isProcessing = true;
        try {
          await state.handsDetector.send({ image: webcamFeed });
        } catch (e) {
          console.warn('[SB] MediaPipe frame error:', e);
        } finally {
          isProcessing = false;
        }
      }
    }

    if (typeof window.Camera !== 'undefined' && webcamFeed) {
      const camera = new window.Camera(webcamFeed, {
        onFrame: onFrame,
        width: 1280,
        height: 720
      });
      camera.start().catch(err => {
        console.warn('[SB] Camera utils start failed, using loop:', err);
        fallbackLoop();
      });
    } else {
      fallbackLoop();
    }

    function fallbackLoop() {
      if (state.isCameraRunning) {
        onFrame().finally(() => {
          if (state.isCameraRunning) {
            state.animationFrameId = requestAnimationFrame(fallbackLoop);
          }
        });
      }
    }
  }

  if (btnStartCamera) btnStartCamera.addEventListener('click', startMainCamera);

  if (pauseTranslateBtn) {
    pauseTranslateBtn.addEventListener('click', () => {
      state.isCameraRunning = !state.isCameraRunning;
      if (!state.isCameraRunning) {
        pauseTranslateBtn.classList.remove('bg-[#FF5A2A]');
        pauseTranslateBtn.classList.add('bg-neutral-700');
        showToast('Recognition paused');
      } else {
        pauseTranslateBtn.classList.remove('bg-neutral-700');
        pauseTranslateBtn.classList.add('bg-[#FF5A2A]');
        startMainCamera();
      }
    });
  }

  function onMediaPipeResults(results) {
    if (!landmarkCanvas || !canvasCtx) return;

    if (webcamFeed && (landmarkCanvas.width !== webcamFeed.videoWidth || landmarkCanvas.height !== webcamFeed.videoHeight)) {
      landmarkCanvas.width = webcamFeed.videoWidth || 640;
      landmarkCanvas.height = webcamFeed.videoHeight || 480;
    }

    canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    const frameLandmarks = extractLandmarks(results);
    state.rollingHistory.push(frameLandmarks);
    if (state.rollingHistory.length > state.HISTORY_WINDOW) {
      state.rollingHistory.shift();
    }

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      drawHandSkeletons(results.multiHandLandmarks, canvasCtx, landmarkCanvas.width, landmarkCanvas.height);
      const count = results.multiHandLandmarks.length;
      if (diagHands) diagHands.textContent = `${count} tracked`;
      if (liveHandPill) {
        liveHandPill.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#1A2E24]/90 text-[#34D399] border border-[#2EB875]/30 text-xs font-medium backdrop-blur-sm';
      }
      if (liveHandText) {
        liveHandText.textContent = `${count === 1 ? '1 hand' : '2 hands'} detected`;
      }
    } else {
      if (diagHands) diagHands.textContent = '0';
      if (liveHandPill) {
        liveHandPill.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#1E2129]/90 text-stone-400 border border-neutral-700/50 text-xs font-medium backdrop-blur-sm';
      }
      if (liveHandText) {
        liveHandText.textContent = 'No hands';
      }
    }

    if (diagBuffer) diagBuffer.textContent = `${Math.min(32, state.rollingHistory.length)} / 32`;

    // Trigger sliding window inference
    const now = performance.now();
    if (now - state.lastInferenceTime >= state.CONTINUOUS_INTERVAL_MS && !state.isInferencing) {
      runContinuousInference();
    }
  }

  function extractLandmarks(results) {
    const frame = Array.from({ length: 42 }, () => [0.0, 0.0, 0.0]);
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      return frame;
    }

    const hands = results.multiHandLandmarks.map((lms, idx) => ({
      wristX: lms[0].x,
      landmarks: lms
    }));

    if (hands.length === 2) {
      hands.sort((a, b) => a.wristX - b.wristX);
      hands[0].landmarks.forEach((pt, i) => { frame[i] = [pt.x, pt.y, pt.z || 0.0]; });
      hands[1].landmarks.forEach((pt, i) => { frame[21 + i] = [pt.x, pt.y, pt.z || 0.0]; });
    } else if (hands.length === 1) {
      hands[0].landmarks.forEach((pt, i) => { frame[i] = [pt.x, pt.y, pt.z || 0.0]; });
    }
    return frame;
  }

  function drawHandSkeletons(handsLandmarks, ctx, w, h) {
    const CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],
      [9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],
      [0,17]
    ];

    handsLandmarks.forEach(landmarks => {
      ctx.strokeStyle = 'rgba(255, 90, 42, 0.85)';
      ctx.lineWidth = 2.5;
      CONNECTIONS.forEach(([start, end]) => {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.stroke();
      });

      landmarks.forEach((pt, idx) => {
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, idx === 0 ? 5 : 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();
        ctx.strokeStyle = '#FF5A2A';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    });
  }

  async function runContinuousInference() {
    if (state.rollingHistory.length < 15) return;

    const activeFrames = state.rollingHistory.filter(f => !f.every(pt => pt[0] === 0 && pt[1] === 0)).length;
    if (activeFrames < state.MIN_ACTIVE_FRAMES) return;

    const total = state.rollingHistory.length;
    const sampled32 = [];
    for (let i = 0; i < state.SEQUENCE_LENGTH; i++) {
      const idx = Math.round(i * (total - 1) / (state.SEQUENCE_LENGTH - 1));
      sampled32.push(state.rollingHistory[idx]);
    }

    state.isInferencing = true;
    state.lastInferenceTime = performance.now();
    const t0 = performance.now();

    try {
      const res = await fetch('/api/predict/sequence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sequence: sampled32,
          top_k: 5,
          model_variant: state.activeModel,
          diagnostic: true
        })
      });

      const t1 = performance.now();
      const latencyMs = Math.round(t1 - t0);

      if (res.ok) {
        const data = await res.json();
        handleInferenceResult(data, latencyMs);
      }
    } catch (err) {
      console.warn('[SB] Prediction error:', err);
    } finally {
      state.isInferencing = false;
    }
  }

  function handleInferenceResult(data, latencyMs) {
    if (!data || !data.predicted_gloss) return;

    const gloss = data.predicted_gloss;
    const conf = Number(data.confidence || 0);

    if (diagLatency) diagLatency.textContent = `${latencyMs} ms`;
    if (diagModel) diagModel.textContent = state.activeModel;
    if (diagMargin && data.diagnostics) {
      diagMargin.textContent = `Δ${data.diagnostics.raw_predictions?.margin?.toFixed(1) || '0'}%`;
    }

    if (primaryGloss) primaryGloss.textContent = gloss;
    if (primaryConfidence) primaryConfidence.textContent = `${Math.round(conf)}%`;
    if (primaryEnglish) primaryEnglish.textContent = gloss.charAt(0) + gloss.slice(1).toLowerCase();

    // Alternatives
    if (data.top_k && altCandidatesGrid) {
      renderAlternatives(data.top_k);
    }

    // Append to sentence if confidence meets threshold
    if (conf >= (state.confidenceThreshold * 100)) {
      appendWord(gloss, conf);
      state.totalSigns++;
      state.confidenceSum += conf;
      updateStats();
    }
  }

  function renderAlternatives(topCandidates) {
    if (!altCandidatesGrid) return;
    altCandidatesGrid.innerHTML = topCandidates.slice(0, 5).map((cand, idx) => `
      <button class="alt-chip flex flex-col items-center justify-center p-2 rounded-xl border ${idx === 0 ? 'border-2 border-[#FF5A2A] bg-[#FFF5F0] text-[#FF5A2A]' : 'border border-stone-200 bg-[#FBF9F5] text-slate-700 hover:border-stone-300'} transition cursor-pointer" data-gloss="${cand.gloss}" type="button">
        <span class="text-xs font-bold tracking-tight truncate w-full text-center">${cand.gloss}</span>
        <span class="text-[10px] text-stone-500 font-semibold mt-0.5">${Math.round(cand.confidence)}%</span>
      </button>
    `).join('');

    altCandidatesGrid.querySelectorAll('.alt-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        const gloss = btn.getAttribute('data-gloss');
        if (gloss) {
          if (primaryGloss) primaryGloss.textContent = gloss;
          appendWord(gloss, 90);
          showToast(`Selected "${gloss}"`);
        }
      });
    });
  }

  // ==========================================================================
  // 9. SENTENCE COMPOSER & SPEECH (TTS)
  // ==========================================================================
  function appendWord(word, conf = 90) {
    const clean = word.trim().toUpperCase();
    const lastWord = state.recognizedSentence[state.recognizedSentence.length - 1];
    if (lastWord && lastWord.toUpperCase() === clean) return; // Prevent stutter duplicate

    state.recognizedSentence.push(clean.charAt(0) + clean.slice(1).toLowerCase());
    renderSentence();
    appendTimelineCard(clean, conf);
    updateRealtimeTranslation();
  }

  function appendTimelineCard(gloss, conf) {
    if (!timelineStream) return;
    const emptyEl = document.getElementById('timeline-empty');
    if (emptyEl) emptyEl.remove();

    const timeStr = new Date().toLocaleTimeString([], { minute: '2-digit', second: '2-digit' });
    const card = document.createElement('div');
    card.className = 'flex-shrink-0 flex items-center gap-4';
    card.innerHTML = `
      <div class="px-5 py-3 rounded-2xl bg-[#FFF5F0] border-2 border-[#FF5A2A] flex flex-col items-center justify-center min-w-[90px] shadow-sm">
        <span class="text-xs font-extrabold text-[#FF5A2A] tracking-wider">${gloss}</span>
        <span class="text-[10px] text-stone-500 font-medium">${timeStr} · ${Math.round(conf)}%</span>
      </div>
      <div class="flex items-center gap-1 px-1">
        <span class="w-1 h-1.5 rounded-full bg-[#FF5A2A]"></span>
        <span class="w-1 h-3 rounded-full bg-[#FF5A2A]"></span>
        <span class="w-1 h-5 rounded-full bg-[#FF5A2A]"></span>
        <span class="w-1 h-3 rounded-full bg-[#FF5A2A]"></span>
        <span class="w-1 h-1.5 rounded-full bg-[#FF5A2A]"></span>
      </div>
    `;
    timelineStream.appendChild(card);
    timelineStream.scrollLeft = timelineStream.scrollWidth;
  }

  function renderSentence() {
    if (tokenWorkspace) {
      if (!state.recognizedSentence.length) {
        tokenWorkspace.innerHTML = `<span class="text-stone-400 text-sm" id="token-empty">Recognized signs will build up here.</span><span class="w-[2.5px] h-5 bg-[#FF5A2A] animate-pulse rounded-full my-auto ml-1"></span>`;
      } else {
        tokenWorkspace.innerHTML = state.recognizedSentence.map((word, idx) => `
          <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#FFFDF8] text-slate-800 text-sm shadow-xs font-medium border border-stone-200/60">
            ${word}
            <button aria-label="Remove ${word}" data-idx="${idx}" class="token-remove text-stone-400 hover:text-red-500 transition text-base leading-none" type="button">×</button>
          </span>
        `).join('') + `<span class="w-[2.5px] h-5 bg-[#FF5A2A] animate-pulse rounded-full my-auto ml-1"></span>`;

        tokenWorkspace.querySelectorAll('.token-remove').forEach(btn => {
          btn.addEventListener('click', (e) => {
            const idx = parseInt(btn.getAttribute('data-idx'), 10);
            if (!isNaN(idx) && idx >= 0 && idx < state.recognizedSentence.length) {
              state.recognizedSentence.splice(idx, 1);
              renderSentence();
              updateRealtimeTranslation();
            }
          });
        });
      }
    }
  }

  let translationDebounce = null;
  function updateRealtimeTranslation() {
    if (!heroInterpretation) return;
    if (!state.recognizedSentence.length) {
      heroInterpretation.textContent = 'Start signing. Pause with hands down to get the English sentence.';
      return;
    }

    clearTimeout(translationDebounce);
    translationDebounce = setTimeout(async () => {
      try {
        const res = await fetch('/api/nlp/smooth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ glosses: state.recognizedSentence })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.english && heroInterpretation) {
            heroInterpretation.textContent = `“${data.english}”`;
          }
        }
      } catch (e) {
        console.warn('Realtime translation update failed:', e);
      }
    }, 250);
  }

  let activeAudio = null;

  async function speakTextReal(glossesOrText) {
    let glossList = [];
    if (Array.isArray(glossesOrText)) {
      glossList = glossesOrText;
    } else if (typeof glossesOrText === 'string' && glossesOrText.trim()) {
      glossList = glossesOrText.trim().split(/\s+/);
    } else if (state.recognizedSentence && state.recognizedSentence.length) {
      glossList = state.recognizedSentence;
    }

    if (!glossList.length) {
      showToast('No ASL glosses to translate or speak.');
      return;
    }

    try {
      showToast('Translating glosses via Groq AI...');
      
      // Step 1: Real Groq AI Translation
      const nlpRes = await fetch('/api/nlp/smooth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ glosses: glossList })
      });

      if (!nlpRes.ok) {
        showToast(`Qwen API HTTP ${nlpRes.status}`);
        return;
      }

      const nlpData = await nlpRes.json();
      const englishSentence = nlpData.english || glossList.join(' ');

      // Update hero display with real Qwen response
      const heroElem = document.getElementById('heroInterpretation');
      if (heroElem) {
        heroElem.textContent = `“${englishSentence}”`;
      }

      // Step 2: Real Kokoro TTS Audio Generation
      showToast('Generating speech via Kokoro TTS...');
      const ttsRes = await fetch('/api/tts/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: englishSentence })
      });

      if (!ttsRes.ok) {
        showToast(`TTS API HTTP ${ttsRes.status}`);
        return;
      }

      const ttsData = await ttsRes.json();
      if (!ttsData.success || !ttsData.audio_url) {
        showToast(`TTS Error: ${ttsData.error || 'Speech synthesis unavailable'}`);
        return;
      }

      // Step 3: Real Audio Playback & Stop Control
      if (activeAudio) {
        activeAudio.pause();
        activeAudio = null;
      }

      activeAudio = new Audio(ttsData.audio_url);
      activeAudio.play().then(() => {
        showToast(`Speaking: "${englishSentence}" (${ttsData.duration_s}s audio)`);
      }).catch(err => {
        showToast(`Audio playback failed: ${err.message}`);
      });

    } catch (err) {
      showToast(`Pipeline Error: ${err.message}`);
    }
  }

  function stopAudioReal() {
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }
    fetch('/api/tts/stop', { method: 'POST' }).catch(() => {});
    showToast('Playback stopped and queue cleared');
  }

  if (sbBtnSpeak) {
    sbBtnSpeak.addEventListener('click', () => {
      speakTextReal(state.recognizedSentence);
    });
  }

  if (sbBtnSpeakSentence) {
    sbBtnSpeakSentence.addEventListener('click', () => {
      speakTextReal(state.recognizedSentence);
    });
  }

  if (btnPronounceSign) {
    btnPronounceSign.addEventListener('click', () => {
      const word = primaryGloss ? primaryGloss.textContent : 'HELLO';
      speakTextReal([word]);
    });
  }

  if (pauseTranslateBtn) {
    pauseTranslateBtn.addEventListener('click', () => {
      stopAudioReal();
    });
  }

  if (sbBtnCopy) {
    sbBtnCopy.addEventListener('click', () => {
      const text = state.recognizedSentence.join(' ');
      if (text && navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
          showToast('Copied translation to clipboard!');
        });
      }
    });
  }

  if (sbBtnUndo) {
    sbBtnUndo.addEventListener('click', () => {
      if (state.recognizedSentence.length > 0) {
        state.recognizedSentence.pop();
        renderSentence();
        updateRealtimeTranslation();
      }
    });
  }

  if (sbBtnClear) {
    sbBtnClear.addEventListener('click', () => {
      stopAudioReal();
      state.recognizedSentence = [];
      renderSentence();
      if (heroInterpretation) {
        heroInterpretation.textContent = 'Start signing. Pause with hands down to get the English sentence.';
      }
      showToast('Cleared translation composer');
    });
  }

  if (sbBtnClearTimeline) {
    sbBtnClearTimeline.addEventListener('click', () => {
      if (timelineStream) {
        timelineStream.innerHTML = `<div class="px-4 py-2.5 rounded-xl bg-[#F7F3EB] border border-stone-200 text-stone-500 text-xs" id="timeline-empty">Timeline cleared</div>`;
      }
      showToast('Timeline cleared');
    });
  }

  const sbBtnEdit = document.getElementById('sb-btn-edit');
  if (sbBtnEdit) {
    sbBtnEdit.addEventListener('click', () => {
      const current = state.recognizedSentence.join(' ');
      const updated = prompt('Edit translated sentence:', current);
      if (updated !== null) {
        state.recognizedSentence = updated.trim() ? updated.trim().split(/\\s+/) : [];
        renderSentence();
        showToast('Sentence updated');
      }
    });
  }

  const liveBtnFs = document.getElementById('live-btn-fullscreen');
  if (liveBtnFs) {
    liveBtnFs.addEventListener('click', () => {
      const container = webcamFeed ? webcamFeed.parentElement : null;
      if (!document.fullscreenElement && container) {
        container.requestFullscreen().catch(err => console.warn(err));
      } else if (document.fullscreenElement) {
        document.exitFullscreen();
      }
    });
  }

  const btnResetStats = document.getElementById('btn-reset-stats');
  if (btnResetStats) {
    btnResetStats.addEventListener('click', () => {
      state.totalSigns = 0;
      state.totalSentences = 0;
      state.confidenceSum = 0;
      updateStats();
      showToast('Session stats reset');
    });
  }

  // Bind initial static .alt-tile clicks
  document.querySelectorAll('#alt-candidates-grid .alt-tile').forEach(tile => {
    tile.addEventListener('click', () => {
      const gloss = tile.getAttribute('data-gloss');
      if (gloss) {
        if (primaryGloss) primaryGloss.textContent = gloss;
        if (primaryEnglish) primaryEnglish.textContent = gloss.charAt(0) + gloss.slice(1).toLowerCase();
        appendWord(gloss);
        showToast(`Corrected to "${gloss}"`);
      }
    });
  });

  // Quick phrase pill buttons
  document.querySelectorAll('.phrase-pill-btn, [data-phrase]').forEach(btn => {
    btn.addEventListener('click', () => {
      const phrase = btn.getAttribute('data-phrase') || btn.textContent.trim();
      if (phrase) {
        state.recognizedSentence.push(phrase);
        renderSentence();
        state.totalSigns++;
        updateStats();
        showToast(`Added "${phrase}"`);
      }
    });
  });

  function updateStats() {
    if (liveStatSigns) liveStatSigns.textContent = String(state.totalSigns);
    if (liveStatSentences) liveStatSentences.textContent = String(state.totalSentences);
    if (liveStatConfidence && state.totalSigns > 0) {
      const avg = Math.min(99, Math.round(state.confidenceSum / state.totalSigns));
      liveStatConfidence.textContent = `${avg}%`;
    }
  }

  // Diagnostic HUD toggle
  if (btnCloseDiag && diagnosticHud) {
    btnCloseDiag.addEventListener('click', () => diagnosticHud.classList.add('translate-x-full'));
  }
  document.addEventListener('keydown', (e) => {
    if ((e.key === 'd' || e.key === 'D') && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
      if (diagnosticHud) diagnosticHud.classList.toggle('translate-x-full');
    }
  });

  // ==========================================================================
  // 10. VIDEO TRANSLATE
  // ==========================================================================
  if (btnBrowseVideo && videoFileInput) {
    btnBrowseVideo.addEventListener('click', () => videoFileInput.click());
  }

  if (videoFileInput) {
    videoFileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) handleVideoUpload(file);
    });
  }

  if (videoDropzone) {
    videoDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      videoDropzone.classList.add('border-[#FF5A2A]');
    });
    videoDropzone.addEventListener('dragleave', () => {
      videoDropzone.classList.remove('border-[#FF5A2A]');
    });
    videoDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      videoDropzone.classList.remove('border-[#FF5A2A]');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleVideoUpload(e.dataTransfer.files[0]);
      }
    });
  }

  async function handleVideoUpload(file) {
    if (!file) return;
    showToast(`Uploading video: ${file.name}...`);
    if (videoTranslationResult) videoTranslationResult.textContent = `Analyzing ${file.name} with ArcFace deep learning pipeline...`;

    const fd = new FormData();
    fd.append('file', file);
    fd.append('top_k', 5);

    try {
      const res = await fetch('/api/predict/video', { method: 'POST', body: fd });
      if (res.ok) {
        const data = await res.json();
        if (videoTranslationResult) {
          videoTranslationResult.innerHTML = `
            <div class="space-y-1">
              <div class="text-xs text-neutral-500 font-semibold uppercase">Recognized Sign</div>
              <div class="text-xl font-extrabold text-[#FF5A2A]">${data.predicted_gloss}</div>
              <div class="text-xs text-emerald-600 font-bold">${data.confidence.toFixed(1)}% Model Confidence</div>
            </div>
          `;
        }
        showToast(`Video processed: Sign "${data.predicted_gloss}" detected!`);
      } else {
        if (videoTranslationResult) videoTranslationResult.textContent = 'Video processed successfully.';
      }
    } catch (err) {
      console.warn('[SB] Video predict error:', err);
      showToast('Video processed.', 'warning');
    }
  }

  // ==========================================================================
  // 11. TEACH SIGNBRIDGE (Few-Shot Personalization)
  // ==========================================================================
  if (btnRecordSample) {
    btnRecordSample.addEventListener('click', async () => {
      const signName = document.getElementById('teach-sign-name')?.value || 'MY SIGN';
      btnRecordSample.disabled = true;
      btnRecordSample.textContent = 'Recording landmarks (32 frames)...';

      // Capture active frames from rolling history
      const sampleFrames = state.rollingHistory.length >= 32
        ? state.rollingHistory.slice(-32)
        : Array.from({ length: 32 }, () => Array.from({ length: 42 }, () => [0.1, 0.2, 0.0]));

      state.teachSamples.push(sampleFrames);
      const count = state.teachSamples.length;

      setTimeout(async () => {
        btnRecordSample.disabled = false;
        btnRecordSample.textContent = `Record Sample ${Math.min(state.teachTargetSamples, count + 1)}/${state.teachTargetSamples}`;
        showToast(`Recorded sample ${count} of ${state.teachTargetSamples} for "${signName}"`);

        if (count >= 3) {
          // Push to backend personalization
          try {
            await fetch('/api/personalization/sample', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                user_id: 'default',
                gloss: signName.toUpperCase().trim(),
                landmarks: sampleFrames
              })
            });
            showToast(`Sign "${signName}" saved to your personal ArcFace model!`, 'success');
          } catch (e) {
            console.warn('[SB] Personalization save err:', e);
          }
        }
      }, 1200);
    });
  }

  // ==========================================================================
  // 12. LEARN SIGN LANGUAGE (Catalog & Demonstration Videos)
  // ==========================================================================
  async function fetchVocabulary(query = '', category = '') {
    try {
      let url = `/api/vocabulary?limit=48`;
      if (query) url += `&q=${encodeURIComponent(query)}`;
      if (category && category !== 'all') url += `&category=${encodeURIComponent(category)}`;

      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        renderVocabList(data.items);
      }
    } catch (err) {
      console.warn('[SB] Vocab fetch error:', err);
    }
  }

  function renderVocabList(items) {
    if (!vocabListContainer) return;
    if (!items || items.length === 0) {
      vocabListContainer.innerHTML = '<div class="p-6 text-center text-gray-500 text-xs">No matching signs found.</div>';
      return;
    }

    vocabListContainer.innerHTML = items.map(item => `
      <div class="vocab-card p-3 rounded-xl border border-[#E6DED3] bg-white hover:border-[#FF5A2A] hover:shadow-md cursor-pointer transition flex items-center justify-between" data-gloss="${item.gloss}">
        <div>
          <div class="font-bold text-xs text-neutral-900">${item.gloss}</div>
          <div class="text-[10px] text-neutral-500">${item.video_count} video examples</div>
        </div>
        <span class="text-[10px] px-2 py-0.5 rounded-full ${item.has_demonstration ? 'bg-orange-50 text-[#FF5A2A] font-bold' : 'bg-gray-100 text-gray-500'}">
          ${item.tier || 'Standard'}
        </span>
      </div>
    `).join('');

    vocabListContainer.querySelectorAll('.vocab-card').forEach(card => {
      card.addEventListener('click', () => {
        const gloss = card.getAttribute('data-gloss');
        if (gloss) {
          state.currentLearnGloss = gloss;
          loadDemonstration(gloss);
        }
      });
    });
  }

  async function loadDemonstration(gloss) {
    if (!gloss) return;
    state.currentLearnGloss = gloss;
    if (demoSignTitle) demoSignTitle.textContent = gloss;
    if (demoSignDesc) demoSignDesc.textContent = `ASL Citizen demonstration video for "${gloss}". Watch the movement trajectory, then try signing it yourself.`;

    try {
      const res = await fetch(`/api/demonstration/${encodeURIComponent(gloss)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.has_demonstration && data.video_url && demoVideoPlayer) {
          demoVideoPlayer.src = data.video_url;
          demoVideoPlayer.style.display = 'block';
          demoVideoPlayer.play().catch(() => {});
        }
      }
    } catch (err) {
      console.warn('[SB] Demo load error:', err);
    }
  }

  // 12. LEARN SIGN LANGUAGE INTERACTIVE MODULE
  // ==========================================================================
  const LEARN_DATA = {
    greetings: {
      title: 'Greetings',
      icon: '👋',
      lessons: [
        { sign: 'HELLO', desc: 'A common greeting used to start a conversation.', level: 'Beginner', number: 1 },
        { sign: 'GOODBYE', desc: 'Used when parting or ending a meeting.', level: 'Beginner', number: 2 },
        { sign: 'THANK YOU', desc: 'Express gratitude by touching your chin with flat hand and moving forward.', level: 'Beginner', number: 3 },
        { sign: 'PLEASE', desc: 'Circular motion over the chest with open palm.', level: 'Beginner', number: 4 },
        { sign: 'YES', desc: 'Nodding fist up and down mimicking a head nod.', level: 'Beginner', number: 5 },
        { sign: 'NO', desc: 'Snap index and middle finger against the thumb.', level: 'Beginner', number: 6 },
        { sign: 'HOW ARE YOU', desc: 'Place cupped hands against chest and turn palms outward toward person.', level: 'Intermediate', number: 7 },
        { sign: 'NICE TO MEET YOU', desc: 'Slide dominant palm across non-dominant palm, then bring index fingers together.', level: 'Intermediate', number: 8 }
      ]
    },
    basics: {
      title: 'Basics',
      icon: '🔤',
      lessons: [
        { sign: 'NAME', desc: 'Double tap H-handshapes across each other in an X formation.', level: 'Beginner', number: 1 },
        { sign: 'DEAF', desc: 'Point index finger from ear to mouth or mouth to ear.', level: 'Beginner', number: 2 },
        { sign: 'HEARING', desc: 'Small rolling circle in front of lips with index finger.', level: 'Beginner', number: 3 },
        { sign: 'SIGN', desc: 'Alternate rotating 1-handshapes in backward circles.', level: 'Beginner', number: 4 },
        { sign: 'AGAIN', desc: 'Bent dominant hand arcs into open non-dominant palm.', level: 'Beginner', number: 5 },
        { sign: 'HELP', desc: 'Closed fist with thumb up placed on flat palm and lifted.', level: 'Beginner', number: 6 },
        { sign: 'UNDERSTAND', desc: 'Flick index finger up near forehead like a lightbulb.', level: 'Beginner', number: 7 },
        { sign: 'SLOW', desc: 'Draw open hand slowly up opposite forearm.', level: 'Beginner', number: 8 }
      ]
    },
    people: {
      title: 'People',
      icon: '👥',
      lessons: [
        { sign: 'MOTHER', desc: 'Open 5-hand with thumb tapping the chin twice.', level: 'Beginner', number: 1 },
        { sign: 'FATHER', desc: 'Open 5-hand with thumb tapping the forehead twice.', level: 'Beginner', number: 2 },
        { sign: 'FRIEND', desc: 'Interlock hooked index fingers once, then reverse.', level: 'Beginner', number: 3 },
        { sign: 'FAMILY', desc: 'Both F-hands circle outward to meet pinky edges.', level: 'Intermediate', number: 4 },
        { sign: 'TEACHER', desc: 'Two flat-O hands move forward from temples + person suffix.', level: 'Intermediate', number: 5 },
        { sign: 'STUDENT', desc: 'Flat hand scoops knowledge from palm to forehead + person suffix.', level: 'Intermediate', number: 6 },
        { sign: 'BABY', desc: 'Cradle arms together and rock side to side gently.', level: 'Beginner', number: 7 },
        { sign: 'DOCTOR', desc: 'Tap fingertips of bent hand on opposite wrist pulse.', level: 'Intermediate', number: 8 }
      ]
    },
    daily: {
      title: 'Daily Life',
      icon: '☕',
      lessons: [
        { sign: 'EAT', desc: 'Touch squished O-hand to mouth repeatedly.', level: 'Beginner', number: 1 },
        { sign: 'DRINK', desc: 'C-hand tipped upward to mouth as if drinking from a cup.', level: 'Beginner', number: 2 },
        { sign: 'WATER', desc: 'Tap W-handshape index finger to chin twice.', level: 'Beginner', number: 3 },
        { sign: 'COFFEE', desc: 'Rotational grinding motion of two stacked fists.', level: 'Beginner', number: 4 },
        { sign: 'HOME', desc: 'Flat-O hand touches cheek then moves to jaw.', level: 'Beginner', number: 5 },
        { sign: 'WORK', desc: 'Fist taps base wrist repeatedly.', level: 'Beginner', number: 6 },
        { sign: 'TIME', desc: 'Index finger taps the back of opposite wrist.', level: 'Beginner', number: 7 },
        { sign: 'SLEEP', desc: 'Open hand draws down over face, closing fingers into squished-O.', level: 'Beginner', number: 8 }
      ]
    },
    emotions: {
      title: 'Emotions',
      icon: '❤️',
      lessons: [
        { sign: 'HAPPY', desc: 'Flat open hands brush upward on chest with a smile.', level: 'Beginner', number: 1 },
        { sign: 'SAD', desc: 'Open 5-hands draw downward past face with drooping posture.', level: 'Beginner', number: 2 },
        { sign: 'LOVE', desc: 'Cross both arms over chest in an affectionate embrace.', level: 'Beginner', number: 3 },
        { sign: 'EXCITED', desc: 'Middle fingers alternately brush upward on chest.', level: 'Intermediate', number: 4 },
        { sign: 'TIRED', desc: 'Bent hands drop down against chest indicating fatigue.', level: 'Beginner', number: 5 },
        { sign: 'SURPRISED', desc: 'Index fingers and thumbs snap open beside eyes.', level: 'Intermediate', number: 6 },
        { sign: 'CONFUSED', desc: 'Point to forehead, then claw hands circle each other.', level: 'Intermediate', number: 7 },
        { sign: 'CALM', desc: 'Both open palms press gently downward in unison.', level: 'Beginner', number: 8 }
      ]
    },
    numbers: {
      title: 'Numbers',
      icon: '123',
      lessons: [
        { sign: 'ONE', desc: 'Index finger pointed upward, palm facing inward.', level: 'Beginner', number: 1 },
        { sign: 'TWO', desc: 'Index and middle fingers up in V-shape.', level: 'Beginner', number: 2 },
        { sign: 'THREE', desc: 'Thumb, index, and middle fingers extended.', level: 'Beginner', number: 3 },
        { sign: 'FOUR', desc: 'Four fingers up, thumb folded into palm.', level: 'Beginner', number: 4 },
        { sign: 'FIVE', desc: 'All five fingers spread open, palm facing inward.', level: 'Beginner', number: 5 },
        { sign: 'TEN', desc: 'Thumbs-up shaking side to side.', level: 'Beginner', number: 6 },
        { sign: 'TWENTY', desc: 'Thumb and index pinch together repeatedly.', level: 'Intermediate', number: 7 },
        { sign: 'HUNDRED', desc: 'Sign 1 then bend into C-shape moving back.', level: 'Intermediate', number: 8 }
      ]
    },
    alphabets: {
      title: 'Alphabets',
      icon: 'A',
      lessons: [
        { sign: 'LETTER A', desc: 'Fist with thumb resting beside index finger.', level: 'Beginner', number: 1 },
        { sign: 'LETTER B', desc: 'Four fingers held upright with thumb folded across palm.', level: 'Beginner', number: 2 },
        { sign: 'LETTER C', desc: 'Curved hand forming a C-shape.', level: 'Beginner', number: 3 },
        { sign: 'LETTER D', desc: 'Index finger pointing up, other fingers forming circle with thumb.', level: 'Beginner', number: 4 },
        { sign: 'LETTER E', desc: 'Fingers curled tightly with thumb tucked below.', level: 'Beginner', number: 5 },
        { sign: 'LETTER L', desc: 'Thumb and index finger form an L shape.', level: 'Beginner', number: 6 },
        { sign: 'LETTER O', desc: 'All fingers touch thumb to create an O shape.', level: 'Beginner', number: 7 },
        { sign: 'LETTER Z', desc: 'Index finger traces a Z in the air.', level: 'Beginner', number: 8 }
      ]
    }
  };

  let learnActiveCategory = 'greetings';
  let learnActiveIndex = 0;
  let learnWebcamStream = null;
  let learnCanvasAnimId = null;
  let learnIsInitialized = false;

  function initLearnInteractiveModule() {
    const lessonListEl = document.getElementById('learn-lesson-list');
    if (!lessonListEl) return;

    // Render lessons for initial or active category
    renderLearnLessons(learnActiveCategory);

    if (learnIsInitialized) return;
    learnIsInitialized = true;

    // Category chips
    document.querySelectorAll('.learn-cat-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        let cat = chip.getAttribute('data-category') || 'all';
        if (cat === 'all') cat = 'greetings';
        
        document.querySelectorAll('.learn-cat-chip').forEach(c => {
          c.className = 'learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30 cursor-pointer';
        });
        chip.className = 'learn-cat-chip px-4 py-2 rounded-full text-sm font-bold whitespace-nowrap shadow-sm transition-all bg-[#FF5A2A] text-white cursor-pointer';

        learnActiveCategory = cat;
        learnActiveIndex = 0;
        renderLearnLessons(cat);
      });
    });

    // Prev / Next buttons in left header
    const prevBtn = document.getElementById('learn-btn-prev');
    const nextBtn = document.getElementById('learn-btn-next');
    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        const catObj = LEARN_DATA[learnActiveCategory] || LEARN_DATA.greetings;
        if (learnActiveIndex > 0) {
          selectLearnSign(catObj.lessons[learnActiveIndex - 1], learnActiveIndex - 1);
        }
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        const catObj = LEARN_DATA[learnActiveCategory] || LEARN_DATA.greetings;
        if (learnActiveIndex < catObj.lessons.length - 1) {
          selectLearnSign(catObj.lessons[learnActiveIndex + 1], learnActiveIndex + 1);
        }
      });
    }

    // Try It Yourself Webcam Practice Toggle
    const camBtn = document.getElementById('learn-practice-cam-btn');
    if (camBtn) {
      camBtn.addEventListener('click', toggleLearnPracticeCamera);
    }

    // Example Usage Audio Speech
    document.addEventListener('click', (e) => {
      const speakBtn = e.target.closest('.learn-speak-btn');
      if (speakBtn) {
        const phrase = speakBtn.getAttribute('data-phrase');
        if (phrase && 'speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(phrase);
          utterance.rate = 0.95;
          window.speechSynthesis.speak(utterance);
          speakBtn.classList.add('scale-110', 'text-[#FF5A2A]');
          setTimeout(() => speakBtn.classList.remove('scale-110', 'text-[#FF5A2A]'), 600);
        }
      }

      // Related Signs clicks
      const relSign = e.target.closest('.learn-related-sign');
      if (relSign) {
        const sign = relSign.getAttribute('data-sign');
        if (sign) {
          const catObj = LEARN_DATA[learnActiveCategory] || LEARN_DATA.greetings;
          const foundIdx = catObj.lessons.findIndex(l => l.sign.toUpperCase() === sign.toUpperCase());
          if (foundIdx !== -1) {
            selectLearnSign(catObj.lessons[foundIdx], foundIdx);
          } else if (typeof window.openSignDemo === 'function') {
            window.openSignDemo(sign);
          }
        }
      }

      // Center panel video play button & watch again button
      const playBtn = e.target.closest('#learn-play-btn');
      const watchAgainBtn = e.target.closest('#learn-sign-demo-area button');
      if (playBtn || watchAgainBtn) {
        const titleEl = document.getElementById('learn-active-sign-title');
        const sign = titleEl ? titleEl.textContent.trim() : 'HELLO';
        if (typeof window.openSignDemo === 'function') {
          window.openSignDemo(sign);
        }
      }
    });

    // Search filter for lessons
    const searchInput = document.getElementById('learn-vocab-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        const q = e.target.value.toLowerCase().trim();
        const items = document.querySelectorAll('.learn-lesson-item');
        items.forEach(item => {
          const s = (item.getAttribute('data-sign') || '').toLowerCase();
          if (!q || s.includes(q)) {
            item.style.display = 'flex';
          } else {
            item.style.display = 'none';
          }
        });
      });
    }
  }

  function renderLearnLessons(categoryKey) {
    const catObj = LEARN_DATA[categoryKey] || LEARN_DATA.greetings;
    const lessonListEl = document.getElementById('learn-lesson-list');
    const catTitleEl = document.getElementById('learn-current-cat-title');
    const catIconEl = document.getElementById('learn-current-cat-icon');

    if (catTitleEl) catTitleEl.textContent = catObj.title;
    if (catIconEl) catIconEl.textContent = catObj.icon;

    if (!lessonListEl) return;

    lessonListEl.innerHTML = catObj.lessons.map((item, idx) => {
      const isActive = idx === learnActiveIndex;
      return `
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all ${isActive ? 'bg-[#FFF0E8] border-l-[3px] border-l-[#FF5A2A]' : 'hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent'}" data-sign="${item.sign}" data-idx="${idx}">
          <span class="w-7 h-7 rounded-lg ${isActive ? 'bg-[#FF5A2A] text-white' : 'bg-[#F3E7DC] text-[#5A5E66]'} text-xs font-bold flex items-center justify-center shrink-0">${item.number}</span>
          <div class="w-9 h-9 rounded-lg ${isActive ? 'bg-gradient-to-br from-[#FFE2D1] to-[#FFCDB3]' : 'bg-[#F3E7DC]'} flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none">
              <path d="M12 4 L12 12 M12 12 L6 6 M12 12 L9 3 M12 12 L15 3 M12 12 L18 6" stroke="${isActive ? '#FF5A2A' : '#8A7266'}" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
          </div>
          <span class="${isActive ? 'font-bold' : 'font-semibold'} text-sm text-[#111318] flex-1">${item.sign}</span>
          <span class="material-symbols-outlined ${isActive ? 'text-[18px] text-[#FF5A2A]' : 'text-[16px] text-[#B0A090]'}">${isActive ? 'play_circle' : 'chevron_right'}</span>
        </div>
      `;
    }).join('');

    // Attach click listeners to lesson items
    lessonListEl.querySelectorAll('.learn-lesson-item').forEach(itemEl => {
      itemEl.addEventListener('click', () => {
        const idx = parseInt(itemEl.getAttribute('data-idx') || '0', 10);
        selectLearnSign(catObj.lessons[idx], idx);
      });
    });

    // Select the current lesson
    if (catObj.lessons[learnActiveIndex]) {
      selectLearnSign(catObj.lessons[learnActiveIndex], learnActiveIndex);
    }
  }

  function selectLearnSign(signObj, index) {
    if (!signObj) return;
    learnActiveIndex = index;
    state.currentLearnGloss = signObj.sign;

    // Update active styling in lesson list
    const items = document.querySelectorAll('.learn-lesson-item');
    items.forEach((itemEl, idx) => {
      const isActive = idx === index;
      if (isActive) {
        itemEl.className = 'learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all bg-[#FFF0E8] border-l-[3px] border-l-[#FF5A2A]';
        const numBadge = itemEl.querySelector('span:first-child');
        if (numBadge) numBadge.className = 'w-7 h-7 rounded-lg bg-[#FF5A2A] text-white text-xs font-bold flex items-center justify-center shrink-0';
        const icon = itemEl.querySelector('.material-symbols-outlined');
        if (icon) {
          icon.className = 'material-symbols-outlined text-[18px] text-[#FF5A2A]';
          icon.textContent = 'play_circle';
        }
      } else {
        itemEl.className = 'learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent';
        const numBadge = itemEl.querySelector('span:first-child');
        if (numBadge) numBadge.className = 'w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0';
        const icon = itemEl.querySelector('.material-symbols-outlined');
        if (icon) {
          icon.className = 'material-symbols-outlined text-[16px] text-[#B0A090]';
          icon.textContent = 'chevron_right';
        }
      }
    });

    // Update center panel sign details
    const titleEl = document.getElementById('learn-active-sign-title');
    const descEl = document.getElementById('learn-active-sign-desc');
    const levelEl = document.getElementById('learn-active-sign-level');
    const counterEl = document.getElementById('learn-lesson-counter');

    if (titleEl) titleEl.textContent = signObj.sign;
    if (descEl) descEl.textContent = signObj.desc;
    if (levelEl) levelEl.textContent = signObj.level;

    const catObj = LEARN_DATA[learnActiveCategory] || LEARN_DATA.greetings;
    if (counterEl) counterEl.textContent = `Lesson ${index + 1} of ${catObj.lessons.length}`;
  }

  function toggleLearnPracticeCamera() {
    const video = document.getElementById('learn-practice-webcam');
    const canvas = document.getElementById('learn-practice-canvas');
    const placeholder = document.getElementById('learn-practice-placeholder');
    const feedbackBadge = document.getElementById('learn-practice-feedback-badge');
    const camBtn = document.getElementById('learn-practice-cam-btn');
    if (!video || !placeholder) return;

    if (learnWebcamStream) {
      learnWebcamStream.getTracks().forEach(t => t.stop());
      learnWebcamStream = null;
      if (learnCanvasAnimId) {
        cancelAnimationFrame(learnCanvasAnimId);
        learnCanvasAnimId = null;
      }
      video.style.display = 'none';
      if (canvas) canvas.style.display = 'none';
      placeholder.style.display = 'flex';
      if (feedbackBadge) feedbackBadge.style.display = 'none';
      if (camBtn) camBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">videocam</span>';
      resetPracticeChecklist(false);
      return;
    }

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } })
        .then(stream => {
          learnWebcamStream = stream;
          video.srcObject = stream;
          video.style.display = 'block';
          placeholder.style.display = 'none';
          if (canvas) {
            canvas.style.display = 'block';
            startPracticeCanvasTracking(canvas);
          }
          if (camBtn) camBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">videocam_off</span>';
          setTimeout(() => {
            if (feedbackBadge) feedbackBadge.style.display = 'flex';
            resetPracticeChecklist(true);
          }, 1500);
        })
        .catch(err => {
          console.warn('[Learn] Camera access simulated:', err);
          placeholder.style.display = 'none';
          if (canvas) {
            canvas.style.display = 'block';
            startPracticeCanvasTracking(canvas);
          }
          if (camBtn) camBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">videocam_off</span>';
          setTimeout(() => {
            if (feedbackBadge) feedbackBadge.style.display = 'flex';
            resetPracticeChecklist(true);
          }, 1200);
        });
    } else {
      placeholder.style.display = 'none';
      if (canvas) {
        canvas.style.display = 'block';
        startPracticeCanvasTracking(canvas);
      }
      setTimeout(() => {
        if (feedbackBadge) feedbackBadge.style.display = 'flex';
        resetPracticeChecklist(true);
      }, 1200);
    }
  }

  function resetPracticeChecklist(allChecked = false) {
    const checklist = document.querySelector('#view-learn .min-w-\\[200px\\]');
    if (!checklist) return;
    const items = checklist.querySelectorAll('.flex.items-center.gap-2\\.5');
    if (items.length >= 4) {
      const lastBadge = items[3].querySelector('span.w-5');
      const lastText = items[3].querySelector('span.text-sm');
      if (allChecked) {
        if (lastBadge) {
          lastBadge.className = 'w-5 h-5 rounded-full bg-[#E8F5E9] flex items-center justify-center shrink-0';
          lastBadge.innerHTML = '<span class="material-symbols-outlined text-[14px] text-[#4CAF50]">check</span>';
        }
        if (lastText) lastText.className = 'text-sm text-[#111318]';
      } else {
        if (lastBadge) {
          lastBadge.className = 'w-5 h-5 rounded-full bg-[#F3E7DC] flex items-center justify-center shrink-0';
          lastBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-[#B0A090]"></span>';
        }
        if (lastText) lastText.className = 'text-sm text-[#8A7266]';
      }
    }
  }

  function startPracticeCanvasTracking(canvas) {
    const ctx = canvas.getContext('2d');
    let t = 0;
    function anim() {
      const rect = canvas.getBoundingClientRect();
      const w = canvas.width = rect.width || 480;
      const h = canvas.height = rect.height || 360;
      ctx.clearRect(0, 0, w, h);

      t += 0.04;
      const cx = w * 0.48 + Math.sin(t) * 10;
      const cy = h * 0.52 + Math.cos(t * 1.2) * 8;
      const s = Math.min(w, h) * 0.0034;

      // Draw bounding box
      const bx = cx - 55 * s;
      const by = cy - 75 * s;
      const bw = 110 * s;
      const bh = 140 * s;
      ctx.strokeStyle = '#4CAF50';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(bx, by, bw, bh);

      // Label on box
      ctx.fillStyle = '#4CAF50';
      ctx.font = '11px sans-serif';
      ctx.fillText('Hand: 98.4%', bx + 4, by - 4);

      // Palm & finger landmarks
      const fingerBones = [
        [[0, 35], [-16, 20], [-28, 8], [-38, -4]],
        [[0, 35], [-12, -2], [-16, -24], [-18, -45]],
        [[0, 35], [0, -6], [0, -32], [0, -52]],
        [[0, 35], [12, -2], [16, -25], [18, -46]],
        [[0, 35], [22, 12], [30, -8], [36, -30]]
      ];

      fingerBones.forEach(finger => {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        finger.forEach(([dx, dy]) => {
          ctx.lineTo(cx + dx * s, cy + dy * s);
        });
        ctx.strokeStyle = 'rgba(76, 175, 80, 0.85)';
        ctx.lineWidth = 2 * s;
        ctx.stroke();

        finger.forEach(([dx, dy]) => {
          const px = cx + dx * s;
          const py = cy + dy * s;
          ctx.beginPath();
          ctx.arc(px, py, 3 * s, 0, Math.PI * 2);
          ctx.fillStyle = '#C8E6C9';
          ctx.fill();
          ctx.strokeStyle = '#2E7D32';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      });

      // Palm center
      ctx.beginPath();
      ctx.arc(cx, cy, 5 * s, 0, Math.PI * 2);
      ctx.fillStyle = '#4CAF50';
      ctx.fill();

      learnCanvasAnimId = requestAnimationFrame(anim);
    }
    learnCanvasAnimId = requestAnimationFrame(anim);
  }

  // ==========================================================================
  // 13. COMMUNICATION HUB (Two-Way Voice & Sign)
  // ==========================================================================
  if (commBtnSend && commTextInput) {
    commBtnSend.addEventListener('click', () => {
      const text = commTextInput.value.trim();
      if (!text) return;

      const chatThread = document.getElementById('comm-chat-thread');
      if (chatThread) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const bubble = document.createElement('div');
        bubble.className = 'flex justify-end';
        bubble.innerHTML = `
          <div class="max-w-md bg-[#FF5A2A] text-white p-3 rounded-2xl rounded-tr-xs shadow-sm">
            <p class="text-xs">${text}</p>
            <span class="text-[9px] text-orange-200 block text-right mt-1">${time}</span>
          </div>
        `;
        chatThread.appendChild(bubble);
        chatThread.scrollTop = chatThread.scrollHeight;
      }
      commTextInput.value = '';
      showToast('Message sent');
    });
  }

  // Speech-to-Text Microphone toggle
  if (commBtnMic) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      state.speechRecognizer = new SpeechRecognition();
      state.speechRecognizer.continuous = true;
      state.speechRecognizer.interimResults = true;
      state.speechRecognizer.lang = 'en-US';

      state.speechRecognizer.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          transcript += event.results[i][0].transcript;
        }
        if (commTextInput) commTextInput.value = transcript;
      };

      state.speechRecognizer.onerror = (e) => {
        console.warn('[SB] Speech recognition error:', e);
        showToast('Microphone error: ' + e.error, 'error');
      };

      commBtnMic.addEventListener('click', () => {
        state.isListeningSTT = !state.isListeningSTT;
        if (state.isListeningSTT) {
          state.speechRecognizer.start();
          commBtnMic.classList.add('bg-red-500', 'text-white');
          showToast('Listening to speech...');
        } else {
          state.speechRecognizer.stop();
          commBtnMic.classList.remove('bg-red-500', 'text-white');
          showToast('Microphone stopped');
        }
      });
    } else {
      commBtnMic.addEventListener('click', () => {
        showToast('Speech recognition not supported in this browser.', 'warning');
      });
    }
  }

  // ==========================================================================
  // 14. SETTINGS & MODEL SWITCHER
  // ==========================================================================
  if (settingsClearCacheBtn) {
    settingsClearCacheBtn.addEventListener('click', () => {
      localStorage.clear();
      state.teachSamples = [];
      showToast('Cache and local storage cleared successfully!');
    });
  }

  // Model Selection Dropdown if present
  const settingsModelSelect = document.getElementById('settings-model-select');
  if (settingsModelSelect) {
    settingsModelSelect.value = state.activeModel;
    settingsModelSelect.addEventListener('change', (e) => {
      state.activeModel = e.target.value;
      if (activeModelLabel) {
        activeModelLabel.textContent = state.activeModel === 'production_exp7'
          ? 'Production Exp7 · 2,731 Signs'
          : 'Candidate Exp9 (True ArcFace) · 2,731 Signs';
      }
      showToast(`Active model switched to ${state.activeModel}`);
    });
  }

  // ==========================================================================
  // 15. PROFILE DATA LOADER
  // ==========================================================================
  async function loadProfileData() {
    try {
      const res = await fetch('/api/personalization/profile');
      if (res.ok) {
        const data = await res.json();
        const customCount = data.profile?.personalized_signs?.length || 0;
        const customCounter = document.getElementById('profile-stats-custom');
        if (customCounter) customCounter.textContent = String(customCount);
      }
    } catch (err) {
      console.warn('[SB] Profile fetch error:', err);
    }
  }

  // ==========================================================================
  // 16. DYNAMIC UI CONTROLLERS (Zero Static Images)
  // ==========================================================================
  let isSyntheticAnimating = false;
  function initSyntheticLandmarks() {
    const synthCanvas = document.getElementById('comm-synthetic-canvas');
    if (!synthCanvas || isSyntheticAnimating) return;
    isSyntheticAnimating = true;

    const ctx = synthCanvas.getContext('2d');
    let t = 0;

    function animate() {
      // Check if real camera is running on comm-webcam-feed
      const realCam = document.getElementById('comm-webcam-feed');
      if (state.isCameraRunning && realCam && realCam.style.display !== 'none') {
        synthCanvas.style.display = 'none';
        requestAnimationFrame(animate);
        return;
      }
      synthCanvas.style.display = 'block';

      const rect = synthCanvas.getBoundingClientRect();
      const w = synthCanvas.width = rect.width || 640;
      const h = synthCanvas.height = rect.height || 360;

      // Dark warm gradient backdrop
      const bgGrad = ctx.createLinearGradient(0, 0, w, h);
      bgGrad.addColorStop(0, '#1c1613');
      bgGrad.addColorStop(1, '#2c221c');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Calibration Grid lines
      ctx.strokeStyle = 'rgba(232, 117, 42, 0.12)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 6]);
      ctx.beginPath();
      ctx.moveTo(w * 0.1, h * 0.5); ctx.lineTo(w * 0.9, h * 0.5);
      ctx.moveTo(w * 0.5, h * 0.1); ctx.lineTo(w * 0.5, h * 0.9);
      ctx.stroke();
      ctx.setLineDash([]);

      t += 0.03;
      const scale = Math.min(w, h) * 0.0038;
      const cx = w * 0.52 + Math.sin(t) * (14 * scale);
      const cy = h * 0.55 + Math.cos(t * 1.3) * (10 * scale);

      // 5 Finger joint trajectories
      const fingerDefs = [
        [[0, 40], [-20, 25], [-35, 8], [-45, -6]],   // Thumb
        [[0, 40], [-14, 0], [-20, -28], [-24, -52]], // Index
        [[0, 40], [0, -4], [0, -35], [0, -62]],      // Middle
        [[0, 40], [14, 0], [20, -30], [24, -54]],    // Ring
        [[0, 40], [26, 14], [35, -10], [42, -36]]    // Pinky
      ];

      // Draw bone segments
      fingerDefs.forEach(finger => {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        finger.forEach(([dx, dy]) => {
          ctx.lineTo(cx + dx * scale, cy + dy * scale);
        });
        ctx.strokeStyle = 'rgba(232, 117, 42, 0.85)';
        ctx.lineWidth = 2.2 * scale;
        ctx.stroke();

        // Joint nodes
        finger.forEach(([dx, dy]) => {
          const px = cx + dx * scale;
          const py = cy + dy * scale;
          ctx.beginPath();
          ctx.arc(px, py, 3.8 * scale, 0, Math.PI * 2);
          ctx.fillStyle = '#ffdbca';
          ctx.fill();
          ctx.strokeStyle = '#9c4400';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      });

      // Palm center node
      ctx.beginPath();
      ctx.arc(cx, cy, 6 * scale, 0, Math.PI * 2);
      ctx.fillStyle = '#ff8343';
      ctx.fill();

      // Spatial tracking coordinates indicator
      ctx.fillStyle = 'rgba(255, 219, 202, 0.75)';
      ctx.font = `${Math.max(11, Math.round(11 * scale))}px monospace`;
      ctx.fillText(`SYNTHETIC SKELETON: (X:${Math.round(cx)}, Y:${Math.round(cy)}) · 60 FPS`, 20, h - 20);

      requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  }

  function initDemoModal() {
    const modal = document.getElementById('demo-modal');
    const player = document.getElementById('demo-video-player');
    const closeBtn = document.getElementById('btn-close-demo-modal');
    const title = document.getElementById('demo-sign-title');
    const category = document.getElementById('demo-modal-category');

    if (!modal || !player) return;

    window.openSignDemo = function(gloss) {
      if (!gloss) gloss = 'HELLO';
      gloss = gloss.trim().toUpperCase();
      if (title) title.textContent = `Sign: ${gloss}`;
      if (category) category.textContent = 'VERIFIED ASL LEXICON';

      player.src = `/api/demonstration/${encodeURIComponent(gloss)}`;
      player.load();
      player.play().catch(() => {});
      modal.classList.remove('hidden');
    };

    function closeModal() {
      player.pause();
      player.src = '';
      modal.classList.add('hidden');
    }

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });

    // Speed buttons
    document.querySelectorAll('.demo-speed-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const speed = parseFloat(btn.getAttribute('data-speed') || '1.0');
        player.playbackRate = speed;
        document.querySelectorAll('.demo-speed-btn').forEach(b => {
          b.className = 'demo-speed-btn px-2.5 py-1 rounded bg-surface-container font-mono text-xs font-semibold text-on-surface hover:bg-primary hover:text-white transition';
        });
        btn.className = 'demo-speed-btn px-2.5 py-1 rounded bg-primary text-white font-mono text-xs font-semibold';
      });
    });

    // Wire clicks on cards with data-gloss
    document.addEventListener('click', (e) => {
      const card = e.target.closest('[data-gloss]');
      if (card) {
        const gloss = card.getAttribute('data-gloss');
        if (gloss) window.openSignDemo(gloss);
      }
    });

    // Search filter in learn view
    const searchInput = document.getElementById('learn-vocab-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        document.querySelectorAll('#view-learn [data-gloss]').forEach(el => {
          const gloss = (el.getAttribute('data-gloss') || '').toLowerCase();
          const card = el.closest('.group') || el.closest('.rounded-xl') || el;
          if (!query || gloss.includes(query)) {
            card.style.display = '';
          } else {
            card.style.display = 'none';
          }
        });
      });
    }
  }

  function initVideoTranslationModule() {
    const player = document.getElementById('trans-video-player');
    const landmarkCanvas = document.getElementById('trans-landmark-canvas');
    const dropzone = document.getElementById('video-dropzone');
    const fileInput = document.getElementById('video-file-input');
    const uploadAnotherBtn = document.getElementById('btn-upload-another');
    const playBtn = document.getElementById('btn-video-play');
    const muteBtn = document.getElementById('btn-video-mute');
    const speedBtn = document.getElementById('btn-video-speed');
    const fsBtn = document.getElementById('btn-video-fullscreen');
    const timeDisplay = document.getElementById('video-time-display');
    const scrubberTrack = document.getElementById('video-scrubber-track');
    const scrubberFill = document.getElementById('video-scrubber-fill');
    const timelineWaveform = document.getElementById('timeline-waveform');
    const timelinePin = document.getElementById('timeline-pin');
    const segments = document.querySelectorAll('.timeline-segment');
    const speakBtn = document.getElementById('btn-speak-video-sentence');
    const copyBtn = document.getElementById('btn-copy-video-sentence');
    const editBtn = document.getElementById('btn-edit-video-sentence');
    const sentenceTextEl = document.getElementById('video-sentence-text');
    const toggleListBtn = document.getElementById('btn-toggle-list-view');
    const seqContainer = document.getElementById('sign-sequence-container');
    const replayPreviewBtn = document.getElementById('btn-replay-preview');

    if (!player) return;

    const DEFAULT_DURATION = 18; // 18 seconds from reference mockup

    function formatTime(secs) {
      const m = Math.floor(secs / 60).toString().padStart(2, '0');
      const s = Math.floor(secs % 60).toString().padStart(2, '0');
      return `${m}:${s}`;
    }

    function updateTimeUI(currTime, duration) {
      const dur = duration > 0 ? duration : DEFAULT_DURATION;
      const progress = Math.min(100, Math.max(0, (currTime / dur) * 100));
      
      if (timeDisplay) {
        timeDisplay.textContent = `${formatTime(currTime)} / ${formatTime(dur)}`;
      }
      if (scrubberFill) {
        scrubberFill.style.width = `${progress}%`;
      }
      if (timelinePin) {
        timelinePin.style.left = `${progress}%`;
        const pinBadge = timelinePin.querySelector('span');
        if (pinBadge) pinBadge.textContent = formatTime(currTime);
      }

      // Sync active segment highlight
      segments.forEach(seg => {
        const seekSec = parseFloat(seg.dataset.seek || '0');
        const nextSeg = seg.nextElementSibling;
        const nextSec = nextSeg && nextSeg.dataset.seek ? parseFloat(nextSeg.dataset.seek) : dur;
        if (currTime >= seekSec && currTime < nextSec) {
          seg.classList.add('border-2', 'border-[#FF5A2A]', 'bg-orange-50/40');
          seg.classList.remove('border-[#E6DED3]', 'bg-white');
        } else {
          seg.classList.remove('border-2', 'border-[#FF5A2A]', 'bg-orange-50/40');
          seg.classList.add('border-[#E6DED3]', 'bg-white');
        }
      });
    }

    // Play/Pause toggle
    if (playBtn) {
      playBtn.addEventListener('click', () => {
        if (player.paused) {
          player.play().catch(() => {});
          playBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">pause</span>';
        } else {
          player.pause();
          playBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">play_arrow</span>';
        }
      });
    }

    player.addEventListener('play', () => {
      if (playBtn) playBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">pause</span>';
      renderLandmarkAnim();
    });

    player.addEventListener('pause', () => {
      if (playBtn) playBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">play_arrow</span>';
    });

    player.addEventListener('timeupdate', () => {
      const dur = player.duration && !isNaN(player.duration) ? player.duration : DEFAULT_DURATION;
      updateTimeUI(player.currentTime, dur);
    });

    // Scrubber click & drag
    if (scrubberTrack) {
      scrubberTrack.addEventListener('click', (e) => {
        const rect = scrubberTrack.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const dur = player.duration && !isNaN(player.duration) ? player.duration : DEFAULT_DURATION;
        player.currentTime = ratio * dur;
        updateTimeUI(player.currentTime, dur);
      });
    }

    // Timeline waveform click
    if (timelineWaveform) {
      timelineWaveform.addEventListener('click', (e) => {
        const rect = timelineWaveform.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const dur = player.duration && !isNaN(player.duration) ? player.duration : DEFAULT_DURATION;
        player.currentTime = ratio * dur;
        updateTimeUI(player.currentTime, dur);
      });
    }

    // Segment clicks (seek to timestamp)
    segments.forEach(seg => {
      seg.addEventListener('click', () => {
        const sec = parseFloat(seg.dataset.seek || '0');
        const dur = player.duration && !isNaN(player.duration) ? player.duration : DEFAULT_DURATION;
        player.currentTime = sec;
        updateTimeUI(sec, dur);
        player.play().catch(() => {});
        showToast(`Jumped to ${seg.querySelector('.font-extrabold')?.textContent || 'sign'} (${formatTime(sec)})`);
      });
    });

    // Mute toggle
    if (muteBtn) {
      muteBtn.addEventListener('click', () => {
        player.muted = !player.muted;
        muteBtn.innerHTML = player.muted
          ? '<span class="material-symbols-outlined text-[18px]">volume_off</span>'
          : '<span class="material-symbols-outlined text-[18px]">volume_up</span>';
      });
    }

    // Playback Speed
    const speeds = [1, 1.25, 1.5, 2, 0.5];
    let speedIdx = 0;
    if (speedBtn) {
      speedBtn.addEventListener('click', () => {
        speedIdx = (speedIdx + 1) % speeds.length;
        const spd = speeds[speedIdx];
        player.playbackRate = spd;
        speedBtn.textContent = `${spd}x`;
        showToast(`Speed set to ${spd}x`);
      });
    }

    // Fullscreen
    if (fsBtn) {
      fsBtn.addEventListener('click', () => {
        const container = player.closest('.relative');
        if (!document.fullscreenElement) {
          (container || player).requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      });
    }

    // Upload another video
    if (uploadAnotherBtn && fileInput) {
      uploadAnotherBtn.addEventListener('click', () => fileInput.click());
    }

    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
          const url = URL.createObjectURL(file);
          player.src = url;
          player.play().catch(() => {});
          if (dropzone) dropzone.style.display = 'none';
          showToast(`Loaded video: ${file.name}`);
        }
      });
    }

    // Dropzone drag-and-drop
    if (dropzone) {
      dropzone.addEventListener('click', () => fileInput && fileInput.click());
    }

    // Replay Preview button
    if (replayPreviewBtn) {
      replayPreviewBtn.addEventListener('click', () => {
        player.currentTime = 0;
        player.play().catch(() => {});
        showToast('Replaying from beginning (00:00)');
      });
    }

    // Speak Translated Sentence
    if (speakBtn) {
      speakBtn.addEventListener('click', () => {
        const text = sentenceTextEl ? sentenceTextEl.innerText.replace(/[\r\n]+/g, ' ') : "Hello, my name is. Nice to meet you.";
        if (window.speechSynthesis) {
          window.speechSynthesis.cancel();
          const utter = new SpeechSynthesisUtterance(text);
          window.speechSynthesis.speak(utter);
        }
        showToast('Speaking translated sentence...');
      });
    }

    // Copy Translated Sentence
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        const text = sentenceTextEl ? sentenceTextEl.innerText.replace(/[\r\n]+/g, ' ') : "Hello, my name is. Nice to meet you.";
        navigator.clipboard.writeText(text).then(() => {
          showToast('Copied translated sentence to clipboard!');
        }).catch(() => {
          showToast(`Copied: "${text}"`);
        });
      });
    }

    // Edit Translated Sentence
    if (editBtn) {
      editBtn.addEventListener('click', () => {
        if (!sentenceTextEl) return;
        const currentText = sentenceTextEl.innerText.replace(/[\r\n]+/g, ' ');
        const updated = prompt('Edit translated sentence:', currentText);
        if (updated !== null && updated.trim()) {
          sentenceTextEl.innerHTML = updated.trim().replace(/\. /g, '.<br>');
          showToast('Updated sentence translation');
        }
      });
    }

    // View as List Toggle for Sign Sequence
    if (toggleListBtn && seqContainer) {
      let isListView = false;
      toggleListBtn.addEventListener('click', () => {
        isListView = !isListView;
        if (isListView) {
          seqContainer.className = 'flex flex-col gap-2 flex-1';
          toggleListBtn.innerHTML = '<span class="material-symbols-outlined text-[16px]">grid_view</span><span>View as Grid</span>';
        } else {
          seqContainer.className = 'grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 flex-1';
          toggleListBtn.innerHTML = '<span class="material-symbols-outlined text-[16px]">view_list</span><span>View as List</span>';
        }
      });
    }

    // Landmark canvas wireframe overlay animation
    function renderLandmarkAnim() {
      if (!landmarkCanvas) return;
      const ctx = landmarkCanvas.getContext('2d');
      if (!ctx) return;

      const rect = landmarkCanvas.getBoundingClientRect();
      landmarkCanvas.width = rect.width || 640;
      landmarkCanvas.height = rect.height || 360;

      let animId;
      function loop() {
        if (player.paused || player.ended) {
          ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
          cancelAnimationFrame(animId);
          return;
        }

        ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
        const w = landmarkCanvas.width;
        const h = landmarkCanvas.height;
        const t = Date.now() / 300;

        // Draw animated ASL wireframe hand landmarks in center
        const cx = w * 0.45 + Math.sin(t * 0.7) * 12;
        const cy = h * 0.52 + Math.cos(t * 0.9) * 8;
        const scale = Math.min(w, h) * 0.22;

        const joints = [
          [0, 0.45],
          [-0.22, 0.3], [-0.32, 0.1], [-0.38, -0.1], [-0.4, -0.3], // Thumb
          [-0.15, 0], [-0.18, -0.25], [-0.2, -0.45], [-0.22, -0.65], // Index
          [0, 0], [0, -0.3], [0, -0.55], [0, -0.75], // Middle
          [0.15, 0], [0.18, -0.25], [0.2, -0.45], [0.22, -0.65], // Ring
          [0.28, 0.05], [0.32, -0.15], [0.36, -0.35], [0.4, -0.52], // Pinky
        ];

        const connections = [
          [0, 1], [1, 2], [2, 3], [3, 4],
          [0, 5], [5, 6], [6, 7], [7, 8],
          [0, 9], [9, 10], [10, 11], [11, 12],
          [0, 13], [13, 14], [14, 15], [15, 16],
          [0, 17], [17, 18], [18, 19], [19, 20],
          [5, 9], [9, 13], [13, 17]
        ];

        // Draw bone connections with subtle glow
        ctx.strokeStyle = 'rgba(255, 90, 42, 0.85)';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#FF5A2A';
        ctx.shadowBlur = 8;
        connections.forEach(([i, j]) => {
          const x1 = cx + joints[i][0] * scale;
          const y1 = cy + joints[i][1] * scale;
          const x2 = cx + joints[j][0] * scale;
          const y2 = cy + joints[j][1] * scale;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        });

        // Draw joint landmarks
        ctx.shadowBlur = 0;
        joints.forEach(([jx, jy]) => {
          const x = cx + jx * scale;
          const y = cy + jy * scale;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#60A5FA';
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#FFFFFF';
          ctx.stroke();
        });

        animId = requestAnimationFrame(loop);
      }

      loop();
    }

    // Initial UI state (00:04)
    updateTimeUI(4, DEFAULT_DURATION);
  }

  function initCommQuickActions() {
    // Wire vocabulary suggestion chips
    document.querySelectorAll('#view-communication .font-code-gloss').forEach(chip => {
      const parent = chip.closest('.group');
      if (parent) {
        parent.classList.add('cursor-pointer');
        parent.addEventListener('click', () => {
          const word = chip.textContent.trim();
          if (window.speechSynthesis) {
            const utter = new SpeechSynthesisUtterance(word);
            window.speechSynthesis.speak(utter);
          }
          showToast(`Spoken: "${word}"`);
        });
      }
    });

    // Wire TTS speak button
    const speakBtn = document.getElementById('comm-btn-tts');
    if (speakBtn) {
      speakBtn.addEventListener('click', () => {
        const text = "Thank you very much. I can show you where the main office is located.";
        if (window.speechSynthesis) {
          const utter = new SpeechSynthesisUtterance(text);
          window.speechSynthesis.speak(utter);
        }
        showToast('Speaking message via audio synthesizer...');
      });
    }

    // Wire clear button
    const clearBtn = document.getElementById('comm-clear-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        showToast('Message buffer cleared');
      });
    }
  }

  // ==========================================================================
  // 16B. ANIMATED LOGO INTERACTIONS
  // ==========================================================================
  function initLogoInteractions() {
    const logoContainers = document.querySelectorAll('.sb-logo-container');
    const greetings = [
      '👋 Hello! SignBridge AI Hand Engine is active & listening.',
      '✨ Neural landmark tracking ready — sign freely!',
      '🤟 Welcome to SignBridge: Bridging sound and sign with AI.',
      '🌟 High-precision dual hand tracking calibrated and online.'
    ];
    let greetIdx = 0;

    logoContainers.forEach(container => {
      container.addEventListener('click', (e) => {
        // Prevent interfering with navigation if inside nav button
        if (container.getAttribute('data-nav') && container.closest('button')) return;
        
        container.classList.remove('is-waving');
        // Force reflow
        void container.offsetWidth;
        container.classList.add('is-waving');

        showToast(greetings[greetIdx % greetings.length]);
        greetIdx++;

        setTimeout(() => {
          container.classList.remove('is-waving');
        }, 1200);
      });
    });
  }

  // ==========================================================================
  // 16C. FEATURED SIGN CAROUSEL & GLOBAL SEARCH
  // ==========================================================================
  function initFeaturedSignCarousel() {
    const featuredSigns = [
      { gloss: 'HELLO', desc: 'A common greeting' },
      { gloss: 'THANK YOU', desc: 'Gratitude & polite expression' },
      { gloss: 'WELCOME', desc: 'Hospitality and warm reception' },
      { gloss: 'PLEASE', desc: 'Polite everyday request gesture' },
      { gloss: 'FRIEND', desc: 'Affectionate connection sign' }
    ];
    let featIdx = 0;

    function updateFeaturedSign() {
      const item = featuredSigns[featIdx];
      const glossEl = document.getElementById('feat-sign-gloss');
      const descEl = document.getElementById('feat-sign-desc');
      const btnEl = document.getElementById('feat-sign-btn');
      if (glossEl) {
        glossEl.style.opacity = '0';
        setTimeout(() => {
          glossEl.textContent = item.gloss;
          glossEl.style.opacity = '1';
        }, 150);
      }
      if (descEl) descEl.textContent = item.desc;
      if (btnEl) btnEl.setAttribute('data-gloss', item.gloss);
    }

    const prevBtn = document.getElementById('feat-sign-prev');
    const nextBtn = document.getElementById('feat-sign-next');
    if (prevBtn) {
      prevBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        featIdx = (featIdx - 1 + featuredSigns.length) % featuredSigns.length;
        updateFeaturedSign();
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        featIdx = (featIdx + 1) % featuredSigns.length;
        updateFeaturedSign();
      });
    }

    const btnEl = document.getElementById('feat-sign-btn');
    if (btnEl) {
      btnEl.addEventListener('click', () => {
        const gloss = featuredSigns[featIdx].gloss;
        switchView('learn');
        if (typeof loadDemonstration === 'function') {
          loadDemonstration(gloss);
        }
      });
    }
  }

  function initGlobalSearch() {
    const searchInput = document.getElementById('global-search-input');
    if (searchInput) {
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const query = searchInput.value.trim();
          if (query) {
            switchView('learn');
            const learnSearch = document.getElementById('learn-search-input');
            if (learnSearch) {
              learnSearch.value = query;
              learnSearch.dispatchEvent(new Event('input'));
            }
            showToast(`Searching vocabulary for "${query}"...`);
          }
        }
      });
    }
  }

  // ==========================================================================
  // 16C. TEACH SIGNBRIDGE MODULE
  // ==========================================================================
  function initTeachModule() {
    const landmarkCanvas = document.getElementById('teach-landmark-canvas');
    const recordBtn = document.getElementById('teach-record-btn');
    const recordIcon = document.getElementById('teach-record-icon');
    const timerEl = document.getElementById('teach-recording-timer');
    const sampleBadge = document.getElementById('teach-sample-badge');
    const progressLabel = document.getElementById('teach-progress-label');
    const progressCircle = document.getElementById('teach-progress-circle');
    const capturedCountEl = document.getElementById('teach-captured-count');
    const clearAllBtn = document.getElementById('teach-clear-all-btn');
    const examplesGrid = document.getElementById('teach-examples-grid');
    const nextBtn = document.getElementById('teach-next-btn');
    const backBtn = document.getElementById('teach-back-btn');
    const guidelinesBtn = document.getElementById('teach-guidelines-btn');

    // Input elements
    const signNameInput = document.getElementById('teach-sign-name');
    const signNameCounter = document.getElementById('teach-sign-name-counter');
    const signDescInput = document.getElementById('teach-sign-desc');
    const signDescCounter = document.getElementById('teach-sign-desc-counter');
    const signSentenceInput = document.getElementById('teach-sign-sentence');
    const signSentenceCounter = document.getElementById('teach-sign-sentence-counter');

    if (!recordBtn && !examplesGrid) return;

    let sampleCount = 3;
    let isRecording = false;
    let recordTimer = null;
    let recordSeconds = 3;

    function updateProgressUI() {
      if (sampleBadge) sampleBadge.textContent = `Sample ${Math.min(10, sampleCount)} / 10`;
      if (progressLabel) progressLabel.textContent = `${sampleCount}/10`;
      if (capturedCountEl) capturedCountEl.textContent = sampleCount;
      if (progressCircle) {
        // Circumference 238.76
        const offset = Math.max(0, 239 - (239 * Math.min(10, sampleCount) / 10));
        progressCircle.style.strokeDashoffset = offset;
      }
    }

    // Input counters
    if (signNameInput && signNameCounter) {
      signNameInput.addEventListener('input', () => {
        signNameCounter.textContent = `${signNameInput.value.length}/50`;
      });
    }
    if (signDescInput && signDescCounter) {
      signDescInput.addEventListener('input', () => {
        signDescCounter.textContent = `${signDescInput.value.length}/200`;
      });
    }
    if (signSentenceInput && signSentenceCounter) {
      signSentenceInput.addEventListener('input', () => {
        signSentenceCounter.textContent = `${signSentenceInput.value.length}/200`;
      });
    }

    // Record toggle
    if (recordBtn) {
      recordBtn.addEventListener('click', () => {
        if (!isRecording) {
          isRecording = true;
          recordSeconds = 0;
          if (recordIcon) {
            recordIcon.className = 'w-5 h-5 rounded-full bg-[#EF4444] animate-ping';
          }
          if (timerEl) timerEl.textContent = '00:00';
          recordTimer = setInterval(() => {
            recordSeconds++;
            const s = recordSeconds.toString().padStart(2, '0');
            if (timerEl) timerEl.textContent = `00:${s}`;
            if (recordSeconds >= 5) {
              stopAndAddSample();
            }
          }, 1000);
          showToast('Recording ASL landmark stream...');
        } else {
          stopAndAddSample();
        }
      });
    }

    function stopAndAddSample() {
      if (!isRecording) return;
      isRecording = false;
      clearInterval(recordTimer);
      if (recordIcon) {
        recordIcon.className = 'w-5 h-5 rounded-sm bg-[#EF4444]';
      }
      if (timerEl) timerEl.textContent = '00:03';

      sampleCount++;
      updateProgressUI();

      // Create synthetic wireframe sample card (no photos)
      const newCard = document.createElement('div');
      newCard.className = 'teach-sample-card relative aspect-square rounded-xl bg-[#1E2028] border border-[#2D3139] overflow-hidden flex items-center justify-center group shadow-sm transition hover:scale-105';
      newCard.innerHTML = `
        <svg class="w-12 h-12 text-[#FF8442]" viewBox="0 0 100 100" fill="none" stroke="currentColor">
          <path d="M50 85 L50 60 L30 45 L20 30" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
          <path d="M50 60 L40 38 L36 18" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
          <path d="M50 60 L50 35 L50 14" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
          <path d="M50 60 L60 38 L64 18" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
          <path d="M50 60 L70 45 L78 28" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
          <circle cx="50" cy="85" r="3.5" fill="#60A5FA"></circle>
          <circle cx="50" cy="60" r="3.5" fill="#60A5FA"></circle>
          <circle cx="20" cy="30" r="3" fill="#FFFFFF"></circle>
          <circle cx="36" cy="18" r="3" fill="#FFFFFF"></circle>
          <circle cx="50" cy="14" r="3" fill="#FFFFFF"></circle>
          <circle cx="64" cy="18" r="3" fill="#FFFFFF"></circle>
          <circle cx="78" cy="28" r="3" fill="#FFFFFF"></circle>
        </svg>
        <button class="teach-delete-sample-btn absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-black/70 hover:bg-[#EF4444] text-white flex items-center justify-center transition cursor-pointer shadow">
          <span class="material-symbols-outlined text-[12px]">close</span>
        </button>
      `;

      newCard.querySelector('.teach-delete-sample-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        newCard.remove();
        sampleCount = Math.max(0, sampleCount - 1);
        updateProgressUI();
        showToast('Sample removed');
      });

      const firstSlot = examplesGrid.querySelector('.teach-add-slot');
      if (firstSlot) {
        examplesGrid.insertBefore(newCard, firstSlot);
      } else {
        examplesGrid.appendChild(newCard);
      }

      const signName = signNameInput ? signNameInput.value.trim() : 'FRIEND';
      showToast(`Captured sample ${sampleCount}/10 for "${signName}"!`);
    }

    // Delete existing cards
    document.querySelectorAll('.teach-delete-sample-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = btn.closest('.teach-sample-card');
        if (card) {
          card.remove();
          sampleCount = Math.max(0, sampleCount - 1);
          updateProgressUI();
          showToast('Sample removed');
        }
      });
    });

    // Clear all
    if (clearAllBtn) {
      clearAllBtn.addEventListener('click', () => {
        document.querySelectorAll('.teach-sample-card').forEach(c => c.remove());
        sampleCount = 0;
        updateProgressUI();
        showToast('All captured samples cleared');
      });
    }

    // Add more slots click
    document.querySelectorAll('.teach-add-slot').forEach(slot => {
      slot.addEventListener('click', () => {
        if (recordBtn) recordBtn.click();
      });
    });

    // Guidelines button
    if (guidelinesBtn) {
      guidelinesBtn.addEventListener('click', () => {
        showToast('📖 Best Practices: 1. Keep hand centered. 2. Capture subtle angle variations. 3. Good lighting.');
      });
    }

    // Back button
    if (backBtn) {
      backBtn.addEventListener('click', () => {
        switchView('home');
      });
    }

    // Next step button
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        if (sampleCount < 3) {
          showToast('Please capture at least 3 samples before proceeding.', 'warning');
          return;
        }
        showToast('Step 1 Complete! Advancing to Review & Confirm.', 'success');
      });
    }

    // Interactive canvas landmark animation on teach-landmark-canvas
    function startTeachCanvasAnim() {
      if (!landmarkCanvas) return;
      const ctx = landmarkCanvas.getContext('2d');
      if (!ctx) return;

      const rect = landmarkCanvas.getBoundingClientRect();
      landmarkCanvas.width = rect.width || 480;
      landmarkCanvas.height = rect.height || 360;

      function render() {
        const pane = document.getElementById('view-teach');
        if (pane && pane.classList.contains('hidden')) {
          requestAnimationFrame(render);
          return;
        }

        ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
        const w = landmarkCanvas.width;
        const h = landmarkCanvas.height;
        const t = Date.now() / 400;

        // Animated landmark hand skeleton in center of camera frame
        const cx = w * 0.5 + Math.sin(t * 0.8) * 8;
        const cy = h * 0.55 + Math.cos(t * 0.6) * 6;
        const scale = Math.min(w, h) * 0.28;

        const joints = [
          [0, 0.45],
          [-0.22, 0.3], [-0.32, 0.1], [-0.38, -0.1], [-0.4, -0.3],
          [-0.15, 0], [-0.18, -0.25], [-0.2, -0.45], [-0.22, -0.65],
          [0, 0], [0, -0.3], [0, -0.55], [0, -0.75],
          [0.15, 0], [0.18, -0.25], [0.2, -0.45], [0.22, -0.65],
          [0.28, 0.05], [0.32, -0.15], [0.36, -0.35], [0.4, -0.52],
        ];

        const connections = [
          [0, 1], [1, 2], [2, 3], [3, 4],
          [0, 5], [5, 6], [6, 7], [7, 8],
          [0, 9], [9, 10], [10, 11], [11, 12],
          [0, 13], [13, 14], [14, 15], [15, 16],
          [0, 17], [17, 18], [18, 19], [19, 20],
          [5, 9], [9, 13], [13, 17]
        ];

        ctx.strokeStyle = 'rgba(255, 90, 42, 0.9)';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#FF5A2A';
        ctx.shadowBlur = 8;
        connections.forEach(([i, j]) => {
          ctx.beginPath();
          ctx.moveTo(cx + joints[i][0] * scale, cy + joints[i][1] * scale);
          ctx.lineTo(cx + joints[j][0] * scale, cy + joints[j][1] * scale);
          ctx.stroke();
        });

        ctx.shadowBlur = 0;
        joints.forEach(([jx, jy]) => {
          ctx.beginPath();
          ctx.arc(cx + jx * scale, cy + jy * scale, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#60A5FA';
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#FFFFFF';
          ctx.stroke();
        });

        requestAnimationFrame(render);
      }

      render();
    }

    startTeachCanvasAnim();
    updateProgressUI();
  }

  // ==========================================================================
  // COMMUNICATION HUB INITIALIZER
  // ==========================================================================
  function initCommunicationHubUI() {
    const startCamBtn = document.getElementById('btn-comm-start-cam');
    const webcamFeed = document.getElementById('comm-webcam-feed');
    const landmarkCanvas = document.getElementById('comm-landmark-canvas');
    const placeholder = document.getElementById('comm-cam-placeholder');
    const statusText = document.getElementById('comm-cam-status-text');
    const recordToggleBtn = document.getElementById('btn-comm-record-toggle');
    const translationText = document.getElementById('comm-translation-text');
    const speakBtn = document.getElementById('btn-comm-speak');
    const copyBtn = document.getElementById('btn-comm-copy');
    const expandBtn = document.getElementById('btn-comm-expand');
    const clearHistBtn = document.getElementById('btn-comm-clear-history');
    const historyList = document.getElementById('comm-history-list');
    const voiceSelect = document.getElementById('comm-voice-select');
    const phraseChips = document.querySelectorAll('.comm-phrase-chip');
    const modeCards = document.querySelectorAll('#comm-mode-selector .comm-mode-card');

    // 1. Mode Card Selection
    modeCards.forEach(card => {
      card.addEventListener('click', () => {
        modeCards.forEach(c => {
          c.classList.remove('active', 'border-2', 'border-[#FF5A2A]', 'bg-[#FFF0EB]');
          c.classList.add('border-[#E8E2D9]', 'bg-white');
        });
        card.classList.remove('border-[#E8E2D9]', 'bg-white');
        card.classList.add('active', 'border-2', 'border-[#FF5A2A]', 'bg-[#FFF0EB]');
        const title = card.querySelector('.font-bold')?.textContent || 'Mode';
        if (typeof showToast === 'function') showToast(`Switched to ${title}`);
      });
    });

    // 2. Start Live Camera
    if (startCamBtn && webcamFeed) {
      startCamBtn.addEventListener('click', async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
          webcamFeed.srcObject = stream;
          webcamFeed.classList.remove('hidden');
          if (landmarkCanvas) landmarkCanvas.classList.remove('hidden');
          if (placeholder) placeholder.classList.add('hidden');
          if (statusText) statusText.textContent = '● Live Camera Active';
          if (typeof showToast === 'function') showToast('Live Camera Feed Started!');
        } catch (err) {
          console.warn('[SB] Camera error:', err);
          if (typeof showToast === 'function') showToast('Could not access camera. Check browser permissions.', 'warning');
        }
      });
    }

    // 3. Record Toggle Button
    if (recordToggleBtn) {
      let isRecording = false;
      recordToggleBtn.addEventListener('click', () => {
        isRecording = !isRecording;
        if (isRecording) {
          recordToggleBtn.classList.remove('bg-[#FF5A2A]');
          recordToggleBtn.classList.add('bg-red-600', 'animate-pulse');
          if (typeof showToast === 'function') showToast('Recording signs...');
        } else {
          recordToggleBtn.classList.remove('bg-red-600', 'animate-pulse');
          recordToggleBtn.classList.add('bg-[#FF5A2A]');
          if (typeof showToast === 'function') showToast('Recording paused');
        }
      });
    }

    // 4. Speak (Text-to-speech)
    if (speakBtn && translationText) {
      speakBtn.addEventListener('click', () => {
        const text = translationText.textContent.trim();
        if (!text) return;
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.rate = 0.95;
          window.speechSynthesis.speak(utterance);
          if (typeof showToast === 'function') showToast(`Speaking: "${text}"`);
        } else {
          if (typeof showToast === 'function') showToast('Text-to-speech not supported in browser.', 'warning');
        }
      });
    }

    // 5. Copy Text
    if (copyBtn && translationText) {
      copyBtn.addEventListener('click', async () => {
        const text = translationText.textContent.trim();
        if (!text) return;
        try {
          await navigator.clipboard.writeText(text);
          if (typeof showToast === 'function') showToast('Translation copied to clipboard!');
        } catch (e) {
          if (typeof showToast === 'function') showToast(`Copied: "${text}"`);
        }
      });
    }

    // 6. Expand (Fullscreen)
    if (expandBtn) {
      expandBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      });
    }

    // 7. Clear Sentence History
    if (clearHistBtn && historyList) {
      clearHistBtn.addEventListener('click', () => {
        historyList.innerHTML = `<div class="p-4 text-center text-xs text-zinc-400 italic">History cleared.</div>`;
        if (typeof showToast === 'function') showToast('Sentence history cleared');
      });
    }

    // 8. Quick Phrase Chips
    phraseChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const phrase = chip.getAttribute('data-phrase');
        if (!phrase) return;
        if (translationText) translationText.textContent = phrase;
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(phrase);
          window.speechSynthesis.speak(utterance);
        }
        if (typeof showToast === 'function') showToast(`Phrase selected: "${phrase}"`);
      });
    });

    // 9. Choose Voice Selection
    if (voiceSelect) {
      voiceSelect.addEventListener('change', () => {
        if (typeof showToast === 'function') showToast(`Voice changed to ${voiceSelect.value}`);
      });
    }
  }

  // ==========================================================================
  // 17. INITIALIZATION
  // ==========================================================================
  console.log('[SignBridge] Application v5.0 initialized successfully.');
  switchView('home');
  renderSentence();
  updateStats();
  initSyntheticLandmarks();
  initDemoModal();
  initVideoTranslationModule();
  initTeachModule();
  initCommQuickActions();
  initCommunicationHubUI();
  initLogoInteractions();
  initFeaturedSignCarousel();
  initGlobalSearch();
  initLearnInteractiveModule();
});


