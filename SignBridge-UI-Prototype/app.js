/* SignBridge prototype — hash router + interactions.
   No build step: this file is loaded directly by index.html. */
(function () {
  'use strict';

  var app = document.getElementById('app');
  var view = document.getElementById('view');
  var screens = Array.prototype.slice.call(document.querySelectorAll('[data-screen]'));
  var navItems = Array.prototype.slice.call(document.querySelectorAll('.nav-item'));

  var ROUTES = ['home','live','video','teach','learn','practice','communication','profile','settings',
    'onboarding/welcome','onboarding/camera','onboarding/preferences','onboarding/ready'];
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------- theme ---------------- */
  var themeToggle = document.getElementById('themeToggle');
  var themeToggleIcon = themeToggle.querySelector('use');
  var mq = window.matchMedia('(prefers-color-scheme: dark)');

  function storedTheme() {
    try { return localStorage.getItem('sb-theme') || 'system'; } catch (e) { return 'system'; }
  }
  function effectiveTheme(mode) {
    return mode === 'system' ? (mq.matches ? 'dark' : 'light') : mode;
  }
  function applyTheme(mode) {
    var eff = effectiveTheme(mode);
    document.documentElement.dataset.theme = eff;
    document.documentElement.style.colorScheme = eff;
    themeToggleIcon.setAttribute('href', eff === 'dark' ? '#i-sun' : '#i-moon');
    themeToggle.setAttribute('aria-pressed', String(eff === 'dark'));
    themeToggle.setAttribute('aria-label', eff === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    document.querySelectorAll('[data-theme-set]').forEach(function (b) {
      var on = b.getAttribute('data-theme-set') === mode;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-checked', String(on));
    });
  }
  function setTheme(mode) {
    try { localStorage.setItem('sb-theme', mode); } catch (e) {}
    applyTheme(mode);
  }
  themeToggle.addEventListener('click', function () {
    setTheme(effectiveTheme(storedTheme()) === 'dark' ? 'light' : 'dark');
  });
  if (mq.addEventListener) {
    mq.addEventListener('change', function () { if (storedTheme() === 'system') applyTheme('system'); });
  }
  applyTheme(storedTheme());

  /* theme chips and generic radio groups are handled by delegation so they
     keep working when the settings panel is re-rendered */

  /* ---------------- toasts ---------------- */
  var toastWrap = document.getElementById('toasts');
  function toast(msg, icon) {
    var el = document.createElement('div');
    el.className = 'toast';
    el.setAttribute('role', 'status');
    el.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#' + (icon || 'i-check') + '"/></svg><span></span>';
    el.querySelector('span').textContent = msg;
    toastWrap.appendChild(el);
    window.setTimeout(function () { el.remove(); }, 2600);
  }

  /* ---------------- modal ---------------- */
  var modalRoot = document.getElementById('modalRoot');
  var modalOpener = null;
  function openModal(opts) {
    modalOpener = document.activeElement;
    var scrim = document.createElement('div');
    scrim.className = 'modal-scrim';
    scrim.innerHTML =
      '<div class="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">' +
        '<h2 id="modalTitle"></h2><p></p><div class="actions">' +
        '<button class="btn btn-secondary" data-close>Cancel</button>' +
        '<button class="btn ' + (opts.danger ? 'btn-danger' : 'btn-primary') + '" data-confirm></button>' +
      '</div></div>';
    scrim.querySelector('h2').textContent = opts.title;
    scrim.querySelector('p').textContent = opts.body;
    scrim.querySelector('[data-confirm]').textContent = opts.confirm || 'Confirm';
    modalRoot.appendChild(scrim);
    var lastFocus = null;
    scrim.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { close(); }
      if (e.key === 'Tab') {
        var f = scrim.querySelectorAll('button');
        if (!f.length) return;
        var first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });
    function close() {
      scrim.remove();
      document.removeEventListener('keydown', onKey);
      if (modalOpener && modalOpener.focus) modalOpener.focus();
    }
    function onKey(e) { if (e.key === 'Escape') close(); }
    document.addEventListener('keydown', onKey);
    scrim.querySelector('[data-close]').addEventListener('click', close);
    scrim.addEventListener('click', function (e) { if (e.target === scrim) close(); });
    scrim.querySelector('[data-confirm]').addEventListener('click', function () {
      close();
      if (opts.onConfirm) opts.onConfirm();
    });
    lastFocus = scrim.querySelector('[data-confirm]');
    lastFocus.focus();
  }

  /* ---------------- router ---------------- */
  var TITLES = {};
  screens.forEach(function (s) { TITLES[s.dataset.screen] = s.dataset.title; });

  function currentRoute() {
    var h = (location.hash || '').replace(/^#\/?/, '');
    return ROUTES.indexOf(h) >= 0 ? h : 'home';
  }

  function render() {
    var route = currentRoute();
    var onboarding = route.indexOf('onboarding/') === 0;
    app.classList.toggle('is-onboarding', onboarding);

    screens.forEach(function (s) { s.hidden = s.dataset.screen !== route; });
    var active = document.querySelector('[data-screen="' + route + '"]');

    navItems.forEach(function (b) {
      var on = b.dataset.route === route;
      b.classList.toggle('is-active', on);
      if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
    });

    document.title = (TITLES[route] ? TITLES[route] + ' · ' : '') + 'SignBridge';
    if (view) view.scrollTop = 0;
    window.scrollTo(0, 0);
    if (active && !onboarding) animateCounters(active);
  }

  window.addEventListener('hashchange', render);

  document.addEventListener('click', function (e) {
    var themeEl = e.target.closest('[data-theme-set]');
    if (themeEl) { setTheme(themeEl.getAttribute('data-theme-set')); return; }

    var routeEl = e.target.closest('[data-route]');
    if (routeEl) { e.preventDefault(); location.hash = '#/' + routeEl.dataset.route; return; }

    var group = e.target.closest('[role="radiogroup"]');
    if (group) {
      var radio = e.target.closest('[role="radio"]');
      if (radio && group.contains(radio) && radio.id !== 'themeChoice') {
        group.querySelectorAll('[role="radio"]').forEach(function (r) {
          var on = r === radio;
          r.setAttribute('aria-checked', String(on));
          r.classList.toggle('is-active', on);
        });
        return;
      }
    }

    var toastEl = e.target.closest('[data-toast]');
    if (toastEl) { toast(toastEl.dataset.toast); }
  });

  /* ---------------- counters ---------------- */
  function animateCounters(scope) {
    if (reduceMotion) return;
    scope.querySelectorAll('[data-count]').forEach(function (el) {
      if (el.dataset.done) return;
      el.dataset.done = '1';
      var target = parseInt(el.dataset.count, 10);
      if (isNaN(target)) return;
      var start = performance.now(), dur = 620;
      function step(now) {
        var p = Math.min(1, (now - start) / dur);
        var eased = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(target * eased);
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }

  /* ---------------- notifications popover ---------------- */
  var notifBtn = document.getElementById('notifBtn');
  var notifPop = document.getElementById('notifPop');
  notifBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    var open = notifPop.hidden;
    notifPop.hidden = !open;
    notifBtn.setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('click', function (e) {
    if (!notifPop.hidden && !e.target.closest('.popover-wrap')) {
      notifPop.hidden = true;
      notifBtn.setAttribute('aria-expanded', 'false');
    }
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !notifPop.hidden) {
      notifPop.hidden = true;
      notifBtn.setAttribute('aria-expanded', 'false');
      notifBtn.focus();
    }
  });

  /* ---------------- global search ---------------- */
  var SEARCH = [
    { route: 'home', label: 'Home', icon: 'i-home', keys: 'home workspace dashboard overview' },
    { route: 'live', label: 'Live Translate', icon: 'i-cam', keys: 'live translate camera real-time signing' },
    { route: 'video', label: 'Video Translate', icon: 'i-video', keys: 'video translate upload file mp4' },
    { route: 'teach', label: 'Teach SignBridge', icon: 'i-teach', keys: 'teach add sign record personal' },
    { route: 'learn', label: 'Learn', icon: 'i-book', keys: 'learn lessons course vocabulary' },
    { route: 'practice', label: 'Practice', icon: 'i-target', keys: 'practice drill recognize exercises' },
    { route: 'communication', label: 'Communication Hub', icon: 'i-chat', keys: 'communication conversation two-way chat' },
    { route: 'profile', label: 'My Profile', icon: 'i-user', keys: 'profile progress achievements streak' },
    { route: 'settings', label: 'Settings', icon: 'i-settings', keys: 'settings camera audio model accessibility privacy' },
    { route: 'onboarding/welcome', label: 'Onboarding tour', icon: 'i-sparkle', keys: 'tour welcome setup onboarding' }
  ];
  var searchInput = document.getElementById('globalSearch');
  var searchResults = document.getElementById('searchResults');

  function closeSearch() { searchResults.hidden = true; searchResults.innerHTML = ''; }
  function runSearch(q) {
    q = q.trim().toLowerCase();
    if (!q) { closeSearch(); return; }
    var hits = SEARCH.filter(function (s) {
      return (s.label + ' ' + s.keys).toLowerCase().indexOf(q) >= 0;
    }).slice(0, 6);
    searchResults.innerHTML = '';
    if (!hits.length) {
      var empty = document.createElement('p');
      empty.className = 'search-empty';
      empty.textContent = 'No sections match “' + q + '”. Try “translate”, “learn”, or “settings”.';
      searchResults.appendChild(empty);
    } else {
      hits.forEach(function (s, i) {
        var b = document.createElement('button');
        if (i === 0) b.className = 'is-current';
        b.innerHTML = '<svg class="ic" style="width:16px;height:16px;color:var(--orange)" aria-hidden="true"><use href="#' + s.icon + '"/></svg><span></span>';
        b.querySelector('span').textContent = s.label;
        b.addEventListener('click', function () { location.hash = '#/' + s.route; searchInput.value = ''; closeSearch(); });
        searchResults.appendChild(b);
      });
    }
    searchResults.hidden = false;
  }
  searchInput.addEventListener('input', function () { runSearch(searchInput.value); });
  searchInput.addEventListener('focus', function () { if (searchInput.value) runSearch(searchInput.value); });
  searchInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      var first = searchResults.querySelector('button');
      if (first) { first.click(); }
      else if (searchResults.hidden === false) { /* keep open */ }
    }
    if (e.key === 'Escape') { searchInput.value = ''; closeSearch(); searchInput.blur(); }
  });
  document.addEventListener('click', function (e) {
    if (!e.target.closest('.search')) closeSearch();
  });

  /* ---------------- Live Translate ---------------- */
  var liveWord = document.getElementById('liveWord');
  var liveSub = document.getElementById('liveSub');
  var liveCaption = document.getElementById('liveCaption');
  var captionHistory = [];
  var correctionRow = document.getElementById('correctionRow');
  if (correctionRow) {
    correctionRow.addEventListener('click', function (e) {
      var tile = e.target.closest('.sign-tile');
      if (!tile) return;
      correctionRow.querySelectorAll('.sign-tile').forEach(function (t) {
        var on = t === tile;
        t.classList.toggle('is-active', on);
        t.setAttribute('aria-checked', String(on));
      });
      liveWord.textContent = tile.dataset.word;
      liveSub.textContent = tile.dataset.sub;
      captionHistory.push(liveCaption.textContent);
      liveCaption.textContent = tile.dataset.sub.replace(/ · .*$/, '') + ', how are you?';
      toast('Corrected to “' + tile.dataset.word + '”');
    });
  }
  var liveEdit = document.getElementById('liveEdit');
  if (liveEdit) {
    liveEdit.addEventListener('click', function () {
      var editing = liveCaption.getAttribute('contenteditable') === 'true';
      liveCaption.setAttribute('contenteditable', String(!editing));
      if (!editing) { liveCaption.focus(); liveEdit.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-check"/></svg>Done'; }
      else { liveEdit.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-edit"/></svg>Edit'; toast('Translation updated'); }
    });
  }
  var liveUndo = document.getElementById('liveUndo');
  if (liveUndo) {
    liveUndo.addEventListener('click', function () {
      if (captionHistory.length) { liveCaption.textContent = captionHistory.pop(); toast('Undone', 'i-undo'); }
      else toast('Nothing to undo', 'i-info');
    });
  }
  var liveCopy = document.getElementById('liveCopy');
  if (liveCopy) {
    liveCopy.addEventListener('click', function () {
      copyText(liveCaption.textContent);
    });
  }
  var liveSpeak = document.getElementById('liveSpeak');
  if (liveSpeak) {
    liveSpeak.addEventListener('click', function () { speak(liveCaption.textContent, liveSpeak); });
  }

  /* ---------------- quick phrases ---------------- */
  document.querySelectorAll('.phrase').forEach(function (chip) {
    chip.addEventListener('click', function () {
      if (liveCaption) {
        captionHistory.push(liveCaption.textContent);
        liveCaption.textContent = chip.textContent.trim() + '!';
      }
      var b = document.querySelector('#commModes .bubble.them p');
      if (b) b.textContent = chip.textContent.trim() + '!';
      toast('Inserted “' + chip.textContent.trim() + '”', 'i-plus');
    });
  });

  /* ---------------- Video Translate ---------------- */
  var videoCaption = document.getElementById('videoCaption');
  var videoTimeline = document.getElementById('videoTimeline');
  if (videoTimeline) {
    videoTimeline.addEventListener('click', function (e) {
      var frame = e.target.closest('.frame');
      if (!frame) return;
      videoTimeline.querySelectorAll('.frame').forEach(function (f) { f.classList.toggle('is-active', f === frame); });
      videoCaption.textContent = frame.dataset.caption;
      toast('Jumped to ' + frame.querySelector('b').textContent + ' · 00:0' + (Array.prototype.indexOf.call(videoTimeline.children, frame) * 2 + 2));
    });
  }
  var videoEdit = document.getElementById('videoEdit');
  if (videoEdit) {
    videoEdit.addEventListener('click', function () {
      var editing = videoCaption.getAttribute('contenteditable') === 'true';
      videoCaption.setAttribute('contenteditable', String(!editing));
      if (!editing) { videoCaption.focus(); videoEdit.textContent = 'Done'; }
      else { videoEdit.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-edit"/></svg>Edit Text'; toast('Translation updated'); }
    });
  }
  bindCopy('videoCopy', videoCaption);
  var videoSpeak = document.getElementById('videoSpeak');
  if (videoSpeak) videoSpeak.addEventListener('click', function () { speak(videoCaption.textContent, videoSpeak); });

  /* ---------------- Teach ---------------- */
  var teachRec = 3, teachTimer = null;
  var teachLabel = document.getElementById('teachRecLabel');
  var teachDot = document.getElementById('teachDot');
  var teachGallery = document.getElementById('teachGallery');
  var teachAdd = document.getElementById('teachAdd');
  if (teachAdd) {
    teachAdd.addEventListener('click', function () {
      if (teachRec >= 10) { toast('That’s the maximum of 10 examples', 'i-info'); return; }
      teachRec++;
      var shot = document.createElement('div');
      shot.className = 'shot';
      shot.innerHTML = '<img src="assets/feed-live.jpg" alt="Captured example of the sign." style="object-position:' +
        (30 + teachRec * 5) + '% 45%"><button class="rm" data-remove aria-label="Remove example"><svg class="ic" aria-hidden="true"><use href="#i-x"/></svg></button>';
      teachGallery.insertBefore(shot, teachAdd);
      var ring = document.querySelector('#screen-teach .ring');
      if (ring) { ring.style.setProperty('--p', Math.round(teachRec / 10 * 100)); ring.querySelector('.big').textContent = teachRec + '/10'; }
      var badge = document.getElementById('teachCountBadge');
      if (badge) badge.textContent = teachRec + '/10';
      toast('Example ' + teachRec + ' captured', 'i-check');
    });
  }
  document.addEventListener('click', function (e) {
    var rm = e.target.closest('[data-remove]');
    if (rm) {
      var shot = rm.closest('.shot');
      if (shot) { shot.remove(); teachRec = Math.max(0, teachRec - 1); toast('Example removed', 'i-x'); }
    }
  });
  var teachNext = document.getElementById('teachNext');
  if (teachNext) {
    teachNext.addEventListener('click', function () {
      var steps = document.querySelectorAll('#screen-teach .step');
      var done = 0;
      steps.forEach(function (s) { if (s.classList.contains('is-done')) done++; });
      if (done < 2) {
        steps[done].classList.remove('is-active');
        steps[done].classList.add('is-done');
        if (steps[done + 1]) { steps[done + 1].classList.add('is-active'); }
        toast('Step ' + (done + 2) + ' of 4 — ' + steps[Math.min(done + 1, 3)].textContent.replace(/^\d/, '').trim(), 'i-arrow-r');
      } else {
        toast('Sign submitted for review — thank you!', 'i-check');
      }
    });
  }
  if (teachLabel) {
    var t = 3;
    teachTimer = window.setInterval(function () {
      if (document.querySelector('[data-screen="teach"]').hidden) return;
      t++;
      teachLabel.textContent = 'Recording · 00:' + (t < 10 ? '0' : '') + t;
    }, 1000);
  }

  /* ---------------- Learn ---------------- */
  bindChipRow('learnChips', function (chip) { toast('Showing “' + chip.textContent.trim() + '” lessons'); });
  var lessonList = document.getElementById('lessonList');
  if (lessonList) {
    lessonList.addEventListener('click', function (e) {
      var item = e.target.closest('.lesson-item');
      if (!item) return;
      lessonList.querySelectorAll('.lesson-item').forEach(function (i) { i.classList.toggle('is-active', i === item); });
      var name = item.textContent.replace(/^\d+/, '').trim();
      var stage = document.querySelector('#screen-learn .lesson-stage');
      stage.querySelector('.kicker').textContent = 'Lesson ' + item.querySelector('.idx').textContent;
      stage.querySelector('h2').textContent = name;
    });
  }

  /* ---------------- Practice ---------------- */
  bindChipRow('practiceModes', function (chip) { toast('Mode: ' + chip.textContent.trim()); });
  var setList = document.getElementById('setList');
  if (setList) {
    setList.addEventListener('click', function (e) {
      var item = e.target.closest('.set-item');
      if (!item) return;
      setList.querySelectorAll('.set-item').forEach(function (i) { i.classList.remove('is-active'); });
      item.classList.add('is-active');
      toast('Practice set: ' + item.querySelector('b').textContent);
    });
  }
  var pWord = document.getElementById('practiceWord');
  var pFeedback = document.getElementById('practiceFeedback');
  var pProgress = document.querySelector('#screen-practice .progress-bar-row');
  var pPill = document.querySelector('#screen-practice .rec-pill:last-child');
  var pCount = 3;
  var WORDS = ['HELLO', 'THANK YOU', 'PLEASE', 'FRIEND', 'GOOD', 'YES', 'NO', 'SORRY', 'HELP', 'GOODBYE'];
  function practiceAdvance() {
    pWord.textContent = WORDS[pCount % WORDS.length];
    pCount++;
    if (pProgress) {
      var bar = pProgress.querySelector('.bar i');
      var pct = Math.min(100, pCount / 10 * 100);
      if (bar) bar.style.width = pct + '%';
      var b = pProgress.querySelector('b');
      if (b) b.textContent = Math.min(10, pCount) + '/10';
    }
    if (pPill) pPill.textContent = 'Sign ' + Math.min(10, pCount) + ' of 10';
    pFeedback.classList.add('ok');
    pFeedback.querySelector('.word').textContent = pWord.textContent;
    toast('Nice — “' + pWord.textContent + '” recognized', 'i-check');
  }
  var practiceNext = document.getElementById('practiceNext');
  if (practiceNext) practiceNext.addEventListener('click', practiceAdvance);
  var practiceRetry = document.getElementById('practiceRetry');
  if (practiceRetry) practiceRetry.addEventListener('click', function () { toast('Camera ready — try again', 'i-undo'); });

  /* ---------------- Communication ---------------- */
  var commModes = document.getElementById('commModes');
  if (commModes) {
    commModes.addEventListener('click', function (e) {
      var mode = e.target.closest('.mode');
      if (!mode) return;
      commModes.querySelectorAll('.mode').forEach(function (m) {
        var on = m === mode;
        m.classList.toggle('is-active', on);
        m.setAttribute('aria-selected', String(on));
      });
      var label = document.getElementById('commModeLabel');
      if (label) {
        var map = { 'Sign → Text': 'Listening for signs', 'Text → Speech': 'Ready to speak your text', 'Speech → Text': 'Listening for speech', 'Conversation': 'Two-way conversation live' };
        label.textContent = map[mode.dataset.mode] || 'Listening';
      }
      toast('Mode: ' + mode.dataset.mode);
    });
  }
  var commSpeak = document.getElementById('commSpeak');
  if (commSpeak) commSpeak.addEventListener('click', function () { speak('Hello, how are you?', commSpeak); });
  var commConnect = document.getElementById('commConnect');
  if (commConnect) {
    commConnect.addEventListener('click', function () {
      openModal({
        title: 'Start a conversation?',
        body: 'This opens the two-way conversation surface. You can switch modes at any time — nothing is recorded without your say-so.',
        confirm: 'Start',
        onConfirm: function () { toast('Conversation started — you’re live', 'i-users'); }
      });
    });
  }

  /* ---------------- Settings ---------------- */
  var settingsCats = document.getElementById('settingsCats');
  var centerCard = document.querySelector('.settings-grid > .card:nth-child(2)');
  var generalHTML = centerCard ? centerCard.innerHTML : '';

  var PANELS = {
    'Camera': { title: 'Camera', desc: 'Video input settings.', rows: [
      { b: 'Mirror my camera', p: 'Flip the preview so signing feels natural.', on: true },
      { b: 'Auto-adjust exposure', p: 'Balance lighting for clearer hand detection.', on: true },
      { b: 'HD video (1080p)', p: 'Higher detail uses more battery.', on: true },
      { b: 'On-device processing', p: 'Recognize signs without sending video anywhere.', on: true }
    ] },
    'Audio': { title: 'Audio', desc: 'Microphone and speech.', rows: [
      { b: 'Auto-play translations', p: 'Speak translated text aloud automatically.', on: true },
      { b: 'Noise reduction', p: 'Reduce background sound for clearer speech.', on: true },
      { b: 'Voice', p: 'Pick the voice used for spoken output.', action: 'Natural (Default)' }
    ] },
    'AI Model': { title: 'AI Model', desc: 'Recognition preferences.', rows: [
      { b: 'Prefer my personal signs', p: 'Use signs you have taught SignBridge first.', on: true },
      { b: 'Continuous recognition', p: 'Keep recognizing without pressing anything.', on: true },
      { b: 'Model', p: 'SignNet v3.1 · 2,731 classes · v3.1.0', action: 'Details' }
    ] },
    'Accessibility': { title: 'Accessibility', desc: 'Make it yours.', rows: [
      { b: 'Reduce motion', p: 'Minimize animations for a calmer experience.', on: false },
      { b: 'High-contrast captions', p: 'Increase caption contrast and size.', on: false },
      { b: 'Larger text', p: 'Scale interface text up.', on: false },
      { b: 'Captions always visible', p: 'Keep the caption panel on screen.', on: true }
    ] },
    'Notifications': { title: 'Notifications', desc: 'Alerts and updates.', rows: [
      { b: 'Practice reminders', p: 'A gentle nudge to keep your streak.', on: true },
      { b: 'New lessons', p: 'Tell me when a lesson unlocks.', on: true },
      { b: 'Community', p: 'Updates from the SignBridge community.', on: false }
    ] },
    'Privacy & Data': { title: 'Privacy & Data', desc: 'Control your data.', rows: [
      { b: 'Save translation history', p: 'Keep a history of your translations.', on: true },
      { b: 'On-device recognition', p: 'Process signs locally, never in the cloud.', on: true },
      { b: 'Allow model improvement', p: 'Share anonymized examples to improve recognition.', on: false }
    ] },
    'Devices': { title: 'Devices', desc: 'Connected devices.', rows: [
      { b: 'MacBook Pro Microphone', p: 'Connected · system default', action: 'Test' },
      { b: 'FaceTime HD Camera', p: 'Connected · built-in', action: 'Test' },
      { b: 'Pair a new device', p: 'Use SignBridge on another device.', action: 'Pair' }
    ] },
    'Help & Support': { title: 'Help & Support', desc: 'Get assistance.', rows: [
      { b: 'Getting started guide', p: 'Learn the basics in two minutes.', action: 'Open' },
      { b: 'Camera troubleshooting', p: 'Fix common camera issues.', action: 'Open' },
      { b: 'Contact support', p: 'We usually reply within a day.', action: 'Open' }
    ] }
  };

  function renderPanel(name) {
    if (!centerCard) return;
    if (name === 'General') { centerCard.innerHTML = generalHTML; applyTheme(storedTheme()); return; }
    var cfg = PANELS[name];
    if (!cfg) return;
    var html = '<div class="setting-group" style="margin-bottom:0"><h3>' + cfg.title + '</h3><p>' + cfg.desc + '</p>';
    cfg.rows.forEach(function (r) {
      html += '<div class="setting-row"><div class="meta"><b>' + r.b + '</b><p>' + r.p + '</p></div>';
      if (r.action) html += '<button class="btn btn-secondary" data-toast="' + r.action + ' · ' + r.b + '">' + r.action + '</button>';
      else html += '<label class="switch"><input type="checkbox" ' + (r.on ? 'checked' : '') + ' aria-label="' + r.b + '"><span class="track"></span></label>';
      html += '</div>';
    });
    centerCard.innerHTML = html + '</div>';
  }

  if (settingsCats) {
    settingsCats.addEventListener('click', function (e) {
      var item = e.target.closest('.cat-item');
      if (!item) return;
      settingsCats.querySelectorAll('.cat-item').forEach(function (c) {
        var on = c === item;
        c.classList.toggle('is-active', on);
        c.setAttribute('aria-selected', String(on));
      });
      var name = item.querySelector('b').textContent;
      renderPanel(name);
      if (name !== 'General') toast(name + ' settings');
    });
  }

  var signOutBtn = document.getElementById('signOutBtn');
  if (signOutBtn) {
    signOutBtn.addEventListener('click', function () {
      openModal({
        title: 'Sign out of SignBridge?',
        body: 'You’ll need to sign in again to translate, practise, and keep your streak going. Your on-device data stays on this device.',
        confirm: 'Sign out',
        danger: true,
        onConfirm: function () { toast('Signed out — see you soon', 'i-check'); }
      });
    });
  }

  var onbFinish = document.getElementById('onbFinish');
  if (onbFinish) onbFinish.addEventListener('click', function () { location.hash = '#/home'; toast('Welcome to SignBridge', 'i-sparkle'); });

  /* ---------------- helpers ---------------- */
  function copyText(text) {
    var t = text.trim();
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(t).then(function () { toast('Copied to clipboard', 'i-copy'); },
        function () { toast('Copy is unavailable here', 'i-info'); });
    } else {
      toast('Copy is unavailable here', 'i-info');
    }
  }
  function bindCopy(id, el) {
    var b = document.getElementById(id);
    if (b && el) b.addEventListener('click', function () { copyText(el.textContent); });
  }
  function speak(text, btn) {
    if (!('speechSynthesis' in window)) { toast('Speech is unavailable in this preview', 'i-info'); return; }
    try {
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(text.trim());
      u.rate = 0.98;
      window.speechSynthesis.speak(u);
      toast('Speaking…', 'i-speaker');
    } catch (e) { toast('Speech is unavailable in this preview', 'i-info'); }
  }
  function bindChipRow(id, cb) {
    var row = document.getElementById(id);
    if (!row) return;
    row.addEventListener('click', function (e) {
      var chip = e.target.closest('.chip');
      if (!chip) return;
      row.querySelectorAll('.chip').forEach(function (c) { c.classList.toggle('is-active', c === chip); });
      if (cb) cb(chip);
    });
  }

  /* ---------------- boot ---------------- */
  /* keep an accessible name on nav items when the icon-rail hides the labels */
  navItems.forEach(function (b) {
    var label = b.querySelector('.label');
    if (label && !b.getAttribute('aria-label')) b.setAttribute('aria-label', label.textContent.trim());
  });
  if (!location.hash) location.replace('#/home');
  render();
  if (teachTimer) { /* keep reference alive */ }
})();
