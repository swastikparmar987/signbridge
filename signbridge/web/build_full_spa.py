#!/usr/bin/env python3
"""
SignBridge Single Page Application Generator
Builds the complete frontend combining all 10 Stitch screens into signbridge_p/web/templates/index.html
"""

import os
import re
from bs4 import BeautifulSoup

ROOT_DIR = "/Users/pro/Desktop/projects/signbridge"
CODE_DIR = os.path.join(ROOT_DIR, "stitch_designs/code")
TEMPLATE_PATH = os.path.join(ROOT_DIR, "signbridge/web/templates/index.html")

def read_stitch_file(filename):
    path = os.path.join(CODE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_main_soup(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    main = soup.find("main")
    # The SPA shell already provides a shared #topbar; drop each screen's own copy.
    if main is not None:
        for hdr in main.find_all("header", attrs={"data-purpose": "top-navigation-bar"}):
            hdr.decompose()
    return main

print("Loading stitch screens...")
dash_soup = extract_main_soup(read_stitch_file("03_dashboard.html"))
live_soup = extract_main_soup(read_stitch_file("04_live_translate.html"))
video_soup = extract_main_soup(read_stitch_file("05_video_translate.html"))
teach_soup = extract_main_soup(read_stitch_file("06_teach.html"))
learn_soup = extract_main_soup(read_stitch_file("07_learn.html"))
practice_soup = extract_main_soup(read_stitch_file("08_practice.html"))
comm_soup = extract_main_soup(read_stitch_file("09_communication.html"))
settings_soup = extract_main_soup(read_stitch_file("10_settings.html"))
profile_soup = extract_main_soup(read_stitch_file("11_profile.html"))
welcome_soup = extract_main_soup(read_stitch_file("02_welcome.html"))

# 1. Adapt Dashboard:
dash_html = str(dash_soup)
# Add IDs to hero buttons
dash_html = dash_html.replace('Start Live Translate</span>', 'Start Live Translate</span>', 1)
dash_soup = BeautifulSoup(dash_html, "html.parser")
for btn in dash_soup.find_all("button"):
    text = btn.get_text()
    if "Start Live Translate" in text:
        btn["id"] = "btn-hero-start-live"
        btn["data-nav"] = "live"
    elif "Upload a Video" in text:
        btn["id"] = "btn-hero-upload-video"
        btn["data-nav"] = "video"
# Quick actions
for a in dash_soup.find_all(["a", "button"]):
    t = a.get_text().strip()
    if "Live Translate" in t:
        a["data-nav"] = "live"
    elif "Video Translate" in t:
        a["data-nav"] = "video"
    elif "Teach SignBridge" in t:
        a["data-nav"] = "teach"
    elif "Practice" in t:
        a["data-nav"] = "practice"
    elif "Communication" in t:
        a["data-nav"] = "communication"
    elif "Learn" in t and "Start" not in t:
        a["data-nav"] = "learn"

# 2. Adapt Live Translate:
live_html = str(live_soup)
live_soup = BeautifulSoup(live_html, "html.parser")

# Locate the viewfinder card
viewfinder = None
for div in live_soup.find_all("div"):
    cls = div.get("class", [])
    if any("bg-[#14161B]" in c for c in cls):
        viewfinder = div
        break

if viewfinder:
    # Inject webcam video and canvas
    video_tag = live_soup.new_tag("video", id="webcam-feed", playsinline="", muted="", autoplay="",
                                  style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover; border-radius:1rem; display:none;")
    canvas_tag = live_soup.new_tag("canvas", id="landmark-canvas",
                                   style="position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:10; border-radius:1rem; display:none;")
    viewfinder.insert(0, canvas_tag)
    viewfinder.insert(0, video_tag)

    # Find the prompt in viewfinder to wrap as camera-prompt-overlay
    prompt_div = viewfinder.find("div", class_=lambda c: c and "flex-col items-center justify-center" in c)
    if prompt_div:
        prompt_div["id"] = "camera-prompt-overlay"
        # Add start camera button inside prompt
        start_btn = live_soup.new_tag("button", id="btn-start-camera", type="button",
                                      class_="mt-3 inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#FF5A2A] to-[#FF7744] hover:brightness-105 active:scale-95 text-white font-semibold text-xs shadow-lg shadow-[#FF5A2A]/30 transition cursor-pointer")
        start_btn.string = "Start Camera"
        prompt_div.append(start_btn)

# Locate primary recognition word in Live Translate
for h2 in live_soup.find_all(["h2", "h3", "h4", "div"]):
    if h2.string and h2.string.strip() == "HELLO":
        h2["id"] = "primary-gloss"
        break

# Locate English text under it
for p in live_soup.find_all("p"):
    if p.string and "Common Greeting" in p.string:
        p["id"] = "primary-english"
        break

# Audio pronounce button
for btn in live_soup.find_all("button"):
    if btn.find("span", string=lambda s: s and "volume_up" in s):
        btn["id"] = "btn-pronounce-sign"
        break

# Locate alternatives container (the 5 tiles)
for div in live_soup.find_all("div"):
    cls = " ".join(div.get("class", []))
    if "grid-cols-5" in cls or "gap-2" in cls and div.find_all("div", class_=lambda c: c and "rounded-xl" in c):
        div["id"] = "alt-candidates-grid"
        break

# Locate sentence composer text area
for div in live_soup.find_all("div"):
    txt = div.get_text()
    if "Hello how are you?" in txt and len(div.find_all("span")) >= 2:
        div["id"] = "sentence-composer-text"
        break

# Live composer buttons (Undo, Copy, Speak, Clear)
for btn in live_soup.find_all("button"):
    txt = btn.get_text().strip().lower()
    span_mat = btn.find("span", class_="material-symbols-outlined")
    mat_txt = span_mat.get_text().strip() if span_mat else ""
    if "undo" in txt or mat_txt == "undo":
        btn["id"] = "sb-btn-undo"
    elif "copy" in txt or mat_txt == "content_copy":
        btn["id"] = "sb-btn-copy"
    elif "speak" in txt or mat_txt == "volume_up" or "listen" in txt:
        btn["id"] = "sb-btn-speak"
    elif "clear" in txt or mat_txt == "close" or mat_txt == "clear":
        btn["id"] = "sb-btn-clear"

# Session stats
for div in live_soup.find_all(["div", "span"]):
    txt = div.get_text().strip()
    if txt == "12" and not div.get("id"):
        div["id"] = "live-stat-signs"
    elif txt == "1" and not div.get("id"):
        div["id"] = "live-stat-sentences"
    elif txt == "95%" and not div.get("id"):
        div["id"] = "live-stat-confidence"

# 3. Adapt Video Translate:
video_html = str(video_soup)
video_soup = BeautifulSoup(video_html, "html.parser")

# Find the upload dropzone
dropzone = video_soup.find("div", class_=lambda c: c and "custom-dashed-border" in c)
if dropzone:
    dropzone["id"] = "video-dropzone"
    # Create file input
    file_input = video_soup.new_tag("input", id="video-file-input", type="file",
                                    accept="video/mp4,video/webm,video/quicktime,video/avi",
                                    style="display:none;")
    dropzone.append(file_input)

# Find Browse button
for btn in video_soup.find_all("button"):
    if "Upload Video" in btn.get_text():
        btn["id"] = "btn-browse-video"

# Find video preview or result container
for p in video_soup.find_all(["p", "div"]):
    if "HELLO HOW ARE YOU" in p.get_text() or "Confidence: 94%" in p.get_text():
        p["id"] = "video-translation-result"
        break

# 4. Adapt Teach:
teach_html = str(teach_soup)
teach_soup = BeautifulSoup(teach_html, "html.parser")

# Find camera viewport in teach
teach_vp = teach_soup.find("div", {"data-purpose": "camera-viewport-card"})
if teach_vp:
    t_video = teach_soup.new_tag("video", id="teach-webcam-feed", playsinline="", muted="", autoplay="",
                                 style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover; border-radius:1rem; display:none;")
    t_canvas = teach_soup.new_tag("canvas", id="teach-landmark-canvas",
                                  style="position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:10; border-radius:1rem; display:none;")
    teach_vp.insert(0, t_canvas)
    teach_vp.insert(0, t_video)

for btn in teach_soup.find_all("button"):
    txt = btn.get_text().strip()
    if "Record" in txt and ("Sample" in txt or "Start" in txt):
        btn["id"] = "btn-record-sample"

# 5. Adapt Learn:
learn_html = str(learn_soup)
learn_soup = BeautifulSoup(learn_html, "html.parser")

for inp in learn_soup.find_all("input"):
    if "search" in inp.get("placeholder", "").lower():
        inp["id"] = "learn-vocab-search"

# Category buttons
for btn in learn_soup.find_all("button"):
    t = btn.get_text().strip().lower()
    if any(k in t for k in ["all", "phrases", "greetings", "numbers", "alphabet", "family", "emotions", "food"]):
        btn["class"] = btn.get("class", []) + ["learn-cat-btn"]
        btn["data-cat"] = t

# Vocabulary cards container
for div in learn_soup.find_all("div"):
    cls = " ".join(div.get("class", []))
    if "grid-cols-2" in cls or "grid-cols-3" in cls or "gap-3" in cls:
        if len(div.find_all("div")) >= 4 and not div.get("id"):
            div["id"] = "vocab-list-container"
            break

# 6. Adapt Practice:
practice_html = str(practice_soup)
practice_soup = BeautifulSoup(practice_html, "html.parser")

for h2 in practice_soup.find_all(["h2", "h3", "h4"]):
    txt = h2.get_text().strip()
    if txt in ["HELLO", "THANK YOU", "APPLE"]:
        h2["id"] = "practice-target-title"
        break

# 7. Adapt Communication:
comm_html = str(comm_soup)
comm_soup = BeautifulSoup(comm_html, "html.parser")

for ta in comm_soup.find_all("textarea"):
    ta["id"] = "comm-text-input"

for btn in comm_soup.find_all("button"):
    txt = btn.get_text().strip().lower()
    if "speech" in txt or "mic" in txt:
        btn["id"] = "comm-btn-mic"
    elif "send" in txt:
        btn["id"] = "comm-btn-send"

# 8. Adapt Settings:
settings_html = str(settings_soup)
settings_soup = BeautifulSoup(settings_html, "html.parser")

for btn in settings_soup.find_all("button"):
    txt = btn.get_text().strip().lower()
    if txt == "light":
        btn["id"] = "settings-theme-light"
    elif txt == "dark":
        btn["id"] = "settings-theme-dark"
    elif txt == "system":
        btn["id"] = "settings-theme-system"
    elif "clear cache" in txt:
        btn["id"] = "settings-clear-cache-btn"

# 9. Adapt Profile:
profile_html = str(profile_soup)
profile_soup = BeautifulSoup(profile_html, "html.parser")

# Build the complete unified template
INDEX_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SignBridge — Different Signs. Same Human Connection.</title>
  <meta name="description" content="AI-powered real-time sign language recognition, translation, learning, and inclusive communication suite.">
  <link rel="icon" type="image/png" href="/static/assets/logo.png">

  <!-- Google Fonts: Plus Jakarta Sans, Inter, Permanent Marker -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&family=Permanent+Marker&display=swap" rel="stylesheet">
  
  <!-- Material Symbols Outlined -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200">

  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          fontFamily: {{
            sans: ['"Plus Jakarta Sans"', 'Inter', 'sans-serif'],
            marker: ['"Permanent Marker"', 'cursive'],
          }},
          colors: {{
            brand: {{
              orange: '#FF5A2A',
              orangeHover: '#E54C1E',
              orangeLight: '#FFF0EB',
              orangeGlow: 'rgba(255, 90, 42, 0.35)',
              bg: '#F7F1E7',
              pagebg: '#F8F7F4',
              surface: '#FFFDF8',
              surfaceSoft: '#F2E9DC',
              dark: '#111318',
              darker: '#0B0D11',
              border: '#E6DED3',
              success: '#2EB875',
              warning: '#F6B73C',
              error: '#EF4444',
            }}
          }}
        }}
      }}
    }}
  </script>

  <!-- MediaPipe Hands & Camera Utils -->
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

  <!-- Custom Stylesheet -->
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body class="bg-[#F7F1E7] text-[#111318] antialiased selection:bg-[#FF5A2A] selection:text-white min-h-screen flex flex-col overflow-hidden">

  <!-- Skip Navigation -->
  <a href="#views-container" class="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:px-4 focus:py-2 focus:bg-[#FF5A2A] focus:text-white focus:rounded-lg">
    Skip to main content
  </a>

  <!-- APP SHELL -->
  <div class="flex-1 flex w-full h-screen overflow-hidden" id="app-shell">

    <!-- LEFT SIDEBAR NAVIGATION -->
    <aside class="w-64 bg-[#111318] text-white flex flex-col justify-between shrink-0 min-h-screen py-6 px-4 select-none border-r border-neutral-800/80 z-20" id="app-sidebar" data-purpose="sidebar-navigation">
      <div class="space-y-6">
        <!-- Brand Logo Header -->
        <div class="flex items-center gap-3 px-2 cursor-pointer" id="brand-home-link" data-nav="home">
          <div class="w-10 h-10 rounded-full bg-white flex items-center justify-center p-1.5 shadow-lg shadow-[#FF5A2A]/20 shrink-0">
            <img alt="SignBridge Logo" class="w-full h-full object-contain" src="/static/assets/logo.png">
          </div>
          <div class="leading-tight">
            <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
              SignBridge
            </h1>
            <p class="text-[10px] text-gray-400 font-medium leading-none mt-0.5" id="sidebar-tagline">
              Different Signs.<br><span class="text-gray-400">Same Human Connection.</span>
            </p>
          </div>
        </div>

        <!-- Main Navigation Links -->
        <nav aria-label="Main Navigation" class="space-y-1.5 pt-1" id="sidebar-nav">
          <!-- 1. Home -->
          <a class="nav-link flex items-center gap-3.5 px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#FF5A2A] to-[#FF7744] text-white font-semibold shadow-md shadow-orange-950/20 text-sm cursor-pointer transition" data-nav="home">
            <svg class="w-5 h-5 fill-current" viewBox="0 0 24 24"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"></path></svg>
            <span>Home</span>
          </a>

          <!-- 2. Live Translate -->
          <a class="nav-link flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="live">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" stroke-linecap="round" stroke-linejoin="round"></path></svg>
            <span>Live Translate</span>
          </a>

          <!-- 3. Video Translate -->
          <a class="nav-link flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="video">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect height="16" rx="3" width="20" x="2" y="4"></rect><polygon fill="currentColor" points="10 8 16 12 10 16 10 8" stroke="none"></polygon></svg>
            <span>Video Translate</span>
          </a>

          <!-- 4. Teach SignBridge -->
          <a class="nav-link flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="teach">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 9v6m3-3H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" stroke-linecap="round" stroke-linejoin="round"></path></svg>
            <span>Teach SignBridge</span>
          </a>

          <!-- 5. Learn -->
          <a class="nav-link flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="learn">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" stroke-linecap="round" stroke-linejoin="round"></path></svg>
            <span>Learn</span>
          </a>

          <!-- 6. Practice -->
          <a class="nav-link flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="practice">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"></circle><circle cx="12" cy="12" r="5"></circle><circle cx="12" cy="12" fill="currentColor" r="1.5"></circle></svg>
            <span>Practice</span>
          </a>

          <!-- 7. Communication -->
          <a class="nav-link flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="communication">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" stroke-linecap="round" stroke-linejoin="round"></path></svg>
            <span>Communication</span>
          </a>
        </nav>

        <!-- Divider Navigation -->
        <div class="pt-4 border-t border-gray-800/80 space-y-1.5">
          <a class="nav-link flex items-center gap-3.5 px-4 py-2 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="settings">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" stroke-linecap="round" stroke-linejoin="round"></path><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" stroke-linecap="round" stroke-linejoin="round"></path></svg>
            <span>Settings</span>
          </a>
          <a class="nav-link flex items-center gap-3.5 px-4 py-2 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition font-medium text-sm cursor-pointer" data-nav="profile">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" stroke-linecap="round" stroke-linejoin="round"></path></svg>
            <span>Profile</span>
          </a>
        </div>
      </div>

      <!-- Bottom Mission Card & Onboarding Trigger -->
      <div class="mt-4 pt-2">
        <div class="rounded-2xl overflow-hidden border border-white/10 relative shadow-lg group cursor-pointer" id="btn-relaunch-onboarding" title="Take the Setup Tour">
          <img src="/static/images/sidebar_mountain_dark.png" alt="A more inclusive world, together" class="w-full h-auto object-cover block rounded-2xl group-hover:scale-105 transition duration-300">
          <div class="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent p-3 flex flex-col justify-end">
            <span class="text-[11px] font-bold text-white tracking-wide">A more inclusive world</span>
            <span class="text-[9px] text-orange-300 flex items-center gap-1 mt-0.5 font-medium">✨ Take Setup Tour &rarr;</span>
          </div>
        </div>
      </div>
    </aside>

    <!-- MAIN VIEWPORT AREA -->
    <div class="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">

      <!-- TOP NAVIGATION BAR -->
      <header class="flex items-center justify-between gap-4 px-8 py-4 bg-[#F7F1E7]/90 backdrop-blur-md border-b border-[#E6DED3]/60 shrink-0 z-10" id="topbar">
        <!-- Global Search Input with Instant Dropdown -->
        <div class="flex-1 max-w-xl relative">
          <span class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" x2="16.65" y1="21" y2="16.65"></line></svg>
          </span>
          <input id="global-search-input" class="w-full pl-10 pr-4 py-2 bg-white/80 hover:bg-white focus:bg-white text-xs text-gray-800 placeholder-gray-400 rounded-xl border border-[#E6DED3] focus:border-[#FF5A2A] focus:ring-1 focus:ring-[#FF5A2A] transition duration-150 outline-none" placeholder="Search 2,731 signs, lessons, or dictionary..." type="text">
          <div id="global-search-results" class="absolute left-0 right-0 top-full mt-1.5 bg-white border border-[#E6DED3] rounded-xl shadow-xl max-h-80 overflow-y-auto hidden z-50"></div>
        </div>

        <!-- Right Utility Actions -->
        <div class="flex items-center gap-3 shrink-0">


          <!-- Day/Night Appearance Toggle -->
          <button class="w-9 h-9 rounded-xl bg-white border border-[#E6DED3] flex items-center justify-center text-gray-700 hover:text-black hover:border-gray-300 shadow-xs transition cursor-pointer" id="theme-toggle-btn" title="Toggle Light / Dark Mode">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
          </button>

          <!-- Notification Bell -->
          <div class="relative">
            <button class="w-9 h-9 rounded-xl bg-white border border-[#E6DED3] flex items-center justify-center text-gray-700 hover:text-black hover:border-gray-300 shadow-xs relative transition cursor-pointer" id="notifications-btn" title="Notifications">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" stroke-linecap="round" stroke-linejoin="round"></path></svg>
              <span class="w-2 h-2 rounded-full bg-[#FF5A2A] absolute top-2 right-2"></span>
            </button>
            <div id="notifications-dropdown" class="absolute right-0 top-full mt-2 w-72 bg-white rounded-xl shadow-xl border border-[#E6DED3] p-3 text-xs hidden z-50">
              <div class="font-bold text-gray-900 pb-2 border-b border-gray-100 flex items-center justify-between">
                <span>Notifications</span>
                <span class="text-[10px] text-[#FF5A2A] font-semibold">2 new</span>
              </div>
              <div class="space-y-2 pt-2">
                <div class="p-2 rounded-lg bg-orange-50/60 border border-orange-100">
                  <div class="font-bold text-neutral-800">18-Day Streak Active!</div>
                  <div class="text-neutral-500 text-[11px]">Practice 3 signs today to keep it burning.</div>
                </div>
                <div class="p-2 rounded-lg bg-neutral-50 border border-neutral-100">
                  <div class="font-bold text-neutral-800">ArcFace 2,731 Signs Loaded</div>
                  <div class="text-neutral-500 text-[11px]">Apple Silicon MPS acceleration active.</div>
                </div>
              </div>
            </div>
          </div>

          <!-- User Profile Pill -->
          <div class="flex items-center gap-2.5 pl-1 cursor-pointer" id="user-profile-menu" data-nav="profile">
            <div class="w-8 h-8 rounded-full bg-[#181a20] text-white flex items-center justify-center font-bold text-xs shadow-xs">
              SP
            </div>
            <div class="text-left text-xs leading-snug hidden md:block">
              <div class="font-bold text-gray-900 flex items-center gap-1">
                <span>Swastik Parmar</span>
                <svg class="w-3 h-3 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path></svg>
              </div>
              <div class="text-gray-500 text-[10px]">Student · Active</div>
            </div>
          </div>
        </div>
      </header>

      <!-- VIEWS CONTAINER -->
      <main class="flex-1 overflow-y-auto min-w-0" id="views-container">

        <!-- 1. VIEW HOME (Dashboard) -->
        <section class="view-pane" id="view-home">
          {str(dash_soup)}
        </section>

        <!-- 2. VIEW LIVE (Live Translate) -->
        <section class="view-pane hidden" id="view-live">
          {str(live_soup)}
        </section>

        <!-- 3. VIEW VIDEO (Video Translate) -->
        <section class="view-pane hidden" id="view-video">
          {str(video_soup)}
        </section>

        <!-- 4. VIEW TEACH (Teach SignBridge) -->
        <section class="view-pane hidden" id="view-teach">
          {str(teach_soup)}
        </section>

        <!-- 5. VIEW LEARN (Learn) -->
        <section class="view-pane hidden" id="view-learn">
          {str(learn_soup)}
        </section>

        <!-- 6. VIEW PRACTICE (Practice) -->
        <section class="view-pane hidden" id="view-practice">
          {str(practice_soup)}
        </section>

        <!-- 7. VIEW COMMUNICATION (Communication) -->
        <section class="view-pane hidden" id="view-communication">
          {str(comm_soup)}
        </section>

        <!-- 8. VIEW SETTINGS (Settings) -->
        <section class="view-pane hidden" id="view-settings">
          {str(settings_soup)}
        </section>

        <!-- 9. VIEW PROFILE (Profile) -->
        <section class="view-pane hidden" id="view-profile">
          {str(profile_soup)}
        </section>

      </main>
    </div>
  </div>

  <!-- ONBOARDING FLOW MODAL / OVERLAY (Steps 1 to 4) -->
  <div id="onboarding-overlay" class="fixed inset-0 z-50 bg-[#F7F1E7] overflow-y-auto flex flex-col justify-between hidden">
    <!-- Header with Stepper -->
    <div class="p-6 max-w-4xl mx-auto w-full flex items-center justify-between border-b border-[#E6DED3]">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-full bg-white flex items-center justify-center p-1 shadow-xs">
          <img src="/static/assets/logo.png" alt="Logo" class="w-full h-full object-contain">
        </div>
        <span class="font-bold text-neutral-900 text-sm">SignBridge Setup</span>
      </div>

      <!-- Stepper Badges -->
      <div class="flex items-center gap-3 text-xs font-semibold" id="onboarding-stepper">
        <div class="stepper-step flex items-center gap-1.5 text-[#FF5A2A]" data-step="1">
          <span class="w-6 h-6 rounded-full bg-[#FF5A2A] text-white flex items-center justify-center text-xs">1</span>
          <span>Welcome</span>
        </div>
        <div class="w-8 h-px bg-[#E6DED3]"></div>
        <div class="stepper-step flex items-center gap-1.5 text-neutral-400" data-step="2">
          <span class="w-6 h-6 rounded-full bg-neutral-200 text-neutral-600 flex items-center justify-center text-xs">2</span>
          <span>Camera</span>
        </div>
        <div class="w-8 h-px bg-[#E6DED3]"></div>
        <div class="stepper-step flex items-center gap-1.5 text-neutral-400" data-step="3">
          <span class="w-6 h-6 rounded-full bg-neutral-200 text-neutral-600 flex items-center justify-center text-xs">3</span>
          <span>Preferences</span>
        </div>
        <div class="w-8 h-px bg-[#E6DED3]"></div>
        <div class="stepper-step flex items-center gap-1.5 text-neutral-400" data-step="4">
          <span class="w-6 h-6 rounded-full bg-neutral-200 text-neutral-600 flex items-center justify-center text-xs">4</span>
          <span>Ready</span>
        </div>
      </div>

      <button id="btn-skip-onboarding" class="text-xs font-semibold text-neutral-500 hover:text-neutral-900 cursor-pointer">
        Skip for now &rarr;
      </button>
    </div>

    <!-- Step 1: Welcome -->
    <div id="onboarding-step-1" class="onboarding-content max-w-4xl mx-auto w-full p-8 flex flex-col justify-center flex-1">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
        <div>
          <span class="text-xs font-extrabold tracking-wider text-[#FF5A2A] uppercase">WELCOME TO SIGNBRIDGE</span>
          <h1 class="text-4xl font-extrabold text-[#111318] tracking-tight mt-2 leading-tight">
            Different Signs.<br><span class="text-[#FF5A2A]">Same Human Connection.</span>
          </h1>
          <p class="text-sm text-neutral-600 mt-4 leading-relaxed">
            Understand, learn, and communicate using sign language — powered by advanced AI and real-time computer vision with 2,731 supported signs.
          </p>
          <div class="mt-8 flex items-center gap-3">
            <button id="btn-welcome-next" class="px-6 py-3 rounded-xl bg-[#FF5A2A] hover:bg-[#E54C1E] text-white text-xs font-bold shadow-md shadow-orange-950/20 transition cursor-pointer">
              Get Started &rarr;
            </button>
          </div>
        </div>
        <div class="rounded-3xl overflow-hidden shadow-xl border border-orange-200/60 bg-[#faeee5]">
          <img src="/static/assets/hero.jpg" alt="Hands Connecting" class="w-full h-auto object-cover">
        </div>
      </div>
    </div>

    <!-- Step 2: Camera Setup -->
    <div id="onboarding-step-2" class="onboarding-content max-w-4xl mx-auto w-full p-8 flex flex-col justify-center flex-1 hidden">
      <div class="text-center max-w-md mx-auto mb-6">
        <span class="text-xs font-extrabold tracking-wider text-[#FF5A2A] uppercase">STEP 2 OF 4</span>
        <h2 class="text-2xl font-bold text-neutral-900 mt-1">Set Up Your Camera</h2>
        <p class="text-xs text-neutral-600 mt-1">Position yourself so both hands and face are comfortably in frame.</p>
      </div>
      <div class="max-w-lg mx-auto w-full bg-[#111318] rounded-2xl p-4 shadow-xl border border-neutral-800 text-center relative overflow-hidden aspect-video flex flex-col justify-center items-center">
        <video id="onboarding-webcam-feed" playsinline muted autoplay class="absolute inset-0 w-full h-full object-cover hidden"></video>
        <div id="onboarding-cam-fallback" class="text-neutral-400 space-y-2">
          <span class="material-symbols-outlined text-4xl text-[#FF5A2A]">videocam</span>
          <p class="text-xs text-neutral-300">Camera preview will appear here</p>
          <button id="btn-onboarding-test-cam" class="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-xs font-semibold rounded-lg border border-neutral-700 cursor-pointer">Test Camera</button>
        </div>
      </div>
      <div class="mt-6 flex justify-between max-w-lg mx-auto w-full">
        <button id="btn-cam-back" class="px-5 py-2.5 rounded-xl border border-[#E6DED3] text-xs font-semibold text-neutral-700 hover:bg-neutral-100 cursor-pointer">&larr; Back</button>
        <button id="btn-cam-continue" class="px-6 py-2.5 rounded-xl bg-[#FF5A2A] text-white text-xs font-bold hover:bg-[#E54C1E] shadow-sm cursor-pointer">Continue &rarr;</button>
      </div>
    </div>

    <!-- Step 3: Preferences -->
    <div id="onboarding-step-3" class="onboarding-content max-w-4xl mx-auto w-full p-8 flex flex-col justify-center flex-1 hidden">
      <div class="text-center max-w-md mx-auto mb-6">
        <span class="text-xs font-extrabold tracking-wider text-[#FF5A2A] uppercase">STEP 3 OF 4</span>
        <h2 class="text-2xl font-bold text-neutral-900 mt-1">Your Preferences</h2>
        <p class="text-xs text-neutral-600 mt-1">Choose how you want SignBridge to work best for you.</p>
      </div>
      <div class="max-w-md mx-auto w-full space-y-4 bg-white p-6 rounded-2xl border border-[#E6DED3] shadow-sm">
        <div>
          <label class="block text-xs font-bold text-neutral-800 mb-1">Primary Sign Language</label>
          <select class="w-full text-xs p-2.5 rounded-xl border border-neutral-300 bg-neutral-50 outline-none">
            <option selected>American Sign Language (ASL - 2,731 signs)</option>
            <option>British Sign Language (BSL)</option>
            <option>Indian Sign Language (ISL)</option>
          </select>
        </div>
        <div>
          <label class="block text-xs font-bold text-neutral-800 mb-1">Voice Feedback (TTS)</label>
          <select class="w-full text-xs p-2.5 rounded-xl border border-neutral-300 bg-neutral-50 outline-none">
            <option selected>Natural English (US)</option>
            <option>Natural English (UK)</option>
          </select>
        </div>
        <div class="flex items-center justify-between pt-2">
          <span class="text-xs font-bold text-neutral-800">Show Hand Tracking Mesh</span>
          <input type="checkbox" checked class="rounded text-[#FF5A2A] focus:ring-[#FF5A2A]">
        </div>
      </div>
      <div class="mt-6 flex justify-between max-w-md mx-auto w-full">
        <button id="btn-pref-back" class="px-5 py-2.5 rounded-xl border border-[#E6DED3] text-xs font-semibold text-neutral-700 hover:bg-neutral-100 cursor-pointer">&larr; Back</button>
        <button id="btn-pref-continue" class="px-6 py-2.5 rounded-xl bg-[#FF5A2A] text-white text-xs font-bold hover:bg-[#E54C1E] shadow-sm cursor-pointer">Continue &rarr;</button>
      </div>
    </div>

    <!-- Step 4: Ready -->
    <div id="onboarding-step-4" class="onboarding-content max-w-4xl mx-auto w-full p-8 flex flex-col justify-center flex-1 text-center hidden">
      <div class="max-w-md mx-auto space-y-4">
        <div class="w-16 h-16 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center mx-auto text-3xl">
          🎉
        </div>
        <h2 class="text-3xl font-extrabold text-neutral-900">You're All Set!</h2>
        <p class="text-xs text-neutral-600 leading-relaxed">
          SignBridge is fully calibrated and ready. You can translate live signs, practice vocabulary, upload videos, or teach custom signs anytime.
        </p>
        <button id="btn-start-exploring" class="mt-4 px-8 py-3 rounded-xl bg-[#FF5A2A] hover:bg-[#E54C1E] text-white text-xs font-bold shadow-lg shadow-orange-950/20 transition cursor-pointer">
          Enter SignBridge &rarr;
        </button>
      </div>
    </div>

    <div class="p-4 text-center text-[11px] text-neutral-400 border-t border-[#E6DED3]">
      SignBridge v4.0 · Powered by Deep Learning & Computer Vision
    </div>
  </div>

  <!-- DIAGNOSTIC HUD DRAWER -->
  <aside id="diagnostic-hud" class="fixed right-0 top-0 bottom-0 w-80 bg-[#111318] text-white shadow-2xl z-40 transform translate-x-full transition-transform duration-300 p-5 flex flex-col justify-between border-l border-neutral-800">
    <div class="space-y-4">
      <div class="flex items-center justify-between pb-3 border-b border-neutral-800">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <h3 class="font-bold text-sm text-white">Diagnostic HUD</h3>
        </div>
        <button id="btn-close-diag" class="w-7 h-7 rounded-lg bg-neutral-800 hover:bg-neutral-700 flex items-center justify-center text-neutral-400 hover:text-white cursor-pointer">&times;</button>
      </div>
      <div class="space-y-3 text-xs">
        <div class="flex justify-between py-1 border-b border-neutral-800/60">
          <span class="text-neutral-400">Model:</span>
          <span class="font-bold text-white" id="diag-model">ArcFace Exp7</span>
        </div>
        <div class="flex justify-between py-1 border-b border-neutral-800/60">
          <span class="text-neutral-400">Latency:</span>
          <span class="font-bold text-emerald-400" id="diag-latency">-- ms</span>
        </div>
        <div class="flex justify-between py-1 border-b border-neutral-800/60">
          <span class="text-neutral-400">Hands Tracked:</span>
          <span class="font-bold text-white" id="diag-hands">0</span>
        </div>
        <div class="flex justify-between py-1 border-b border-neutral-800/60">
          <span class="text-neutral-400">Sequence Buffer:</span>
          <span class="font-bold text-white" id="diag-buffer">0 / 32</span>
        </div>
        <div class="flex justify-between py-1 border-b border-neutral-800/60">
          <span class="text-neutral-400">Logit Margin:</span>
          <span class="font-bold text-white" id="diag-margin">--</span>
        </div>
        <div class="flex justify-between py-1 border-b border-neutral-800/60">
          <span class="text-neutral-400">Acceleration:</span>
          <span class="font-bold text-orange-400">Apple Silicon MPS</span>
        </div>
      </div>
    </div>
    <div class="p-3 bg-neutral-900 rounded-xl border border-neutral-800 text-[11px] text-neutral-400">
      Press <kbd class="px-1.5 py-0.5 bg-neutral-800 rounded text-neutral-300 font-mono">D</kbd> to toggle HUD anytime.
    </div>
  </aside>

  <!-- TOAST NOTIFICATION CONTAINER -->
  <div id="toast-container" class="fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-none"></div>

  <!-- APPLICATION JAVASCRIPT -->
  <script src="/static/js/nlp.js"></script>
  <script src="/static/js/app.js"></script>
</body>
</html>
"""

print("Writing template to:", TEMPLATE_PATH)
with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
    f.write(INDEX_TEMPLATE)

print("Successfully written index.html! File size:", os.path.getsize(TEMPLATE_PATH), "bytes")
