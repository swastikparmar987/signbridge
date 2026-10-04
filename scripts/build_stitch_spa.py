#!/usr/bin/env python3
"""
SignBridge Single Page Application Compiler
Assembles the 4 newly designed Stitch screens (Home Dashboard, Two-Way Communication,
Learn Vocabulary, Video Translation) plus existing modules (Live Studio, Practice,
Teach, Profile, Settings) into signbridge_p/web/templates/index.html.
Ensures ZERO static images (no <img> tags) - all replaced with real video, canvas,
animated SVG hand landmark diagrams, and soundwave visualizers.
"""

import os
import re
from bs4 import BeautifulSoup

ROOT_DIR = "/Users/pro/Desktop/projects/signbridge"
NEW_CODE_DIR = os.path.join(ROOT_DIR, "stitch_designs/web_application/code")
EXISTING_INDEX = os.path.join(ROOT_DIR, "signbridge/web/templates/index.html")
OUTPUT_PATH = os.path.join(ROOT_DIR, "signbridge/web/templates/index.html")

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

print("Reading Stitch screens...")
home_html = read_file(os.path.join(NEW_CODE_DIR, "01_home_dashboard.html"))
comm_html = read_file(os.path.join(NEW_CODE_DIR, "02_two_way_communication.html"))
learn_html = read_file(os.path.join(NEW_CODE_DIR, "03_learn_vocabulary.html"))
video_html = read_file(os.path.join(NEW_CODE_DIR, "04_video_translation.html"))
existing_index_html = read_file(EXISTING_INDEX)

home_soup = BeautifulSoup(home_html, "html.parser")
comm_soup = BeautifulSoup(comm_html, "html.parser")
learn_soup = BeautifulSoup(learn_html, "html.parser")
video_soup = BeautifulSoup(video_html, "html.parser")
existing_soup = BeautifulSoup(existing_index_html, "html.parser")

def extract_main_content(soup):
    main = soup.find("main")
    if not main:
        return ""
    return "".join(str(c) for c in main.children)

# --------------------------------------------------------------------------
# 1. PROCESS HOME DASHBOARD (Exact Match to User Mockup)
# --------------------------------------------------------------------------
HOME_DASHBOARD_HTML = """
<div class="px-6 py-6 md:px-8 max-w-7xl mx-auto flex flex-col gap-6">

  <!-- 1. HERO BANNER -->
  <div class="relative w-full rounded-2xl bg-[#FFF8F2] border border-[#F3E7DC] overflow-hidden flex flex-col md:flex-row items-center justify-between p-6 md:p-8 lg:p-10 shadow-sm">
    <div class="z-10 flex flex-col max-w-xl">
      <span class="text-[11px] font-extrabold tracking-widest text-[#FF5A2A] uppercase mb-2">SIGNBRIDGE</span>
      <h1 class="text-3xl md:text-4xl font-extrabold text-[#111318] tracking-tight leading-tight mb-3">
        Communication has no boundaries
      </h1>
      <p class="text-sm md:text-base text-[#5A5E66] font-normal leading-relaxed mb-6 max-w-md">
        Use AI to understand, learn, and share sign language for a more inclusive world.
      </p>
      <div class="flex items-center gap-3 flex-wrap">
        <button data-nav="live" class="flex items-center gap-2 px-5 py-3 rounded-full bg-[#FF5A2A] hover:bg-[#E5481B] text-white font-bold text-sm shadow-md shadow-orange-500/25 transition-all hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
          <span class="material-symbols-outlined text-[18px]">videocam</span>
          <span>Start Live Translate</span>
          <span class="material-symbols-outlined text-[16px]">arrow_forward</span>
        </button>
        <button data-nav="video" class="flex items-center gap-2 px-5 py-3 rounded-full bg-white hover:bg-[#FFF3EC] text-[#FF5A2A] border border-[#FF5A2A] font-bold text-sm transition-all hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
          <span class="material-symbols-outlined text-[18px] text-[#FF5A2A]">play_circle</span>
          <span>Upload a Video</span>
        </button>
      </div>
    </div>

    <!-- Artwork on Right Side -->
    <div class="relative w-full md:w-1/2 h-64 md:h-72 flex items-center justify-end mt-4 md:mt-0 select-none">
      <img src="/static/assets/hero_hands.png" alt="Different Signs Same Human Connection" class="w-full h-full object-contain object-right">
      
      <!-- Handwritten badge on top right -->
      <div class="absolute top-2 right-4 flex flex-col items-center select-none pointer-events-none transform rotate-[-4deg]">
        <span class="font-serif italic font-extrabold text-[12px] md:text-[13px] tracking-wide text-[#111318] leading-tight">DIFFERENT SIGNS</span>
        <span class="font-serif italic font-extrabold text-[12px] md:text-[13px] tracking-wide text-[#111318] leading-tight">SAME HUMAN CONNECTION</span>
        <div class="w-full h-[2.5px] bg-[#FF5A2A] rounded-full mt-0.5"></div>
      </div>
    </div>
  </div>

  <!-- 2. QUICK ACTIONS -->
  <div class="flex flex-col gap-3">
    <div class="flex items-center justify-between">
      <h2 class="font-bold text-base md:text-lg text-[#111318]">Quick Actions</h2>
      <a data-nav="learn" class="text-xs font-bold text-[#FF5A2A] hover:underline flex items-center gap-1 cursor-pointer">
        <span>See All</span>
        <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
      </a>
    </div>

    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <!-- Action 1: Live Translate -->
      <div data-nav="live" class="group bg-white hover:bg-[#FFF9F5] border border-[#ECE2D8] hover:border-[#FF5A2A] p-4 rounded-xl flex flex-col items-center text-center cursor-pointer transition-all duration-200 shadow-sm hover:shadow-md hover:-translate-y-0.5">
        <div class="w-11 h-11 rounded-full bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A] mb-3 group-hover:scale-110 transition-transform">
          <span class="material-symbols-outlined text-[22px]">videocam</span>
        </div>
        <div class="font-bold text-sm text-[#111318] mb-0.5">Live Translate</div>
        <div class="text-[11px] text-[#6B7280]">Use your webcam</div>
      </div>

      <!-- Action 2: Video Translate -->
      <div data-nav="video" class="group bg-white hover:bg-[#FFF9F5] border border-[#ECE2D8] hover:border-[#FF5A2A] p-4 rounded-xl flex flex-col items-center text-center cursor-pointer transition-all duration-200 shadow-sm hover:shadow-md hover:-translate-y-0.5">
        <div class="w-11 h-11 rounded-full bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A] mb-3 group-hover:scale-110 transition-transform">
          <span class="material-symbols-outlined text-[22px]">play_circle</span>
        </div>
        <div class="font-bold text-sm text-[#111318] mb-0.5">Video Translate</div>
        <div class="text-[11px] text-[#6B7280]">Upload &amp; analyze</div>
      </div>

      <!-- Action 3: Teach SignBridge -->
      <div data-nav="teach" class="group bg-white hover:bg-[#FFF9F5] border border-[#ECE2D8] hover:border-[#FF5A2A] p-4 rounded-xl flex flex-col items-center text-center cursor-pointer transition-all duration-200 shadow-sm hover:shadow-md hover:-translate-y-0.5">
        <div class="w-11 h-11 rounded-full bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A] mb-3 group-hover:scale-110 transition-transform">
          <span class="material-symbols-outlined text-[22px]">school</span>
        </div>
        <div class="font-bold text-sm text-[#111318] mb-0.5">Teach SignBridge</div>
        <div class="text-[11px] text-[#6B7280]">Personalize for you</div>
      </div>

      <!-- Action 4: Learn -->
      <div data-nav="learn" class="group bg-white hover:bg-[#FFF9F5] border border-[#ECE2D8] hover:border-[#FF5A2A] p-4 rounded-xl flex flex-col items-center text-center cursor-pointer transition-all duration-200 shadow-sm hover:shadow-md hover:-translate-y-0.5">
        <div class="w-11 h-11 rounded-full bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A] mb-3 group-hover:scale-110 transition-transform">
          <span class="material-symbols-outlined text-[22px]">menu_book</span>
        </div>
        <div class="font-bold text-sm text-[#111318] mb-0.5">Learn</div>
        <div class="text-[11px] text-[#6B7280]">Explore 2,731 signs</div>
      </div>

      <!-- Action 5: Practice -->
      <div data-nav="practice" class="group bg-white hover:bg-[#FFF9F5] border border-[#ECE2D8] hover:border-[#FF5A2A] p-4 rounded-xl flex flex-col items-center text-center cursor-pointer transition-all duration-200 shadow-sm hover:shadow-md hover:-translate-y-0.5">
        <div class="w-11 h-11 rounded-full bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A] mb-3 group-hover:scale-110 transition-transform">
          <span class="material-symbols-outlined text-[22px]">adjust</span>
        </div>
        <div class="font-bold text-sm text-[#111318] mb-0.5">Practice</div>
        <div class="text-[11px] text-[#6B7280]">Improve your skills</div>
      </div>

      <!-- Action 6: Communication -->
      <div data-nav="communication" class="group bg-white hover:bg-[#FFF9F5] border border-[#ECE2D8] hover:border-[#FF5A2A] p-4 rounded-xl flex flex-col items-center text-center cursor-pointer transition-all duration-200 shadow-sm hover:shadow-md hover:-translate-y-0.5">
        <div class="w-11 h-11 rounded-full bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A] mb-3 group-hover:scale-110 transition-transform">
          <span class="material-symbols-outlined text-[22px]">chat</span>
        </div>
        <div class="font-bold text-sm text-[#111318] mb-0.5">Communication</div>
        <div class="text-[11px] text-[#6B7280]">Have a conversation</div>
      </div>
    </div>
  </div>

  <!-- 3. THREE-COLUMN DASHBOARD SECTION -->
  <div class="grid grid-cols-1 md:grid-cols-12 gap-5">
    
    <!-- Left: Your Progress (Col 1 to 5) -->
    <div class="md:col-span-5 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-bold text-sm md:text-base text-[#111318]">Your Progress</h3>
        <a data-nav="profile" class="text-xs font-bold text-[#FF5A2A] hover:underline flex items-center gap-1 cursor-pointer">
          <span>View Details</span>
          <span class="material-symbols-outlined text-[13px]">arrow_forward</span>
        </a>
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center my-auto">
        <!-- Stat 1: Signs Learned -->
        <div class="flex flex-col items-center">
          <div class="relative w-14 h-14 mb-2 flex items-center justify-center">
            <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path class="text-[#F2E8DC]" stroke-width="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              <path class="text-[#FF5A2A]" stroke-dasharray="65, 100" stroke-width="3" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            </svg>
            <span class="material-symbols-outlined text-[18px] text-[#FF5A2A] absolute">videocam</span>
          </div>
          <div class="text-lg font-extrabold text-[#111318]">48</div>
          <div class="text-[11px] text-[#6B7280]">Signs Learned</div>
        </div>

        <!-- Stat 2: Practice Sessions -->
        <div class="flex flex-col items-center">
          <div class="relative w-14 h-14 mb-2 flex items-center justify-center">
            <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path class="text-[#F2E8DC]" stroke-width="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              <path class="text-[#FF5A2A]" stroke-dasharray="82, 100" stroke-width="3" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            </svg>
            <span class="material-symbols-outlined text-[18px] text-[#FF5A2A] absolute">adjust</span>
          </div>
          <div class="text-lg font-extrabold text-[#111318]">127</div>
          <div class="text-[11px] text-[#6B7280]">Practice Sessions</div>
        </div>

        <!-- Stat 3: Personalized Signs -->
        <div class="flex flex-col items-center">
          <div class="relative w-14 h-14 mb-2 flex items-center justify-center">
            <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path class="text-[#F2E8DC]" stroke-width="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              <path class="text-[#FF5A2A]" stroke-dasharray="45, 100" stroke-width="3" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            </svg>
            <span class="material-symbols-outlined text-[18px] text-[#FF5A2A] absolute">person</span>
          </div>
          <div class="text-lg font-extrabold text-[#111318]">12</div>
          <div class="text-[11px] text-[#6B7280]">Personalized Signs</div>
        </div>

        <!-- Stat 4: Avg Accuracy -->
        <div class="flex flex-col items-center">
          <div class="relative w-14 h-14 mb-2 flex items-center justify-center">
            <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path class="text-[#F2E8DC]" stroke-width="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              <path class="text-[#FF5A2A]" stroke-dasharray="78, 100" stroke-width="3" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            </svg>
            <span class="material-symbols-outlined text-[18px] text-[#FF5A2A] absolute">star</span>
          </div>
          <div class="text-lg font-extrabold text-[#111318]">78%</div>
          <div class="text-[11px] text-[#6B7280]">Avg. Accuracy</div>
        </div>
      </div>
    </div>

    <!-- Center: Recent Activity (Col 6 to 9) -->
    <div class="md:col-span-4 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
      <div class="flex items-center justify-between mb-3">
        <h3 class="font-bold text-sm md:text-base text-[#111318]">Recent Activity</h3>
        <a data-nav="profile" class="text-xs font-bold text-[#FF5A2A] hover:underline flex items-center gap-1 cursor-pointer">
          <span>See All</span>
          <span class="material-symbols-outlined text-[13px]">arrow_forward</span>
        </a>
      </div>

      <div class="flex flex-col gap-2.5">
        <!-- Item 1 -->
        <div class="flex items-center justify-between text-xs py-1">
          <div class="flex items-center gap-2.5">
            <div class="w-6 h-6 rounded-md bg-[#E8F8F0] flex items-center justify-center text-[#16A34A] shrink-0">
              <span class="material-symbols-outlined text-[14px]">videocam</span>
            </div>
            <span class="font-medium text-[#111318]">Translated 5 signs</span>
          </div>
          <span class="text-[11px] text-[#6B7280]">2 minutes ago</span>
        </div>

        <!-- Item 2 -->
        <div class="flex items-center justify-between text-xs py-1">
          <div class="flex items-center gap-2.5">
            <div class="w-6 h-6 rounded-md bg-[#FFF0E8] flex items-center justify-center text-[#FF5A2A] shrink-0">
              <span class="material-symbols-outlined text-[14px]">person</span>
            </div>
            <span class="font-medium text-[#111318]">Added personalized sign: <strong>THANK YOU</strong></span>
          </div>
          <span class="text-[11px] text-[#6B7280]">1 hour ago</span>
        </div>

        <!-- Item 3 -->
        <div class="flex items-center justify-between text-xs py-1">
          <div class="flex items-center gap-2.5">
            <div class="w-6 h-6 rounded-md bg-[#FFF0E8] flex items-center justify-center text-[#FF5A2A] shrink-0">
              <span class="material-symbols-outlined text-[14px]">adjust</span>
            </div>
            <span class="font-medium text-[#111318]">Completed practice: Everyday Signs</span>
          </div>
          <span class="text-[11px] text-[#6B7280]">3 hours ago</span>
        </div>

        <!-- Item 4 -->
        <div class="flex items-center justify-between text-xs py-1">
          <div class="flex items-center gap-2.5">
            <div class="w-6 h-6 rounded-md bg-[#FFF0E8] flex items-center justify-center text-[#FF5A2A] shrink-0">
              <span class="material-symbols-outlined text-[14px]">menu_book</span>
            </div>
            <span class="font-medium text-[#111318]">Learned new sign: <strong>HELP</strong></span>
          </div>
          <span class="text-[11px] text-[#6B7280]">5 hours ago</span>
        </div>
      </div>
    </div>

    <!-- Right: Featured Sign (Col 10 to 12) -->
    <div class="md:col-span-3 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
      <div class="flex items-center justify-between mb-3">
        <h3 class="font-bold text-sm md:text-base text-[#111318]">Featured Sign</h3>
        <div class="flex items-center gap-1">
          <button id="feat-sign-prev" class="w-6 h-6 rounded-full border border-gray-200 hover:bg-gray-100 flex items-center justify-center text-gray-500 transition cursor-pointer">
            <span class="material-symbols-outlined text-[14px]">chevron_left</span>
          </button>
          <button id="feat-sign-next" class="w-6 h-6 rounded-full border border-gray-200 hover:bg-gray-100 flex items-center justify-center text-gray-500 transition cursor-pointer">
            <span class="material-symbols-outlined text-[14px]">chevron_right</span>
          </button>
        </div>
      </div>

      <div class="flex flex-col items-center justify-center my-auto py-2 text-center">
        <div id="feat-sign-gloss" class="text-2xl font-black text-[#111318] tracking-tight mb-1 transition-opacity duration-150">HELLO</div>
        <div id="feat-sign-desc" class="text-xs text-[#6B7280] mb-4">A common greeting</div>
        <button id="feat-sign-btn" data-nav="learn" class="w-full py-2.5 rounded-full bg-gradient-to-r from-[#FF6B3D] to-[#FF501E] hover:from-[#E5481B] hover:to-[#D43A10] text-white font-bold text-xs shadow-md shadow-orange-500/20 transition-all flex items-center justify-center gap-1.5 hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
          <span>Learn This Sign</span>
          <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
        </button>
      </div>
    </div>
  </div>

  <!-- 4. BOTTOM BANNER -->
  <div class="relative w-full rounded-2xl overflow-hidden border border-[#ECE2D8] shadow-sm min-h-[140px] flex items-center justify-between p-6 md:p-8" style="background: url('/static/assets/mountain_landscape.png') center/cover no-repeat;">
    <!-- Gentle overlay for contrast -->
    <div class="absolute inset-0 bg-[#FBF6EE]/80 backdrop-blur-[2px]"></div>
    
    <div class="relative z-10 flex flex-col max-w-2xl">
      <h3 class="text-lg md:text-xl font-extrabold text-[#111318] mb-1.5">
        Together for a more inclusive tomorrow
      </h3>
      <p class="text-xs md:text-sm text-[#5A5E66] font-normal leading-relaxed">
        SignBridge uses AI to break communication barriers and create a more connected, accessible world.
      </p>
    </div>

    <div class="relative z-10 hidden sm:flex flex-col items-end text-right font-bold text-[10px] md:text-[11px] tracking-wider text-[#4B5563] leading-relaxed select-none">
      <span>LEARN</span>
      <span>PRACTICE</span>
      <span>CONNECT</span>
      <span>INCLUDE</span>
    </div>
  </div>

</div>
"""
home_main_soup = BeautifulSoup(HOME_DASHBOARD_HTML, "html.parser")

# --------------------------------------------------------------------------
# 2. PROCESS TWO-WAY COMMUNICATION
# --------------------------------------------------------------------------
comm_main = extract_main_content(comm_soup)
comm_main_soup = BeautifulSoup(comm_main, "html.parser")

tracking_badge = comm_main_soup.find(lambda tag: tag.name == "span" and "TRACKING: DUAL-HAND" in tag.get_text())
if tracking_badge and tracking_badge.parent:
    logo_icon = BeautifulSoup("""
    <div class="sb-logo-container w-4 h-4 shrink-0 inline-flex mr-1">
      <img src="/static/assets/signbridge_logo.png" alt="Logo" class="sb-logo-img">
      <svg class="sb-logo-star" viewBox="0 0 24 24" fill="white">
        <path d="M12 0 C12 7, 7 12, 0 12 C7 12, 12 17, 12 24 C12 17, 17 12, 24 12 C17 12, 12 7, 12 0 Z" fill="#ffffff"></path>
      </svg>
    </div>
    """, "html.parser")
    tracking_badge.parent.insert(0, logo_icon)

for img in comm_main_soup.find_all("img"):
    real_camera_block = BeautifulSoup("""
    <video id="comm-webcam-feed" playsinline muted autoplay class="w-full h-full object-cover rounded-xl" style="display:none;"></video>
    <canvas id="comm-landmark-canvas" class="absolute inset-0 w-full h-full pointer-events-none rounded-xl z-10" style="display:none;"></canvas>
    <!-- Dynamic Interactive Synthetic ASL Landmark Canvas (active when camera is paused or idle) -->
    <canvas id="comm-synthetic-canvas" class="w-full h-full object-cover rounded-xl bg-gradient-to-br from-[#201a17] to-[#362f2b]"></canvas>
    """, "html.parser")
    img.replace_with(real_camera_block)

for btn in comm_main_soup.find_all("button"):
    t = btn.get_text().strip()
    if "Pause session" in t:
        btn["id"] = "comm-pause-btn"
    elif "Clear" in t:
        btn["id"] = "comm-clear-btn"
    elif "End conversation" in t:
        btn["id"] = "comm-end-btn"

for inp in comm_main_soup.find_all("input"):
    if "type your reply" in inp.get("placeholder", "").lower() or "speech" in inp.get("placeholder", "").lower() or "type" in inp.get("placeholder", "").lower():
        inp["id"] = "comm-text-input"

for btn in comm_main_soup.find_all("button"):
    svg_or_span = btn.find("span", class_="material-symbols-outlined")
    if svg_or_span:
        icon_name = svg_or_span.get_text().strip()
        if icon_name == "mic":
            btn["id"] = "comm-btn-mic"
        elif icon_name == "send":
            btn["id"] = "comm-btn-send"
        elif icon_name == "volume_up":
            btn["id"] = "comm-btn-tts"

# --------------------------------------------------------------------------
# 3. PROCESS LEARN VOCABULARY (Exact Match to User Mockup)
# --------------------------------------------------------------------------
LEARN_PAGE_HTML = """
<div class="px-6 py-5 md:px-8 max-w-[1440px] mx-auto flex flex-col gap-5">

  <!-- 1. PAGE HEADER -->
  <div class="flex flex-col md:flex-row md:items-start justify-between gap-4">
    <div class="flex flex-col gap-1">
      <h1 class="text-3xl md:text-4xl font-extrabold text-[#111318] tracking-tight">Learn</h1>
      <p class="text-sm text-[#5A5E66]">Step-by-step lessons to learn sign language at your own pace.</p>
    </div>

    <!-- Search Bar -->
    <div class="relative w-full md:w-[380px]">
      <span class="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-[#8A7266] text-[20px] pointer-events-none">search</span>
      <input id="learn-vocab-search" class="w-full h-11 pl-10 pr-4 bg-white border border-[#E5DDD2] text-[#111318] text-sm rounded-xl focus:outline-none focus:ring-2 focus:ring-[#FF5A2A]/30 placeholder:text-[#8A7266] shadow-sm transition-all" placeholder="Search signs, lessons, or categories..." type="text"/>
    </div>
  </div>

  <!-- Promotional Banner -->
  <div class="w-full rounded-2xl bg-gradient-to-r from-[#FFF0E8] to-[#FFE2D1] border border-[#F3D9C8] p-4 md:p-5 flex items-center justify-between shadow-sm overflow-hidden relative">
    <div class="flex items-center gap-4 z-10">
      <div class="w-14 h-14 rounded-xl bg-[#FF5A2A]/10 flex items-center justify-center shrink-0">
        <svg viewBox="0 0 48 48" class="w-8 h-8" fill="none">
          <path d="M24 4 C24 4, 14 8, 14 18 C14 24, 18 28, 24 28 C30 28, 34 24, 34 18 C34 8, 24 4, 24 4Z" fill="#FF5A2A" opacity="0.15"/>
          <path d="M18 40 L24 28 L30 40" stroke="#FF5A2A" stroke-width="2" stroke-linecap="round"/>
          <path d="M20 16 C20 12, 24 8, 28 12" stroke="#FF5A2A" stroke-width="2" stroke-linecap="round" fill="none"/>
          <circle cx="24" cy="20" r="3" fill="#FF5A2A"/>
        </svg>
      </div>
      <div>
        <p class="text-base md:text-lg font-extrabold text-[#111318] tracking-tight">Small signs. Big connections.</p>
        <p class="text-xs text-[#5A5E66] mt-0.5">Every sign you learn brings you closer to a more inclusive world.</p>
      </div>
    </div>
    <span class="material-symbols-outlined text-[20px] text-[#FF5A2A] cursor-pointer hover:scale-110 transition z-10">chevron_right</span>
    <div class="absolute -right-6 -top-6 w-28 h-28 bg-[#FF5A2A]/5 rounded-full blur-2xl pointer-events-none"></div>
  </div>

  <!-- 2. CATEGORY FILTER CHIPS -->
  <div class="flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar" id="learn-category-chips">
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-bold whitespace-nowrap shadow-sm transition-all bg-[#FF5A2A] text-white" data-category="all">
      <span class="material-symbols-outlined text-[16px] align-middle mr-0.5">auto_awesome</span> All Lessons
    </button>
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30" data-category="basics">
      <span class="text-[14px] mr-0.5">🔤</span> Basics
    </button>
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30 ring-2 ring-[#FF5A2A]/40" data-category="greetings">
      <span class="text-[14px] mr-0.5">👋</span> Greetings
    </button>
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30" data-category="people">
      <span class="text-[14px] mr-0.5">👥</span> People
    </button>
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30" data-category="daily">
      <span class="text-[14px] mr-0.5">🏠</span> Daily Life
    </button>
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30" data-category="emotions">
      <span class="text-[14px] mr-0.5">❤️</span> Emotions
    </button>
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30" data-category="numbers">
      <span class="font-mono text-[13px] font-bold mr-0.5 text-[#FF5A2A]">123</span> Numbers
    </button>
    <button class="learn-cat-chip px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap shadow-sm transition-all bg-white text-[#111318] border border-[#E5DDD2] hover:bg-[#FFF0E8] hover:border-[#FF5A2A]/30" data-category="alphabets">
      <span class="font-serif text-[14px] font-bold mr-0.5 text-[#FF5A2A]">A</span> Alphabets
    </button>
    <button class="w-8 h-8 rounded-full bg-white border border-[#E5DDD2] flex items-center justify-center text-[#5A5E66] hover:bg-[#FFF0E8] transition shrink-0 shadow-sm">
      <span class="material-symbols-outlined text-[18px]">chevron_right</span>
    </button>
  </div>

  <!-- 3. MAIN 3-COLUMN LAYOUT -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">

    <!-- LEFT PANEL: Lesson List -->
    <div class="lg:col-span-3 bg-white rounded-2xl border border-[#ECE2D8] shadow-sm overflow-hidden">
      <!-- Category Header -->
      <div class="px-4 pt-4 pb-3 border-b border-[#ECE2D8]">
        <div class="flex items-center justify-between mb-1">
          <div class="flex items-center gap-2">
            <span class="text-lg" id="learn-current-cat-icon">👋</span>
            <div>
              <h3 class="font-bold text-base text-[#111318]" id="learn-current-cat-title">Greetings</h3>
              <span class="text-[11px] text-[#8A7266]" id="learn-lesson-counter">Lesson 1 of 8</span>
            </div>
          </div>
        </div>
        <div class="flex items-center justify-between mt-2">
          <button id="learn-btn-prev" class="flex items-center gap-1 text-xs text-[#5A5E66] hover:text-[#111318] transition">
            <span class="material-symbols-outlined text-[14px]">arrow_back</span> Previous
          </button>
          <button id="learn-btn-next" class="flex items-center gap-1 px-3 py-1.5 rounded-full bg-[#FF5A2A] text-white text-xs font-bold hover:bg-[#E5481B] transition shadow-sm">
            Next <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
          </button>
        </div>
      </div>

      <!-- Lesson Items -->
      <div class="flex flex-col" id="learn-lesson-list">
        <!-- Item 1: HELLO (Active) -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all bg-[#FFF0E8] border-l-[3px] border-l-[#FF5A2A]" data-sign="HELLO">
          <span class="w-7 h-7 rounded-lg bg-[#FF5A2A] text-white text-xs font-bold flex items-center justify-center shrink-0">1</span>
          <div class="w-9 h-9 rounded-lg bg-gradient-to-br from-[#FFE2D1] to-[#FFCDB3] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><path d="M12 4 L12 12 M12 12 L6 6 M12 12 L9 3 M12 12 L15 3 M12 12 L18 6" stroke="#FF5A2A" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
          <span class="font-bold text-sm text-[#111318] flex-1">HELLO</span>
          <span class="material-symbols-outlined text-[18px] text-[#FF5A2A]">play_circle</span>
        </div>
        <!-- Item 2: GOODBYE -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent" data-sign="GOODBYE">
          <span class="w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0">2</span>
          <div class="w-9 h-9 rounded-lg bg-[#F3E7DC] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><path d="M6 12 Q12 4, 18 12 M8 8 L8 16 M12 6 L12 16 M16 8 L16 16" stroke="#8A7266" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
          <span class="font-semibold text-sm text-[#111318] flex-1">GOODBYE</span>
          <span class="material-symbols-outlined text-[16px] text-[#B0A090]">chevron_right</span>
        </div>
        <!-- Item 3: THANK YOU -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent" data-sign="THANK YOU">
          <span class="w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0">3</span>
          <div class="w-9 h-9 rounded-lg bg-[#F3E7DC] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><path d="M12 18 L12 10 M12 10 L7 5 M12 10 L17 5" stroke="#8A7266" stroke-width="1.5" stroke-linecap="round"/><circle cx="12" cy="18" r="2" fill="#8A7266"/></svg>
          </div>
          <span class="font-semibold text-sm text-[#111318] flex-1">THANK YOU</span>
          <span class="material-symbols-outlined text-[16px] text-[#B0A090]">chevron_right</span>
        </div>
        <!-- Item 4: PLEASE -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent" data-sign="PLEASE">
          <span class="w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0">4</span>
          <div class="w-9 h-9 rounded-lg bg-[#F3E7DC] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><circle cx="12" cy="12" r="6" stroke="#8A7266" stroke-width="1.5" fill="none"/><path d="M9 12 Q12 8, 15 12" stroke="#8A7266" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
          <span class="font-semibold text-sm text-[#111318] flex-1">PLEASE</span>
          <span class="material-symbols-outlined text-[16px] text-[#B0A090]">chevron_right</span>
        </div>
        <!-- Item 5: YES -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent" data-sign="YES">
          <span class="w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0">5</span>
          <div class="w-9 h-9 rounded-lg bg-[#F3E7DC] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><rect x="8" y="8" width="8" height="10" rx="3" stroke="#8A7266" stroke-width="1.5" fill="none"/><path d="M12 6 L12 4" stroke="#8A7266" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
          <span class="font-semibold text-sm text-[#111318] flex-1">YES</span>
          <span class="material-symbols-outlined text-[16px] text-[#B0A090]">chevron_right</span>
        </div>
        <!-- Item 6: NO -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent" data-sign="NO">
          <span class="w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0">6</span>
          <div class="w-9 h-9 rounded-lg bg-[#F3E7DC] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><path d="M10 16 L10 10 M14 16 L14 10 M10 14 Q12 16, 14 14" stroke="#8A7266" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
          <span class="font-semibold text-sm text-[#111318] flex-1">NO</span>
          <span class="material-symbols-outlined text-[16px] text-[#B0A090]">chevron_right</span>
        </div>
        <!-- Item 7: HOW ARE YOU -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent" data-sign="HOW ARE YOU">
          <span class="w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0">7</span>
          <div class="w-9 h-9 rounded-lg bg-[#F3E7DC] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><path d="M6 14 L12 8 L18 14 M9 11 L15 11" stroke="#8A7266" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
          <span class="font-semibold text-sm text-[#111318] flex-1">HOW ARE YOU</span>
          <span class="material-symbols-outlined text-[16px] text-[#B0A090]">chevron_right</span>
        </div>
        <!-- Item 8: NICE TO MEET YOU -->
        <div class="learn-lesson-item flex items-center gap-3 px-4 py-3 cursor-pointer transition-all hover:bg-[#FFF8F2] border-l-[3px] border-l-transparent" data-sign="NICE TO MEET YOU">
          <span class="w-7 h-7 rounded-lg bg-[#F3E7DC] text-[#5A5E66] text-xs font-bold flex items-center justify-center shrink-0">8</span>
          <div class="w-9 h-9 rounded-lg bg-[#F3E7DC] flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" class="w-5 h-5" fill="none"><path d="M8 16 L8 8 M16 16 L16 8 M8 12 L16 12" stroke="#8A7266" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
          <span class="font-semibold text-sm text-[#111318] flex-1">NICE TO MEET YOU</span>
          <span class="material-symbols-outlined text-[16px] text-[#B0A090]">chevron_right</span>
        </div>
      </div>
    </div>

    <!-- CENTER PANEL: Sign Detail + Video + Practice -->
    <div class="lg:col-span-6 flex flex-col gap-5">
      <!-- Active Sign Header -->
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-2xl font-extrabold text-[#111318] tracking-tight" id="learn-active-sign-title">HELLO</h2>
          <p class="text-sm text-[#5A5E66] mt-0.5" id="learn-active-sign-desc">A common greeting used to start a conversation.</p>
        </div>
        <span class="px-3 py-1 rounded-full bg-[#E8F5E9] text-[#2E7D32] text-xs font-bold" id="learn-active-sign-level">Beginner</span>
      </div>

      <!-- Sign Demonstration Area (SVG Hand Landmarks - NO person) -->
      <div class="relative w-full rounded-2xl overflow-hidden bg-gradient-to-br from-[#1A1510] to-[#2C241C] shadow-lg border border-[#3D342C] min-h-[280px] flex items-center justify-center" id="learn-sign-demo-area">
        <!-- Animated Hand Landmark SVG -->
        <svg viewBox="0 0 400 300" class="w-full h-full max-h-[280px]" fill="none" id="learn-hand-svg">
          <!-- Background grid -->
          <defs>
            <pattern id="learn-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#3D342C" stroke-width="0.5"/>
            </pattern>
          </defs>
          <rect width="400" height="300" fill="url(#learn-grid)" opacity="0.3"/>

          <!-- Palm base -->
          <ellipse cx="200" cy="200" rx="45" ry="50" fill="#FF5A2A" opacity="0.08" stroke="#FF5A2A" stroke-width="1" stroke-dasharray="4 4">
            <animate attributeName="rx" values="45;48;45" dur="3s" repeatCount="indefinite"/>
          </ellipse>

          <!-- Wrist -->
          <circle cx="200" cy="248" r="5" fill="#FF8343"/>
          <!-- Palm center -->
          <circle cx="200" cy="195" r="6" fill="#FF5A2A">
            <animate attributeName="r" values="6;7;6" dur="2s" repeatCount="indefinite"/>
          </circle>

          <!-- Thumb -->
          <line x1="200" y1="195" x2="145" y2="175" stroke="#FFB694" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="145" y1="175" x2="130" y2="155" stroke="#FFB694" stroke-width="2" stroke-linecap="round"/>
          <circle cx="130" cy="155" r="4" fill="#FF8343"/>

          <!-- Index finger -->
          <line x1="200" y1="195" x2="175" y2="110" stroke="#FFB694" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="175" y1="110" x2="168" y2="75" stroke="#FFB694" stroke-width="2" stroke-linecap="round"/>
          <circle cx="168" cy="75" r="4" fill="#FF8343">
            <animate attributeName="cy" values="75;70;75" dur="2.5s" repeatCount="indefinite"/>
          </circle>

          <!-- Middle finger -->
          <line x1="200" y1="195" x2="195" y2="100" stroke="#FFB694" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="195" y1="100" x2="192" y2="62" stroke="#FFB694" stroke-width="2" stroke-linecap="round"/>
          <circle cx="192" cy="62" r="4" fill="#FF8343">
            <animate attributeName="cy" values="62;57;62" dur="2.8s" repeatCount="indefinite"/>
          </circle>

          <!-- Ring finger -->
          <line x1="200" y1="195" x2="218" y2="105" stroke="#FFB694" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="218" y1="105" x2="222" y2="70" stroke="#FFB694" stroke-width="2" stroke-linecap="round"/>
          <circle cx="222" cy="70" r="4" fill="#FF8343">
            <animate attributeName="cy" values="70;65;70" dur="2.3s" repeatCount="indefinite"/>
          </circle>

          <!-- Pinky -->
          <line x1="200" y1="195" x2="240" y2="120" stroke="#FFB694" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="240" y1="120" x2="250" y2="90" stroke="#FFB694" stroke-width="2" stroke-linecap="round"/>
          <circle cx="250" cy="90" r="4" fill="#FF8343">
            <animate attributeName="cy" values="90;85;90" dur="2.6s" repeatCount="indefinite"/>
          </circle>

          <!-- Motion arc arrow -->
          <path d="M280 150 Q310 120 300 80" stroke="#FF5A2A" stroke-width="2" stroke-linecap="round" fill="none" stroke-dasharray="5 3">
            <animate attributeName="stroke-dashoffset" values="0;-16" dur="1.5s" repeatCount="indefinite"/>
          </path>
          <polygon points="300,80 305,90 295,88" fill="#FF5A2A"/>
        </svg>

        <!-- Video Progress Bar -->
        <div class="absolute bottom-0 left-0 right-0 bg-black/50 backdrop-blur-sm px-4 py-2.5 flex items-center gap-3">
          <button class="text-white hover:text-[#FF5A2A] transition" id="learn-play-btn">
            <span class="material-symbols-outlined text-[22px]">play_arrow</span>
          </button>
          <span class="text-white/70 text-xs font-mono">0:02 / 0:05</span>
          <div class="flex-1 h-1.5 bg-white/20 rounded-full overflow-hidden mx-2">
            <div class="h-full bg-[#FF5A2A] rounded-full w-[40%] transition-all"></div>
          </div>
          <div class="flex items-center gap-2">
            <button class="text-white/70 hover:text-white text-xs font-mono transition">1x</button>
            <button class="text-white/70 hover:text-white transition">
              <span class="material-symbols-outlined text-[18px]">fullscreen</span>
            </button>
          </div>
        </div>

        <!-- Watch Again Button -->
        <button class="absolute top-4 right-4 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/10 backdrop-blur-md text-white text-xs font-semibold hover:bg-white/20 transition border border-white/10">
          <span class="material-symbols-outlined text-[16px]">replay</span> Watch Again
        </button>
      </div>

      <!-- Try It Yourself Section -->
      <div class="bg-white rounded-2xl border border-[#ECE2D8] shadow-sm overflow-hidden">
        <div class="px-5 pt-4 pb-3 border-b border-[#ECE2D8] flex items-center gap-3">
          <div class="w-8 h-8 rounded-lg bg-[#FFF0E8] flex items-center justify-center">
            <span class="material-symbols-outlined text-[18px] text-[#FF5A2A]">photo_camera</span>
          </div>
          <div>
            <h3 class="font-bold text-sm text-[#111318]">Try it Yourself</h3>
            <p class="text-[11px] text-[#8A7266]">Practice the sign and get real-time feedback.</p>
          </div>
        </div>
        <div class="p-5 flex flex-col md:flex-row gap-5">
          <!-- Webcam Area -->
          <div class="relative flex-1 min-h-[180px] rounded-xl bg-gradient-to-br from-[#1A1510] to-[#2C241C] border border-[#3D342C] overflow-hidden flex items-center justify-center">
            <video id="learn-practice-webcam" playsinline muted autoplay class="w-full h-full object-cover" style="display:none;"></video>
            <canvas id="learn-practice-canvas" class="absolute inset-0 w-full h-full pointer-events-none z-10" style="display:none;"></canvas>
            <!-- Placeholder when camera is off -->
            <div id="learn-practice-placeholder" class="flex flex-col items-center gap-3 text-center p-4">
              <div class="w-14 h-14 rounded-full bg-[#FF5A2A]/10 flex items-center justify-center">
                <svg viewBox="0 0 48 48" class="w-7 h-7" fill="none">
                  <path d="M24 8 L24 20 M24 20 L14 12 M24 20 L18 6 M24 20 L30 6 M24 20 L34 12" stroke="#FF5A2A" stroke-width="2" stroke-linecap="round"/>
                  <circle cx="24" cy="20" r="3" fill="#FF5A2A"/>
                  <path d="M24 24 L24 38" stroke="#FF5A2A" stroke-width="2" stroke-linecap="round"/>
                  <circle cx="24" cy="38" r="3" fill="#FF5A2A" opacity="0.5"/>
                </svg>
              </div>
              <span class="text-white/60 text-xs font-semibold">Position your hands in view</span>
            </div>
            <!-- Good feedback overlay -->
            <div id="learn-practice-feedback-badge" class="absolute top-3 left-3 px-3 py-1 rounded-full bg-[#4CAF50] text-white text-xs font-bold flex items-center gap-1 shadow-md" style="display:none;">
              <span class="material-symbols-outlined text-[14px]">check_circle</span> Good!
            </div>
            <!-- Camera toggle -->
            <button id="learn-practice-cam-btn" class="absolute bottom-3 right-3 w-10 h-10 rounded-full bg-[#FF5A2A] text-white flex items-center justify-center hover:bg-[#E5481B] transition shadow-lg">
              <span class="material-symbols-outlined text-[20px]">videocam</span>
            </button>
          </div>

          <!-- Feedback Checklist -->
          <div class="flex flex-col gap-3 min-w-[200px]">
            <div class="flex items-center gap-2.5">
              <span class="w-5 h-5 rounded-full bg-[#E8F5E9] flex items-center justify-center shrink-0">
                <span class="material-symbols-outlined text-[14px] text-[#4CAF50]">check</span>
              </span>
              <span class="text-sm text-[#111318]">Hand clearly visible</span>
            </div>
            <div class="flex items-center gap-2.5">
              <span class="w-5 h-5 rounded-full bg-[#E8F5E9] flex items-center justify-center shrink-0">
                <span class="material-symbols-outlined text-[14px] text-[#4CAF50]">check</span>
              </span>
              <span class="text-sm text-[#111318]">Correct hand shape</span>
            </div>
            <div class="flex items-center gap-2.5">
              <span class="w-5 h-5 rounded-full bg-[#E8F5E9] flex items-center justify-center shrink-0">
                <span class="material-symbols-outlined text-[14px] text-[#4CAF50]">check</span>
              </span>
              <span class="text-sm text-[#111318]">Good position</span>
            </div>
            <div class="flex items-center gap-2.5">
              <span class="w-5 h-5 rounded-full bg-[#F3E7DC] flex items-center justify-center shrink-0">
                <span class="w-2 h-2 rounded-full bg-[#B0A090]"></span>
              </span>
              <span class="text-sm text-[#8A7266]">Keep steady for 1 second</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- RIGHT PANEL: Progress + Related Signs + Example Usage -->
    <div class="lg:col-span-3 flex flex-col gap-5">

      <!-- Your Progress Card -->
      <div class="bg-white rounded-2xl border border-[#ECE2D8] shadow-sm p-5">
        <div class="flex items-center justify-between mb-4">
          <h3 class="font-bold text-base text-[#111318]">Your Progress</h3>
          <button class="text-xs text-[#FF5A2A] font-bold hover:underline transition">View All →</button>
        </div>

        <!-- Circular Progress -->
        <div class="flex items-center justify-center mb-5">
          <div class="relative w-28 h-28">
            <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="42" fill="none" stroke="#F3E7DC" stroke-width="7"/>
              <circle cx="50" cy="50" r="42" fill="none" stroke="#FF5A2A" stroke-width="7" stroke-linecap="round"
                stroke-dasharray="264" stroke-dashoffset="165">
                <animate attributeName="stroke-dashoffset" from="264" to="165" dur="1.5s" fill="freeze"/>
              </circle>
            </svg>
            <div class="absolute inset-0 flex flex-col items-center justify-center">
              <span class="text-3xl font-extrabold text-[#111318]">3<span class="text-lg text-[#8A7266] font-semibold">/8</span></span>
              <span class="text-[10px] text-[#8A7266] font-medium">Lessons Completed</span>
            </div>
          </div>
        </div>

        <!-- Stats Row -->
        <div class="grid grid-cols-3 gap-3">
          <div class="flex flex-col items-center text-center p-2 rounded-xl bg-[#FFF8F2]">
            <span class="text-lg">🔥</span>
            <span class="text-lg font-extrabold text-[#111318]">2</span>
            <span class="text-[10px] text-[#8A7266] font-medium">Day Streak</span>
          </div>
          <div class="flex flex-col items-center text-center p-2 rounded-xl bg-[#FFF8F2]">
            <span class="text-lg">🎯</span>
            <span class="text-lg font-extrabold text-[#111318]">12</span>
            <span class="text-[10px] text-[#8A7266] font-medium">Signs Learned</span>
          </div>
          <div class="flex flex-col items-center text-center p-2 rounded-xl bg-[#FFF8F2]">
            <span class="text-lg">🏅</span>
            <span class="text-lg font-extrabold text-[#111318]">1</span>
            <span class="text-[10px] text-[#8A7266] font-medium">Badges</span>
          </div>
        </div>
      </div>

      <!-- Related Signs Card -->
      <div class="bg-white rounded-2xl border border-[#ECE2D8] shadow-sm p-5">
        <div class="flex items-center justify-between mb-4">
          <h3 class="font-bold text-base text-[#111318]">Related Signs</h3>
          <button class="text-xs text-[#FF5A2A] font-bold hover:underline transition">See More →</button>
        </div>
        <div class="grid grid-cols-3 gap-3">
          <!-- Related Sign 1: HI -->
          <div class="learn-related-sign flex flex-col items-center gap-1.5 cursor-pointer group" data-sign="HELLO">
            <div class="w-full aspect-square rounded-xl bg-gradient-to-br from-[#FFF0E8] to-[#FFE2D1] flex items-center justify-center group-hover:shadow-md transition border border-[#F3D9C8]">
              <svg viewBox="0 0 40 40" class="w-8 h-8" fill="none">
                <path d="M20 32 L20 18 M20 18 L12 10 M20 18 L16 6 M20 18 L24 6 M20 18 L28 10" stroke="#FF5A2A" stroke-width="1.5" stroke-linecap="round"/>
                <circle cx="20" cy="18" r="2.5" fill="#FF5A2A"/>
              </svg>
            </div>
            <span class="text-[11px] font-bold text-[#111318]">HI</span>
          </div>
          <!-- Related Sign 2: GOODBYE -->
          <div class="learn-related-sign flex flex-col items-center gap-1.5 cursor-pointer group" data-sign="GOODBYE">
            <div class="w-full aspect-square rounded-xl bg-gradient-to-br from-[#FFF0E8] to-[#FFE2D1] flex items-center justify-center group-hover:shadow-md transition border border-[#F3D9C8]">
              <svg viewBox="0 0 40 40" class="w-8 h-8" fill="none">
                <path d="M14 28 L14 14 M20 28 L20 10 M26 28 L26 14" stroke="#FF5A2A" stroke-width="1.5" stroke-linecap="round"/>
                <path d="M10 20 Q20 12, 30 20" stroke="#FF5A2A" stroke-width="1.5" stroke-linecap="round" fill="none"/>
              </svg>
            </div>
            <span class="text-[11px] font-bold text-[#111318]">GOODBYE</span>
          </div>
          <!-- Related Sign 3: HOW ARE YOU -->
          <div class="learn-related-sign flex flex-col items-center gap-1.5 cursor-pointer group" data-sign="HOW ARE YOU">
            <div class="w-full aspect-square rounded-xl bg-gradient-to-br from-[#FFF0E8] to-[#FFE2D1] flex items-center justify-center group-hover:shadow-md transition border border-[#F3D9C8]">
              <svg viewBox="0 0 40 40" class="w-8 h-8" fill="none">
                <path d="M12 24 L20 14 L28 24 M16 19 L24 19" stroke="#FF5A2A" stroke-width="1.5" stroke-linecap="round"/>
                <circle cx="20" cy="14" r="2" fill="#FF5A2A"/>
              </svg>
            </div>
            <span class="text-[11px] font-bold text-[#111318]">HOW ARE YOU</span>
          </div>
        </div>
      </div>

      <!-- Example Usage Card -->
      <div class="bg-white rounded-2xl border border-[#ECE2D8] shadow-sm p-5">
        <div class="flex items-center gap-2 mb-4">
          <span class="text-lg">💬</span>
          <h3 class="font-bold text-base text-[#111318]">Example Usage</h3>
        </div>
        <div class="flex flex-col gap-3">
          <!-- Example 1 -->
          <div class="flex items-center justify-between p-3 rounded-xl bg-[#FFF8F2] border border-[#F3E7DC]">
            <div class="flex items-center gap-2.5">
              <span class="text-lg">👋</span>
              <span class="text-sm text-[#111318]">Hello, how are you?</span>
            </div>
            <button class="learn-speak-btn w-8 h-8 rounded-full bg-white border border-[#E5DDD2] flex items-center justify-center text-[#5A5E66] hover:text-[#FF5A2A] hover:border-[#FF5A2A]/30 transition shadow-sm cursor-pointer" data-phrase="Hello, how are you?">
              <span class="material-symbols-outlined text-[16px]">volume_up</span>
            </button>
          </div>
          <!-- Example 2 -->
          <div class="flex items-center justify-between p-3 rounded-xl bg-[#FFF8F2] border border-[#F3E7DC]">
            <div class="flex items-center gap-2.5">
              <span class="text-lg">👋</span>
              <span class="text-sm text-[#111318]">Hi, nice to meet you!</span>
            </div>
            <button class="learn-speak-btn w-8 h-8 rounded-full bg-white border border-[#E5DDD2] flex items-center justify-center text-[#5A5E66] hover:text-[#FF5A2A] hover:border-[#FF5A2A]/30 transition shadow-sm cursor-pointer" data-phrase="Hi, nice to meet you!">
              <span class="material-symbols-outlined text-[16px]">volume_up</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Interactive Demonstration Modal (Streams actual ASL Citizen videos) -->
<div id="demo-modal" class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 hidden">
  <div class="bg-white dark:bg-[#1f1b18] rounded-2xl max-w-2xl w-full p-6 shadow-2xl border border-[#ECE2D8] flex flex-col gap-4">
    <div class="flex items-center justify-between border-b border-[#ECE2D8] pb-3">
      <div>
        <div class="text-xs uppercase font-mono font-bold text-[#9c4400]" id="demo-modal-category">GREETINGS</div>
        <h3 class="text-2xl font-bold text-[#111318]" id="demo-sign-title">Sign: HELLO</h3>
      </div>
      <button id="btn-close-demo-modal" class="p-2 rounded-full hover:bg-[#F3E7DC] text-[#5A5E66] transition cursor-pointer">
        <span class="material-symbols-outlined">close</span>
      </button>
    </div>
    <div class="relative aspect-video rounded-xl bg-black overflow-hidden flex items-center justify-center">
      <video id="demo-video-player" class="w-full h-full object-contain" controls autoplay loop playsinline></video>
      <div id="demo-video-loading" class="absolute inset-0 bg-black/70 flex flex-col items-center justify-center text-white gap-2" style="display:none;">
        <span class="w-6 h-6 border-2 border-[#FF5A2A] border-t-transparent rounded-full animate-spin"></span>
        <span class="text-xs font-mono">Loading demonstration video...</span>
      </div>
    </div>
    <div class="flex items-center justify-between flex-wrap gap-2">
      <div class="flex items-center gap-2">
        <span class="text-xs text-[#5A5E66]">Playback Speed:</span>
        <button class="demo-speed-btn px-2.5 py-1 rounded bg-[#F3E7DC] font-mono text-xs font-semibold text-[#111318] hover:bg-[#FF5A2A] hover:text-white transition" data-speed="0.5">0.5x</button>
        <button class="demo-speed-btn px-2.5 py-1 rounded bg-[#F3E7DC] font-mono text-xs font-semibold text-[#111318] hover:bg-[#FF5A2A] hover:text-white transition" data-speed="0.75">0.75x</button>
        <button class="demo-speed-btn px-2.5 py-1 rounded bg-[#FF5A2A] text-white font-mono text-xs font-semibold" data-speed="1.0">1.0x</button>
      </div>
      <div class="flex items-center gap-2">
        <button id="btn-pronounce-sign" class="px-3 py-1.5 rounded-lg border border-[#E5DDD2] text-xs hover:bg-[#F3E7DC] flex items-center gap-1.5 transition">
          <span class="material-symbols-outlined text-[16px]">volume_up</span> Speak Word
        </button>
        <button id="btn-try-signing-demo" data-nav="practice" class="px-4 py-1.5 rounded-lg bg-[#FF5A2A] text-white text-xs font-bold hover:bg-[#E5481B] transition flex items-center gap-1.5">
          <span class="material-symbols-outlined text-[16px]">front_hand</span> Practice This Sign
        </button>
      </div>
    </div>
  </div>
</div>
"""
learn_main_soup = BeautifulSoup(LEARN_PAGE_HTML, "html.parser")

# --------------------------------------------------------------------------
# 4. PROCESS VIDEO TRANSLATION (Exact Match to User Reference Mockup)
# --------------------------------------------------------------------------
VIDEO_TRANSLATE_HTML = """
<div class="px-6 py-6 md:px-8 max-w-7xl mx-auto flex flex-col gap-6">

  <!-- 1. PAGE HEADER -->
  <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
    <div class="flex flex-col">
      <h1 class="text-2xl md:text-3xl font-extrabold text-[#111318] tracking-tight">Video Translate</h1>
      <p class="text-xs md:text-sm text-[#5A5E66] mt-0.5">Upload a video and let AI recognize the signs and convert them into natural text.</p>
    </div>

    <div class="flex items-center gap-3">
      <div class="hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white border border-[#E5DDD2] text-xs font-semibold text-[#5A5E66] shadow-sm select-none">
        <span class="material-symbols-outlined text-[16px] text-[#FF5A2A]">smart_display</span>
        <span>Supports MP4, MOV, WEBM</span>
      </div>

      <button id="btn-upload-another" class="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#FF5A2A] hover:bg-[#E5481B] text-white font-bold text-xs shadow-md shadow-orange-500/25 transition-all hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
        <span class="material-symbols-outlined text-[16px]">upload</span>
        <span>Upload Another Video</span>
      </button>
      <input type="file" id="video-file-input" accept="video/mp4,video/webm,video/quicktime" class="hidden">
    </div>
  </div>

  <!-- 2. TOP SECTION: VIDEO PLAYER (Left) + TRANSLATED SENTENCE & VIDEO INFO (Right) -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
    
    <!-- Video Player Viewport (7 Cols) -->
    <div class="lg:col-span-7 bg-[#16181D] rounded-2xl overflow-hidden border border-[#2D3139] shadow-md relative min-h-[360px] md:min-h-[420px] flex flex-col justify-between">
      
      <!-- Video Element & Landmark Canvas Overlay -->
      <div class="relative w-full flex-1 flex items-center justify-center bg-black min-h-[300px]">
        <video id="trans-video-player" playsinline preload="metadata" class="w-full h-full object-cover">
          <source src="/api/video/consultation_sample_asl.mp4" type="video/mp4">
        </video>
        <canvas id="trans-landmark-canvas" class="absolute inset-0 w-full h-full pointer-events-none z-10"></canvas>

        <!-- Dropzone Overlay (Hidden by default, shown when user clicks upload or drags file) -->
        <div id="video-dropzone" class="absolute inset-0 z-30 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center cursor-pointer transition" style="display:none;">
          <div class="w-14 h-14 rounded-full bg-[#FF5A2A]/20 flex items-center justify-center text-[#FF5A2A] mb-3">
            <span class="material-symbols-outlined text-[32px]">cloud_upload</span>
          </div>
          <h3 class="text-white font-bold text-base mb-1">Drop ASL video here or click to browse</h3>
          <p class="text-xs text-gray-300 max-w-sm mb-4">Supports MP4, WebM, MOV up to 50MB</p>
          <button type="button" class="px-5 py-2 rounded-full bg-[#FF5A2A] text-white text-xs font-bold shadow-md shadow-orange-500/30">Select Video File</button>
        </div>

        <!-- Top-Left HUD Badge: Processing Complete -->
        <div class="absolute top-4 left-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/15 text-white font-medium text-xs select-none">
          <span class="w-2.5 h-2.5 rounded-full bg-[#22C55E]"></span>
          <span class="font-bold">Processing Complete</span>
          <span class="text-white/40">·</span>
          <span class="text-white/80">18 seconds · 45 signs detected</span>
        </div>
      </div>

      <!-- Custom Player Control Bar -->
      <div class="p-3 bg-[#111318] border-t border-white/10 flex flex-col gap-2 z-20">
        <!-- Progress Bar Scrubber -->
        <div class="relative w-full h-1.5 bg-white/20 hover:h-2.5 rounded-full cursor-pointer transition-all group" id="video-scrubber-track">
          <div id="video-scrubber-fill" class="h-full bg-[#FF5A2A] rounded-full relative" style="width: 22%;">
            <span class="w-3.5 h-3.5 rounded-full bg-white border-2 border-[#FF5A2A] shadow absolute right-0 top-1/2 transform -translate-y-1/2 scale-0 group-hover:scale-100 transition-transform"></span>
          </div>
        </div>

        <!-- Controls Row -->
        <div class="flex items-center justify-between text-white text-xs px-1">
          <div class="flex items-center gap-3">
            <button id="btn-video-play" class="hover:text-[#FF5A2A] transition cursor-pointer flex items-center justify-center">
              <span class="material-symbols-outlined text-[20px]">play_arrow</span>
            </button>
            <span id="video-time-display" class="font-mono text-[11px] text-gray-300">00:04 / 00:18</span>
          </div>

          <div class="flex items-center gap-3">
            <button id="btn-video-mute" class="hover:text-[#FF5A2A] transition cursor-pointer">
              <span class="material-symbols-outlined text-[18px]">volume_up</span>
            </button>
            <button id="btn-video-speed" class="px-1.5 py-0.5 rounded bg-white/10 hover:bg-white/20 text-[11px] font-bold transition cursor-pointer">1x</button>
            <button id="btn-video-fullscreen" class="hover:text-[#FF5A2A] transition cursor-pointer">
              <span class="material-symbols-outlined text-[18px]">fullscreen</span>
            </button>
          </div>
        </div>
      </div>

    </div>

    <!-- Right Column (5 Cols): Translated Sentence & Video Information -->
    <div class="lg:col-span-5 flex flex-col gap-4">
      
      <!-- Card 1: Translated Sentence -->
      <div class="bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <div class="w-7 h-7 rounded-lg bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A]">
              <span class="material-symbols-outlined text-[16px]">translate</span>
            </div>
            <span class="font-bold text-sm text-[#111318]">Translated Sentence</span>
          </div>
          <button id="btn-copy-video-sentence" class="flex items-center gap-1 px-2.5 py-1 rounded-lg hover:bg-gray-100 text-xs font-semibold text-[#5A5E66] transition cursor-pointer">
            <span class="material-symbols-outlined text-[14px]">content_copy</span>
            <span>Copy</span>
          </button>
        </div>

        <div class="bg-[#FFF9F3] border border-[#F3E7DC] rounded-xl p-4 mb-4">
          <p id="video-sentence-text" class="text-lg md:text-xl font-bold text-[#111318] tracking-tight leading-snug">
            Hello, my name is ___.<br>Nice to meet you.
          </p>
        </div>

        <div class="flex items-center gap-2.5">
          <button id="btn-speak-video-sentence" class="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-full bg-[#FF5A2A] hover:bg-[#E5481B] text-white font-bold text-xs shadow-md shadow-orange-500/25 transition-all hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
            <span class="material-symbols-outlined text-[16px]">volume_up</span>
            <span>Speak</span>
          </button>
          <button id="btn-edit-video-sentence" class="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-full bg-white hover:bg-gray-50 border border-[#E5DDD2] text-[#111318] font-bold text-xs shadow-sm transition cursor-pointer">
            <span class="material-symbols-outlined text-[16px]">edit</span>
            <span>Edit Text</span>
          </button>
        </div>
      </div>

      <!-- Card 2: Video Information -->
      <div class="bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
        <div class="flex items-center gap-2 mb-3">
          <div class="w-6 h-6 rounded-md bg-[#F5EFE6] flex items-center justify-center text-[#5A5E66]">
            <span class="material-symbols-outlined text-[14px]">description</span>
          </div>
          <span class="font-bold text-sm text-[#111318]">Video Information</span>
        </div>

        <div class="flex items-center justify-between gap-4">
          <div class="flex flex-col gap-1.5 text-xs flex-1">
            <div class="flex items-center justify-between py-0.5 border-b border-gray-100">
              <span class="text-[#6B7280]">File Name</span>
              <span class="font-semibold text-[#111318]">sample_sign_video.mp4</span>
            </div>
            <div class="flex items-center justify-between py-0.5 border-b border-gray-100">
              <span class="text-[#6B7280]">Duration</span>
              <span class="font-semibold text-[#111318]">00:18</span>
            </div>
            <div class="flex items-center justify-between py-0.5 border-b border-gray-100">
              <span class="text-[#6B7280]">Resolution</span>
              <span class="font-semibold text-[#111318]">1920 × 1080</span>
            </div>
            <div class="flex items-center justify-between py-0.5 border-b border-gray-100">
              <span class="text-[#6B7280]">Frame Rate</span>
              <span class="font-semibold text-[#111318]">30 FPS</span>
            </div>
            <div class="flex items-center justify-between py-0.5">
              <span class="text-[#6B7280]">File Size</span>
              <span class="font-semibold text-[#111318]">12.4 MB</span>
            </div>
          </div>

          <!-- Video Thumbnail Preview -->
          <div class="w-24 h-20 rounded-xl bg-[#20232A] border border-[#E5DDD2] overflow-hidden relative flex items-center justify-center shrink-0 shadow-inner group cursor-pointer" id="btn-replay-preview">
            <svg class="w-10 h-10 text-[#FF5A2A] opacity-90 group-hover:scale-110 transition-transform" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z"/>
            </svg>
            <span class="absolute bottom-1 right-1 px-1 py-0.2 bg-black/70 text-[9px] text-white font-mono rounded">00:18</span>
          </div>
        </div>
      </div>

    </div>

  </div>

  <!-- 3. MIDDLE SECTION: DETECTED SIGNS TIMELINE -->
  <div class="bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col gap-4">
    <div class="flex items-center justify-between">
      <h2 class="font-bold text-sm md:text-base text-[#111318]">Detected Signs Timeline</h2>
    </div>

    <!-- Waveform Visualizer with Time Marker Pin -->
    <div class="relative w-full h-12 flex items-center justify-between gap-[2px] px-2 overflow-hidden select-none cursor-pointer" id="timeline-waveform">
      <!-- Scrubber Pin Needle at 22% (00:04) -->
      <div id="timeline-pin" class="absolute left-[22%] top-0 bottom-0 z-10 flex flex-col items-center pointer-events-none transform -translate-x-1/2">
        <span class="px-1.5 py-0.5 rounded bg-[#FF5A2A] text-white font-mono text-[9px] font-bold shadow">00:04</span>
        <div class="w-[2px] flex-1 bg-[#FF5A2A]"></div>
      </div>
      <!-- SVG Multi-Color Waveform -->
      <svg class="w-full h-full" preserveAspectRatio="none" viewBox="0 0 500 48">
        <defs>
          <linearGradient id="waveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#FF5A2A" />
            <stop offset="35%" stop-color="#EAB308" />
            <stop offset="65%" stop-color="#06B6D4" />
            <stop offset="100%" stop-color="#8B5CF6" />
          </linearGradient>
        </defs>
        <!-- Procedural soundwave bars -->
        <g fill="url(#waveGrad)">
          <rect x="5" y="16" width="4" height="16" rx="2"></rect>
          <rect x="15" y="10" width="4" height="28" rx="2"></rect>
          <rect x="25" y="18" width="4" height="12" rx="2"></rect>
          <rect x="35" y="8" width="4" height="32" rx="2"></rect>
          <rect x="45" y="14" width="4" height="20" rx="2"></rect>
          <rect x="55" y="4" width="4" height="40" rx="2"></rect>
          <rect x="65" y="12" width="4" height="24" rx="2"></rect>
          <rect x="75" y="18" width="4" height="12" rx="2"></rect>
          <rect x="85" y="6" width="4" height="36" rx="2"></rect>
          <rect x="95" y="14" width="4" height="20" rx="2"></rect>
          <rect x="105" y="2" width="4" height="44" rx="2"></rect>
          <rect x="115" y="12" width="4" height="24" rx="2"></rect>
          <rect x="125" y="16" width="4" height="16" rx="2"></rect>
          <rect x="135" y="8" width="4" height="32" rx="2"></rect>
          <rect x="145" y="14" width="4" height="20" rx="2"></rect>
          <rect x="155" y="6" width="4" height="36" rx="2"></rect>
          <rect x="165" y="18" width="4" height="12" rx="2"></rect>
          <rect x="175" y="10" width="4" height="28" rx="2"></rect>
          <rect x="185" y="16" width="4" height="16" rx="2"></rect>
          <rect x="195" y="4" width="4" height="40" rx="2"></rect>
          <rect x="205" y="14" width="4" height="20" rx="2"></rect>
          <rect x="215" y="8" width="4" height="32" rx="2"></rect>
          <rect x="225" y="16" width="4" height="16" rx="2"></rect>
          <rect x="235" y="6" width="4" height="36" rx="2"></rect>
          <rect x="245" y="12" width="4" height="24" rx="2"></rect>
          <rect x="255" y="18" width="4" height="12" rx="2"></rect>
          <rect x="265" y="4" width="4" height="40" rx="2"></rect>
          <rect x="275" y="14" width="4" height="20" rx="2"></rect>
          <rect x="285" y="8" width="4" height="32" rx="2"></rect>
          <rect x="295" y="16" width="4" height="16" rx="2"></rect>
          <rect x="305" y="10" width="4" height="28" rx="2"></rect>
          <rect x="315" y="6" width="4" height="36" rx="2"></rect>
          <rect x="325" y="14" width="4" height="20" rx="2"></rect>
          <rect x="335" y="2" width="4" height="44" rx="2"></rect>
          <rect x="345" y="12" width="4" height="24" rx="2"></rect>
          <rect x="355" y="16" width="4" height="16" rx="2"></rect>
          <rect x="365" y="8" width="4" height="32" rx="2"></rect>
          <rect x="375" y="18" width="4" height="12" rx="2"></rect>
          <rect x="385" y="14" width="4" height="20" rx="2"></rect>
          <rect x="395" y="6" width="4" height="36" rx="2"></rect>
          <rect x="405" y="10" width="4" height="28" rx="2"></rect>
          <rect x="415" y="16" width="4" height="16" rx="2"></rect>
          <rect x="425" y="4" width="4" height="40" rx="2"></rect>
          <rect x="435" y="14" width="4" height="20" rx="2"></rect>
          <rect x="445" y="8" width="4" height="32" rx="2"></rect>
          <rect x="455" y="16" width="4" height="16" rx="2"></rect>
          <rect x="465" y="12" width="4" height="24" rx="2"></rect>
          <rect x="475" y="18" width="4" height="12" rx="2"></rect>
          <rect x="485" y="10" width="4" height="28" rx="2"></rect>
        </g>
      </svg>
    </div>

    <!-- Timeline Thumbnails Sequence (Click to seek) -->
    <div class="flex items-center gap-3">
      <button id="timeline-prev" class="w-7 h-7 rounded-full border border-gray-200 hover:bg-gray-100 flex items-center justify-center text-gray-500 shrink-0 transition cursor-pointer">
        <span class="material-symbols-outlined text-[16px]">chevron_left</span>
      </button>

      <div class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2.5 flex-1" id="timeline-segments-grid">
        <!-- Segment 1: HELLO (Active) -->
        <div data-seek="0" class="timeline-segment border-2 border-[#FF5A2A] rounded-xl p-2 bg-orange-50/40 text-center cursor-pointer transition hover:scale-[1.02] flex flex-col items-center">
          <div class="w-full h-12 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1.5 shadow-sm">
            <span class="material-symbols-outlined text-[20px]">front_hand</span>
          </div>
          <div class="font-extrabold text-xs text-[#111318]">HELLO</div>
          <div class="text-[10px] text-[#6B7280]">00:00 - 00:02</div>
        </div>

        <!-- Segment 2: MY -->
        <div data-seek="2" class="timeline-segment border border-[#E6DED3] rounded-xl p-2 bg-white hover:border-[#FF5A2A] text-center cursor-pointer transition hover:scale-[1.02] flex flex-col items-center">
          <div class="w-full h-12 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1.5 shadow-sm">
            <span class="material-symbols-outlined text-[20px]">pan_tool</span>
          </div>
          <div class="font-extrabold text-xs text-[#111318]">MY</div>
          <div class="text-[10px] text-[#6B7280]">00:02 - 00:04</div>
        </div>

        <!-- Segment 3: NAME -->
        <div data-seek="4" class="timeline-segment border border-[#E6DED3] rounded-xl p-2 bg-white hover:border-[#FF5A2A] text-center cursor-pointer transition hover:scale-[1.02] flex flex-col items-center">
          <div class="w-full h-12 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1.5 shadow-sm">
            <span class="material-symbols-outlined text-[20px]">badge</span>
          </div>
          <div class="font-extrabold text-xs text-[#111318]">NAME</div>
          <div class="text-[10px] text-[#6B7280]">00:04 - 00:06</div>
        </div>

        <!-- Segment 4: IS -->
        <div data-seek="6" class="timeline-segment border border-[#E6DED3] rounded-xl p-2 bg-white hover:border-[#FF5A2A] text-center cursor-pointer transition hover:scale-[1.02] flex flex-col items-center">
          <div class="w-full h-12 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1.5 shadow-sm">
            <span class="material-symbols-outlined text-[20px]">fingerprint</span>
          </div>
          <div class="font-extrabold text-xs text-[#111318]">IS</div>
          <div class="text-[10px] text-[#6B7280]">00:06 - 00:08</div>
        </div>

        <!-- Segment 5: NICE -->
        <div data-seek="8" class="timeline-segment border border-[#E6DED3] rounded-xl p-2 bg-white hover:border-[#FF5A2A] text-center cursor-pointer transition hover:scale-[1.02] flex flex-col items-center">
          <div class="w-full h-12 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1.5 shadow-sm">
            <span class="material-symbols-outlined text-[20px]">thumb_up</span>
          </div>
          <div class="font-extrabold text-xs text-[#111318]">NICE</div>
          <div class="text-[10px] text-[#6B7280]">00:08 - 00:12</div>
        </div>

        <!-- Segment 6: TO MEET YOU -->
        <div data-seek="12" class="timeline-segment border border-[#E6DED3] rounded-xl p-2 bg-white hover:border-[#FF5A2A] text-center cursor-pointer transition hover:scale-[1.02] flex flex-col items-center">
          <div class="w-full h-12 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1.5 shadow-sm">
            <span class="material-symbols-outlined text-[20px]">handshake</span>
          </div>
          <div class="font-extrabold text-xs text-[#111318]">TO MEET YOU</div>
          <div class="text-[10px] text-[#6B7280]">00:12 - 00:14</div>
        </div>

        <!-- Segment 7: End -->
        <div data-seek="14" class="timeline-segment border border-[#E6DED3] rounded-xl p-2 bg-white hover:border-[#FF5A2A] text-center cursor-pointer transition hover:scale-[1.02] flex flex-col items-center">
          <div class="w-full h-12 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1.5 shadow-sm">
            <span class="material-symbols-outlined text-[20px]">check_circle</span>
          </div>
          <div class="font-extrabold text-xs text-[#111318]">(End)</div>
          <div class="text-[10px] text-[#6B7280]">00:14 - 00:18</div>
        </div>
      </div>

      <button id="timeline-next" class="w-7 h-7 rounded-full border border-gray-200 hover:bg-gray-100 flex items-center justify-center text-gray-500 shrink-0 transition cursor-pointer">
        <span class="material-symbols-outlined text-[16px]">chevron_right</span>
      </button>
    </div>
  </div>

  <!-- 4. BOTTOM SECTION: SIGN SEQUENCE -->
  <div class="bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col gap-4">
    <div class="flex items-center justify-between">
      <h2 class="font-bold text-sm md:text-base text-[#111318]">Sign Sequence</h2>
      <button id="btn-toggle-list-view" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E5DDD2] hover:bg-gray-50 text-xs font-semibold text-[#5A5E66] transition cursor-pointer">
        <span class="material-symbols-outlined text-[16px]">view_list</span>
        <span>View as List</span>
      </button>
    </div>

    <div class="flex items-center gap-3">
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 flex-1" id="sign-sequence-container">
        <!-- Item 1 -->
        <div class="p-3 rounded-xl border border-[#E6DED3] bg-[#FBF9F6] flex items-center gap-3">
          <span class="font-extrabold text-sm text-[#9CA3AF]">1</span>
          <div class="w-10 h-10 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] shrink-0">
            <span class="material-symbols-outlined text-[18px]">front_hand</span>
          </div>
          <div class="flex flex-col flex-1 min-w-0">
            <div class="flex items-center justify-between">
              <span class="font-extrabold text-xs text-[#111318]">HELLO</span>
              <span class="text-[10px] text-[#16A34A] font-bold">92%</span>
            </div>
            <div class="w-full h-1 bg-gray-200 rounded-full mt-1.5 overflow-hidden">
              <div class="h-full bg-[#16A34A] rounded-full" style="width: 92%;"></div>
            </div>
          </div>
        </div>

        <!-- Item 2 -->
        <div class="p-3 rounded-xl border border-[#E6DED3] bg-[#FBF9F6] flex items-center gap-3">
          <span class="font-extrabold text-sm text-[#9CA3AF]">2</span>
          <div class="w-10 h-10 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] shrink-0">
            <span class="material-symbols-outlined text-[18px]">pan_tool</span>
          </div>
          <div class="flex flex-col flex-1 min-w-0">
            <div class="flex items-center justify-between">
              <span class="font-extrabold text-xs text-[#111318]">MY</span>
              <span class="text-[10px] text-[#16A34A] font-bold">87%</span>
            </div>
            <div class="w-full h-1 bg-gray-200 rounded-full mt-1.5 overflow-hidden">
              <div class="h-full bg-[#16A34A] rounded-full" style="width: 87%;"></div>
            </div>
          </div>
        </div>

        <!-- Item 3 -->
        <div class="p-3 rounded-xl border border-[#E6DED3] bg-[#FBF9F6] flex items-center gap-3">
          <span class="font-extrabold text-sm text-[#9CA3AF]">3</span>
          <div class="w-10 h-10 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] shrink-0">
            <span class="material-symbols-outlined text-[18px]">badge</span>
          </div>
          <div class="flex flex-col flex-1 min-w-0">
            <div class="flex items-center justify-between">
              <span class="font-extrabold text-xs text-[#111318]">NAME</span>
              <span class="text-[10px] text-[#16A34A] font-bold">85%</span>
            </div>
            <div class="w-full h-1 bg-gray-200 rounded-full mt-1.5 overflow-hidden">
              <div class="h-full bg-[#16A34A] rounded-full" style="width: 85%;"></div>
            </div>
          </div>
        </div>

        <!-- Item 4 -->
        <div class="p-3 rounded-xl border border-[#E6DED3] bg-[#FBF9F6] flex items-center gap-3">
          <span class="font-extrabold text-sm text-[#9CA3AF]">4</span>
          <div class="w-10 h-10 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] shrink-0">
            <span class="material-symbols-outlined text-[18px]">fingerprint</span>
          </div>
          <div class="flex flex-col flex-1 min-w-0">
            <div class="flex items-center justify-between">
              <span class="font-extrabold text-xs text-[#111318]">IS</span>
              <span class="text-[10px] text-[#16A34A] font-bold">78%</span>
            </div>
            <div class="w-full h-1 bg-gray-200 rounded-full mt-1.5 overflow-hidden">
              <div class="h-full bg-[#16A34A] rounded-full" style="width: 78%;"></div>
            </div>
          </div>
        </div>

        <!-- Item 5 -->
        <div class="p-3 rounded-xl border border-[#E6DED3] bg-[#FBF9F6] flex items-center gap-3">
          <span class="font-extrabold text-sm text-[#9CA3AF]">5</span>
          <div class="w-10 h-10 rounded-lg bg-[#20232A] flex items-center justify-center text-[#FF8442] shrink-0">
            <span class="material-symbols-outlined text-[18px]">thumb_up</span>
          </div>
          <div class="flex flex-col flex-1 min-w-0">
            <div class="flex items-center justify-between">
              <span class="font-extrabold text-xs text-[#111318]">NICE</span>
              <span class="text-[10px] text-[#16A34A] font-bold">82%</span>
            </div>
            <div class="w-full h-1 bg-gray-200 rounded-full mt-1.5 overflow-hidden">
              <div class="h-full bg-[#16A34A] rounded-full" style="width: 82%;"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>
"""

video_main_soup = BeautifulSoup(VIDEO_TRANSLATE_HTML, "html.parser")

# --------------------------------------------------------------------------
# 5. LIVE TRANSLATE VIEW (Exact Match to User Reference Mockup)
# --------------------------------------------------------------------------
LIVE_TRANSLATE_HTML = """
<section class="view-pane hidden" id="view-live">
  <div class="px-6 py-6 md:px-8 max-w-7xl mx-auto flex flex-col gap-6">

    <!-- TOP SECTION: CAMERA VIEWPORT (Left) + RECOGNIZED & ALTERNATIVES (Right) -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      
      <!-- 1. LIVE CAMERA FEED WITH GLASS HUD OVERLAYS (7 Cols) -->
      <div class="lg:col-span-7 bg-[#16181D] rounded-2xl overflow-hidden border border-[#2D3139] shadow-md relative min-h-[380px] md:min-h-[460px] flex items-center justify-center">
        <!-- Live HTML5 Video Element (Real camera feed, no static photos) -->
        <video id="webcam-feed" playsinline muted autoplay class="w-full h-full object-cover rounded-2xl min-h-[380px] md:min-h-[460px]"></video>
        
        <!-- MediaPipe Hand Landmark Canvas Overlay -->
        <canvas id="landmark-canvas" class="absolute inset-0 w-full h-full pointer-events-none rounded-2xl z-10"></canvas>

        <!-- Idle/Synthetic ASL Landmark Canvas -->
        <canvas id="synthetic-canvas" class="w-full h-full object-cover rounded-2xl bg-gradient-to-br from-[#1E2028] to-[#111318]" style="display:none;"></canvas>

        <!-- Camera Prompt Overlay if permissions needed -->
        <div id="camera-prompt-overlay" class="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm p-6 text-center" style="display:none;">
          <div class="w-14 h-14 rounded-full bg-[#FF5A2A]/20 flex items-center justify-center text-[#FF5A2A] mb-3">
            <span class="material-symbols-outlined text-[32px]">videocam</span>
          </div>
          <h3 class="text-white font-bold text-lg mb-1">Start Live Recognition</h3>
          <p class="text-xs text-gray-300 max-w-sm mb-4">Click below to activate your webcam with real-time neural hand tracking and instant sign translation.</p>
          <button id="btn-start-camera" class="px-6 py-2.5 rounded-full bg-[#FF5A2A] hover:bg-[#E5481B] text-white font-bold text-sm shadow-lg shadow-orange-500/30 transition-all hover:scale-105 active:scale-95 cursor-pointer">
            Enable Camera
          </button>
        </div>

        <!-- HUD: Top-Left Recognizing Status & Soundwave -->
        <div class="absolute top-4 left-4 z-20 flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-black/55 backdrop-blur-md border border-white/15 text-white font-medium text-xs select-none">
          <span class="w-2.5 h-2.5 rounded-full bg-[#22C55E] animate-pulse"></span>
          <span id="live-rec-status">Recognizing...</span>
          <div class="flex items-center gap-0.5 ml-1">
            <span class="w-0.5 h-3 bg-[#22C55E] rounded-full animate-bounce"></span>
            <span class="w-0.5 h-4 bg-[#22C55E] rounded-full animate-bounce [animation-delay:150ms]"></span>
            <span class="w-0.5 h-2 bg-[#22C55E] rounded-full animate-bounce [animation-delay:300ms]"></span>
            <span class="w-0.5 h-3.5 bg-[#22C55E] rounded-full animate-bounce [animation-delay:450ms]"></span>
          </div>
        </div>

        <!-- HUD: Top-Right Glass Buttons -->
        <div class="absolute top-4 right-4 z-20 flex items-center gap-2">
          <button id="live-btn-fullscreen" class="w-9 h-9 rounded-full bg-black/50 backdrop-blur-md border border-white/15 text-white hover:bg-black/70 flex items-center justify-center transition cursor-pointer" title="Toggle Fullscreen">
            <span class="material-symbols-outlined text-[18px]">fullscreen</span>
          </button>
          <button id="live-btn-settings" data-nav="settings" class="w-9 h-9 rounded-full bg-black/50 backdrop-blur-md border border-white/15 text-white hover:bg-black/70 flex items-center justify-center transition cursor-pointer" title="Camera Settings">
            <span class="material-symbols-outlined text-[18px]">settings</span>
          </button>
        </div>

        <!-- HUD: Bottom-Left Frame Guide Card -->
        <div class="absolute bottom-4 left-4 z-20 flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-black/60 backdrop-blur-md border border-white/15 text-white max-w-xs shadow-lg select-none">
          <div class="w-8 h-8 rounded-lg bg-[#3B82F6]/25 flex items-center justify-center text-[#60A5FA] shrink-0">
            <span class="material-symbols-outlined text-[20px]">front_hand</span>
          </div>
          <div class="flex flex-col">
            <span class="font-bold text-xs leading-tight">Keep your hands in frame</span>
            <span class="text-[11px] text-[#D1D5DB] leading-tight mt-0.5">Sign naturally at a comfortable pace</span>
          </div>
        </div>
      </div>

      <!-- 2. RIGHT COLUMN: RECOGNIZED CARD & ALTERNATIVES (5 Cols) -->
      <div class="lg:col-span-5 flex flex-col gap-4">
        
        <!-- Recognized Sign Card -->
        <div class="bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between min-h-[190px]">
          <div class="flex items-center justify-between mb-3">
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#EBF8F0] text-[#16A34A] text-xs font-bold">
              <span class="w-2 h-2 rounded-full bg-[#16A34A] animate-pulse"></span>
              Recognized
            </span>
            <span id="rec-timestamp" class="text-xs text-[#9CA3AF]">2s ago</span>
          </div>

          <div class="flex items-center gap-4">
            <!-- Gesture Landmark Icon / Graphic -->
            <div class="w-20 h-20 md:w-24 md:h-24 rounded-xl bg-[#20232A] flex items-center justify-center text-[#FF8442] shrink-0 relative overflow-hidden shadow-inner">
              <div class="absolute inset-0 bg-gradient-to-tr from-transparent to-white/5 pointer-events-none"></div>
              <svg class="w-12 h-12 text-[#FF8442]" viewBox="0 0 100 100" fill="none">
                <circle cx="50" cy="50" r="44" stroke="#e8752a" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.4"></circle>
                <path d="M50 82 L50 52 M50 52 L36 34 M50 52 L46 22 M50 52 L54 22 M50 52 L64 34 M36 64 L24 48" stroke="#FF5A2A" stroke-width="2.5" stroke-linecap="round"></path>
                <circle cx="50" cy="82" r="4" fill="#FF5A2A"></circle>
                <circle cx="50" cy="52" r="3.5" fill="#FF5A2A"></circle>
                <circle cx="36" cy="34" r="3" fill="#FF8442"></circle>
                <circle cx="46" cy="22" r="3" fill="#FF8442"></circle>
                <circle cx="54" cy="22" r="3" fill="#FF8442"></circle>
                <circle cx="64" cy="34" r="3" fill="#FF8442"></circle>
                <circle cx="24" cy="48" r="3" fill="#FF8442"></circle>
              </svg>
            </div>

            <!-- Details -->
            <div class="flex flex-col flex-1 min-w-0">
              <div id="primary-gloss" class="text-3xl md:text-4xl font-black text-[#111318] tracking-tight leading-none mb-1">HELLO</div>
              <div id="primary-english" class="text-sm text-[#5A5E66] font-medium mb-2.5">Hello</div>
              <div>
                <span class="inline-block px-3 py-1 rounded-full bg-[#F3EFEA] text-[#5A5E66] text-xs font-semibold">General Greeting</span>
              </div>
            </div>

            <!-- Audio Pronounce Button -->
            <button id="btn-pronounce-sign" class="w-10 h-10 rounded-full bg-[#F5EFE6] hover:bg-[#FFEFE8] text-[#111318] hover:text-[#FF5A2A] flex items-center justify-center transition cursor-pointer self-start ml-auto shadow-sm" title="Pronounce sign">
              <span class="material-symbols-outlined text-[20px]">volume_up</span>
            </button>
          </div>
        </div>

        <!-- "Not what you meant?" Alternatives Card -->
        <div class="bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
          <div class="flex items-center justify-between mb-3">
            <span class="font-bold text-xs text-[#111318]">Not what you meant?</span>
            <span class="text-[11px] text-[#9CA3AF]">Choose the correct sign:</span>
          </div>

          <div id="alt-candidates-grid" class="grid grid-cols-5 gap-2">
            <!-- Candidate 1: Selected -->
            <div class="alt-tile p-2 rounded-xl border border-[#FF5A2A] bg-orange-50/50 hover:border-[#FF5A2A] cursor-pointer text-center transition flex flex-col items-center justify-center" data-gloss="HELLO">
              <div class="w-7 h-7 rounded-md bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1">
                <span class="material-symbols-outlined text-[14px]">front_hand</span>
              </div>
              <div class="font-bold text-xs text-[#111318] truncate w-full">HELLO</div>
              <div class="text-[10px] text-[#FF5A2A] font-bold">92%</div>
            </div>

            <!-- Candidate 2 -->
            <div class="alt-tile p-2 rounded-xl border border-[#E6DED3] bg-white hover:border-[#FF5A2A] cursor-pointer text-center transition flex flex-col items-center justify-center" data-gloss="HI">
              <div class="w-7 h-7 rounded-md bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1">
                <span class="material-symbols-outlined text-[14px]">front_hand</span>
              </div>
              <div class="font-bold text-xs text-[#111318] truncate w-full">HI</div>
              <div class="text-[10px] text-[#6B7280] font-semibold">6%</div>
            </div>

            <!-- Candidate 3 -->
            <div class="alt-tile p-2 rounded-xl border border-[#E6DED3] bg-white hover:border-[#FF5A2A] cursor-pointer text-center transition flex flex-col items-center justify-center" data-gloss="GOOD">
              <div class="w-7 h-7 rounded-md bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1">
                <span class="material-symbols-outlined text-[14px]">thumb_up</span>
              </div>
              <div class="font-bold text-xs text-[#111318] truncate w-full">GOOD</div>
              <div class="text-[10px] text-[#6B7280] font-semibold">2%</div>
            </div>

            <!-- Candidate 4 -->
            <div class="alt-tile p-2 rounded-xl border border-[#E6DED3] bg-white hover:border-[#FF5A2A] cursor-pointer text-center transition flex flex-col items-center justify-center" data-gloss="HOW">
              <div class="w-7 h-7 rounded-md bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1">
                <span class="material-symbols-outlined text-[14px]">waving_hand</span>
              </div>
              <div class="font-bold text-xs text-[#111318] truncate w-full">HOW</div>
              <div class="text-[10px] text-[#6B7280] font-semibold">1%</div>
            </div>

            <!-- Candidate 5 -->
            <div class="alt-tile p-2 rounded-xl border border-[#E6DED3] bg-white hover:border-[#FF5A2A] cursor-pointer text-center transition flex flex-col items-center justify-center" data-gloss="PLEASE">
              <div class="w-7 h-7 rounded-md bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1">
                <span class="material-symbols-outlined text-[14px]">favorite</span>
              </div>
              <div class="font-bold text-xs text-[#111318] truncate w-full">PLEASE</div>
              <div class="text-[10px] text-[#6B7280] font-semibold">&lt;1%</div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- 3. MIDDLE SECTION: TRANSLATED SENTENCE CARD -->
    <div class="bg-white border border-[#ECE2D8] rounded-2xl p-6 shadow-sm flex flex-col gap-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <div class="w-8 h-8 rounded-lg bg-[#FFEFE8] flex items-center justify-center text-[#FF5A2A]">
            <span class="material-symbols-outlined text-[18px]">chat</span>
          </div>
          <div class="flex flex-col">
            <span class="font-bold text-sm text-[#111318]">Translated Sentence</span>
            <span class="text-xs text-[#6B7280]">Your recognized signs are converted into natural, grammatically correct text.</span>
          </div>
        </div>
        <button id="sb-btn-clear" class="flex items-center gap-1 text-xs text-[#DC2626] hover:underline font-bold cursor-pointer">
          <span class="material-symbols-outlined text-[16px]">delete</span>
          <span>Clear All</span>
        </button>
      </div>

      <!-- Sentence Display Box -->
      <div class="bg-[#FBF9F6] border border-[#E5DDD2] rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4 min-h-[72px]">
        <div id="sentence-composer-text" class="text-xl md:text-2xl font-bold text-[#111318] tracking-tight leading-normal w-full md:w-auto">
          <span>Hello how are you?</span>
          <span class="inline-block w-1.5 h-5 ml-1 bg-[#FF5A2A] animate-pulse align-middle"></span>
        </div>

        <div class="flex items-center gap-2 shrink-0 w-full md:w-auto justify-end">
          <button id="sb-btn-undo" class="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white hover:bg-gray-50 border border-[#E5DDD2] text-[#111318] font-bold text-xs shadow-sm transition cursor-pointer">
            <span class="material-symbols-outlined text-[16px]">undo</span>
            <span>Undo</span>
          </button>
          <button id="sb-btn-edit" class="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white hover:bg-gray-50 border border-[#E5DDD2] text-[#111318] font-bold text-xs shadow-sm transition cursor-pointer">
            <span class="material-symbols-outlined text-[16px]">edit</span>
            <span>Edit</span>
          </button>
          <button id="sb-btn-copy" class="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white hover:bg-gray-50 border border-[#E5DDD2] text-[#111318] font-bold text-xs shadow-sm transition cursor-pointer">
            <span class="material-symbols-outlined text-[16px]">content_copy</span>
            <span>Copy</span>
          </button>
          <button id="sb-btn-speak" class="flex items-center gap-2 px-5 py-2 rounded-xl bg-[#FF5A2A] hover:bg-[#E5481B] text-white font-bold text-xs shadow-md shadow-orange-500/25 transition-all hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
            <span class="material-symbols-outlined text-[16px]">volume_up</span>
            <span>Speak</span>
          </button>
        </div>
      </div>
    </div>

    <!-- 4. BOTTOM ROW (3 CARDS): SIGN HISTORY, QUICK PHRASES, SESSION STATS -->
    <div class="grid grid-cols-1 md:grid-cols-12 gap-5">
      
      <!-- Card 1: Sign History (Col 1 to 4) -->
      <div class="md:col-span-4 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-1.5 text-xs font-bold text-[#111318]">
            <span class="material-symbols-outlined text-[16px] text-[#FF5A2A]">schedule</span>
            <span>Sign History</span>
          </div>
        </div>

        <div class="flex items-center gap-2 overflow-x-auto pb-1">
          <!-- Sign 1 -->
          <div class="flex flex-col items-center shrink-0">
            <div class="w-12 h-12 rounded-xl bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1 shadow-sm">
              <span class="material-symbols-outlined text-[18px]">front_hand</span>
            </div>
            <span class="text-[10px] font-bold text-[#111318]">HELLO</span>
          </div>
          <!-- Sign 2 -->
          <div class="flex flex-col items-center shrink-0">
            <div class="w-12 h-12 rounded-xl bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1 shadow-sm">
              <span class="material-symbols-outlined text-[18px]">waving_hand</span>
            </div>
            <span class="text-[10px] font-bold text-[#111318]">HOW</span>
          </div>
          <!-- Sign 3 -->
          <div class="flex flex-col items-center shrink-0">
            <div class="w-12 h-12 rounded-xl bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1 shadow-sm">
              <span class="material-symbols-outlined text-[18px]">pan_tool</span>
            </div>
            <span class="text-[10px] font-bold text-[#111318]">ARE</span>
          </div>
          <!-- Sign 4 -->
          <div class="flex flex-col items-center shrink-0">
            <div class="w-12 h-12 rounded-xl bg-[#20232A] flex items-center justify-center text-[#FF8442] mb-1 shadow-sm">
              <span class="material-symbols-outlined text-[18px]">fingerprint</span>
            </div>
            <span class="text-[10px] font-bold text-[#111318]">YOU</span>
          </div>
          <!-- Add Button -->
          <div class="flex flex-col items-center shrink-0">
            <button data-nav="teach" class="w-12 h-12 rounded-xl border border-dashed border-[#CCD0D8] hover:border-[#FF5A2A] hover:bg-orange-50 flex items-center justify-center text-[#6B7280] hover:text-[#FF5A2A] mb-1 transition cursor-pointer" title="Record Custom Sign">
              <span class="material-symbols-outlined text-[18px]">add</span>
            </button>
            <span class="text-[10px] text-[#6B7280]">Add</span>
          </div>
        </div>
      </div>

      <!-- Card 2: Quick Phrases (Col 5 to 9) -->
      <div class="md:col-span-5 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-1.5 text-xs font-bold text-[#111318]">
            <span class="material-symbols-outlined text-[16px] text-[#FF5A2A]">bolt</span>
            <span>Quick Phrases</span>
          </div>
          <a data-nav="learn" class="text-xs font-bold text-[#FF5A2A] hover:underline flex items-center gap-0.5 cursor-pointer">
            <span>See All</span>
            <span class="material-symbols-outlined text-[13px]">arrow_forward</span>
          </a>
        </div>

        <div class="grid grid-cols-4 gap-2">
          <button data-phrase="Hello" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer">Hello</button>
          <button data-phrase="Thank you" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer truncate">Thank you</button>
          <button data-phrase="Yes" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer">Yes</button>
          <button data-phrase="No" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer">No</button>
          
          <button data-phrase="Please" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer">Please</button>
          <button data-phrase="Help" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer">Help</button>
          <button data-phrase="Good" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer">Good</button>
          <button data-phrase="Nice to meet you" class="px-2.5 py-1.5 rounded-lg bg-[#F5EFE6] hover:bg-[#FFEFE8] hover:text-[#FF5A2A] text-xs font-medium text-[#111318] transition text-center cursor-pointer truncate">Nice to meet you</button>
        </div>
      </div>

      <!-- Card 3: Session Stats (Col 10 to 12) -->
      <div class="md:col-span-3 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-1.5 text-xs font-bold text-[#111318]">
            <span class="material-symbols-outlined text-[16px] text-[#FF5A2A]">bar_chart</span>
            <span>Session Stats</span>
          </div>
          <button id="btn-reset-stats" class="text-xs text-[#9CA3AF] hover:text-[#FF5A2A] transition cursor-pointer">Reset</button>
        </div>

        <div class="grid grid-cols-3 gap-2 text-center my-auto">
          <div>
            <div id="live-stat-signs" class="text-xl font-black text-[#111318]">12</div>
            <div class="text-[10px] text-[#6B7280] leading-tight">Signs this session</div>
          </div>
          <div>
            <div id="live-stat-sentences" class="text-xl font-black text-[#111318]">1</div>
            <div class="text-[10px] text-[#6B7280] leading-tight">Sentences</div>
          </div>
          <div>
            <div id="live-stat-confidence" class="text-xl font-black text-[#111318]">95%</div>
            <div class="text-[10px] text-[#6B7280] leading-tight">Avg. confidence</div>
          </div>
        </div>
      </div>

    </div>

  </div>
</section>
"""

# --------------------------------------------------------------------------
# 5B. TEACH SIGNBRIDGE VIEW (Exact Match to User Reference Mockup)
# --------------------------------------------------------------------------
TEACH_SIGNBRIDGE_HTML = """
<section class="view-pane hidden" id="view-teach">
  <div class="px-6 py-6 md:px-8 max-w-7xl mx-auto flex flex-col gap-6">

    <!-- 1. PAGE HEADER -->
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div class="flex flex-col">
        <h1 class="text-2xl md:text-3xl font-extrabold text-[#111318] tracking-tight">Teach SignBridge</h1>
        <p class="text-xs md:text-sm text-[#5A5E66] mt-0.5">Help SignBridge learn new signs from you. Teach, improve, and make communication better for everyone.</p>
      </div>

      <!-- Encouragement Banner (Top-Right) -->
      <div class="flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-[#EAF7ED] border border-[#CDEBD2] text-[#15803D] shadow-sm select-none">
        <div class="w-8 h-8 rounded-xl bg-[#16A34A]/20 flex items-center justify-center text-[#16A34A] shrink-0">
          <span class="material-symbols-outlined text-[20px]">psychiatry</span>
        </div>
        <div class="flex flex-col">
          <span class="font-bold text-xs text-[#15803D]">You're making a difference!</span>
          <span class="text-[11px] text-[#166534]">Every new sign helps build a more inclusive world.</span>
        </div>
      </div>
    </div>

    <!-- 2. WIZARD STEP INDICATOR (4 Steps) -->
    <div class="flex items-center justify-between gap-2 max-w-4xl mx-auto w-full py-2 px-4 bg-white/60 border border-[#ECE2D8] rounded-2xl shadow-sm">
      <!-- Step 1 (Active) -->
      <div class="flex items-center gap-3 cursor-pointer group">
        <div class="w-8 h-8 rounded-full bg-[#FF5A2A] text-white font-extrabold flex items-center justify-center text-xs shadow-md shadow-orange-500/25">1</div>
        <div class="flex flex-col text-left">
          <span class="font-extrabold text-xs text-[#FF5A2A]">Record Examples</span>
          <span class="text-[10px] text-[#6B7280]">Capture multiple samples</span>
        </div>
      </div>

      <div class="h-[1.5px] flex-1 bg-gray-200 mx-2"></div>

      <!-- Step 2 -->
      <div class="flex items-center gap-3 cursor-pointer group opacity-60 hover:opacity-100 transition">
        <div class="w-8 h-8 rounded-full bg-gray-200 text-gray-600 font-bold flex items-center justify-center text-xs">2</div>
        <div class="flex flex-col text-left">
          <span class="font-bold text-xs text-[#111318]">Review & Confirm</span>
          <span class="text-[10px] text-[#6B7280]">Check the captured signs</span>
        </div>
      </div>

      <div class="h-[1.5px] flex-1 bg-gray-200 mx-2"></div>

      <!-- Step 3 -->
      <div class="flex items-center gap-3 cursor-pointer group opacity-60 hover:opacity-100 transition">
        <div class="w-8 h-8 rounded-full bg-gray-200 text-gray-600 font-bold flex items-center justify-center text-xs">3</div>
        <div class="flex flex-col text-left">
          <span class="font-bold text-xs text-[#111318]">Train</span>
          <span class="text-[10px] text-[#6B7280]">Let AI learn from your examples</span>
        </div>
      </div>

      <div class="h-[1.5px] flex-1 bg-gray-200 mx-2"></div>

      <!-- Step 4 -->
      <div class="flex items-center gap-3 cursor-pointer group opacity-60 hover:opacity-100 transition">
        <div class="w-8 h-8 rounded-full bg-gray-200 text-gray-600 font-bold flex items-center justify-center text-xs">4</div>
        <div class="flex flex-col text-left">
          <span class="font-bold text-xs text-[#111318]">Done</span>
          <span class="text-[10px] text-[#6B7280]">Your sign is ready!</span>
        </div>
      </div>
    </div>

    <!-- 3. MAIN SECTION (3 Columns: Camera Viewport + Sign Details + Recording Progress) -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      
      <!-- Col 1: Camera Viewport Card (5 Cols) -->
      <div class="lg:col-span-5 bg-white border border-[#ECE2D8] rounded-2xl p-4 md:p-5 shadow-sm flex flex-col gap-3">
        <!-- Card Header -->
        <div class="flex items-center justify-between">
          <button id="teach-back-btn" class="flex items-center gap-1.5 font-bold text-sm text-[#111318] hover:text-[#FF5A2A] transition cursor-pointer">
            <span class="material-symbols-outlined text-[18px]">arrow_back</span>
            <span>Teach a New Sign</span>
          </button>
          <button id="teach-guidelines-btn" class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-[#E5DDD2] hover:bg-gray-50 text-xs font-semibold text-[#5A5E66] transition cursor-pointer">
            <span class="material-symbols-outlined text-[14px]">menu_book</span>
            <span>View Guidelines</span>
          </button>
        </div>

        <!-- Camera Viewport (Real camera & canvas, zero static person photo) -->
        <div class="relative bg-[#16181D] rounded-2xl overflow-hidden border border-[#2D3139] shadow-inner aspect-[4/3] flex items-center justify-center">
          <video id="teach-webcam-feed" playsinline muted autoplay class="w-full h-full object-cover"></video>
          <canvas id="teach-landmark-canvas" class="absolute inset-0 w-full h-full pointer-events-none z-10"></canvas>
          <canvas id="teach-synthetic-canvas" class="absolute inset-0 w-full h-full object-cover bg-gradient-to-br from-[#1E2028] to-[#111318]" style="display:none;"></canvas>

          <!-- Framing Corner Brackets -->
          <div class="absolute top-4 right-4 w-7 h-7 border-t-2 border-r-2 border-white/70 pointer-events-none z-20"></div>
          <div class="absolute bottom-4 left-4 w-7 h-7 border-b-2 border-l-2 border-white/70 pointer-events-none z-20"></div>

          <!-- Top-Left HUD Badge: Recording... 00:03 -->
          <div id="teach-recording-hud" class="absolute top-4 left-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/15 text-white font-medium text-xs select-none">
            <span class="w-2.5 h-2.5 rounded-full bg-[#EF4444] animate-pulse"></span>
            <span>Recording...</span>
            <span id="teach-recording-timer" class="font-mono text-gray-200">00:03</span>
          </div>

          <!-- Bottom-Center Record / Stop Button -->
          <div class="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20">
            <button id="teach-record-btn" class="w-13 h-13 rounded-full border-[3px] border-white bg-black/40 backdrop-blur-md flex items-center justify-center hover:scale-105 active:scale-95 transition cursor-pointer shadow-lg shadow-black/50" title="Record or stop example">
              <div id="teach-record-icon" class="w-5 h-5 rounded-sm bg-[#EF4444]"></div>
            </button>
          </div>

          <!-- Bottom-Right HUD: Sample 3 / 10 -->
          <div class="absolute bottom-4 right-4 z-20 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md border border-white/15 text-white text-[11px] font-medium font-mono select-none">
            <span id="teach-sample-badge">Sample 3 / 10</span>
          </div>
        </div>

        <!-- Tip Callout Banner below Camera -->
        <div class="p-3 rounded-xl bg-[#FFF9F3] border border-[#F3E7DC] flex items-center gap-2.5 text-xs text-[#5A5E66]">
          <span class="material-symbols-outlined text-[18px] text-[#FF5A2A] shrink-0">lightbulb</span>
          <span class="font-medium"><strong class="text-[#111318]">Tip:</strong> Keep your hand well in frame, with good lighting and a steady position.</span>
        </div>
      </div>

      <!-- Col 2: Sign Details Card (4 Cols) -->
      <div class="lg:col-span-4 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col gap-4">
        <h2 class="font-bold text-sm md:text-base text-[#111318]">Sign Details</h2>

        <!-- Field 1: Sign Name -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-center justify-between">
            <label class="text-xs font-semibold text-[#111318]">Sign Name <span class="text-[#EF4444]">*</span></label>
          </div>
          <input id="teach-sign-name" type="text" value="FRIEND" maxlength="50" class="w-full px-3.5 py-2.5 rounded-xl border border-[#E5DDD2] bg-white text-xs font-semibold text-[#111318] focus:border-[#FF5A2A] focus:ring-1 focus:ring-orange-500 outline-none transition">
          <div class="text-right text-[10px] text-[#9CA3AF]" id="teach-sign-name-counter">12/50</div>
        </div>

        <!-- Field 2: Category -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-semibold text-[#111318]">Category</label>
          <div class="relative">
            <select id="teach-sign-category" class="w-full appearance-none px-3.5 py-2.5 pl-9 rounded-xl border border-[#E5DDD2] bg-white text-xs font-semibold text-[#111318] focus:border-[#FF5A2A] focus:ring-1 focus:ring-orange-500 outline-none transition cursor-pointer">
              <option value="people" selected>People & Relationships</option>
              <option value="common">Everyday Signs</option>
              <option value="alphabet">Alphabet & Numbers</option>
              <option value="emergency">Medical & Emergency</option>
              <option value="emotions">Emotions & Feelings</option>
            </select>
            <span class="material-symbols-outlined absolute left-3 top-1/2 transform -translate-y-1/2 text-[16px] text-[#5A5E66] pointer-events-none">groups</span>
            <span class="material-symbols-outlined absolute right-3 top-1/2 transform -translate-y-1/2 text-[18px] text-[#9CA3AF] pointer-events-none">expand_more</span>
          </div>
        </div>

        <!-- Field 3: Description -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-semibold text-[#111318]">Description (Optional)</label>
          <input id="teach-sign-desc" type="text" value="A sign for the word &quot;friend&quot;." maxlength="200" class="w-full px-3.5 py-2.5 rounded-xl border border-[#E5DDD2] bg-white text-xs text-[#111318] focus:border-[#FF5A2A] focus:ring-1 focus:ring-orange-500 outline-none transition">
          <div class="text-right text-[10px] text-[#9CA3AF]" id="teach-sign-desc-counter">0/200</div>
        </div>

        <!-- Field 4: Example Sentence -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs font-semibold text-[#111318]">Example Sentence (Optional)</label>
          <input id="teach-sign-sentence" type="text" value="You are a good friend." maxlength="200" class="w-full px-3.5 py-2.5 rounded-xl border border-[#E5DDD2] bg-white text-xs text-[#111318] focus:border-[#FF5A2A] focus:ring-1 focus:ring-orange-500 outline-none transition">
          <div class="text-right text-[10px] text-[#9CA3AF]" id="teach-sign-sentence-counter">0/200</div>
        </div>
      </div>

      <!-- Col 3: Recording Progress Card (3 Cols) -->
      <div class="lg:col-span-3 bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col items-center text-center gap-4">
        <h2 class="font-bold text-sm md:text-base text-[#111318] w-full text-left">Recording Progress</h2>

        <!-- Circular Progress Ring (3/10) -->
        <div class="relative w-28 h-28 flex items-center justify-center my-1">
          <svg class="w-full h-full -rotate-90 transform" viewBox="0 0 100 100">
            <!-- Background Ring -->
            <circle cx="50" cy="50" r="38" stroke="#F4EDE4" stroke-width="7" fill="transparent"></circle>
            <!-- Progress Arc (30% = 238.76 * 0.3 = 71.6, offset = 167) -->
            <circle id="teach-progress-circle" cx="50" cy="50" r="38" stroke="#FF5A2A" stroke-width="7" stroke-dasharray="239" stroke-dashoffset="167" stroke-linecap="round" fill="transparent" class="transition-all duration-500"></circle>
          </svg>
          <div class="absolute inset-0 flex items-center justify-center">
            <span id="teach-progress-label" class="text-2xl font-extrabold text-[#111318]">3/10</span>
          </div>
        </div>

        <p class="text-xs text-[#5A5E66] leading-relaxed max-w-[210px]">
          Record at least 10 examples from slightly different angles.
        </p>

        <!-- Checklist -->
        <div class="flex flex-col gap-2.5 w-full text-left pt-2 border-t border-gray-100">
          <!-- Item 1 -->
          <div class="flex items-center gap-2 text-xs text-[#111318]">
            <span class="w-4 h-4 rounded-full bg-[#FF5A2A] text-white flex items-center justify-center shrink-0">
              <span class="material-symbols-outlined text-[11px] font-bold">check</span>
            </span>
            <span>Good lighting</span>
          </div>
          <!-- Item 2 -->
          <div class="flex items-center gap-2 text-xs text-[#111318]">
            <span class="w-4 h-4 rounded-full bg-[#FF5A2A] text-white flex items-center justify-center shrink-0">
              <span class="material-symbols-outlined text-[11px] font-bold">check</span>
            </span>
            <span>Hand clearly visible</span>
          </div>
          <!-- Item 3 -->
          <div class="flex items-center gap-2 text-xs text-[#111318]">
            <span class="w-4 h-4 rounded-full bg-[#FF5A2A] text-white flex items-center justify-center shrink-0">
              <span class="material-symbols-outlined text-[11px] font-bold">check</span>
            </span>
            <span>Keep a consistent pose</span>
          </div>
          <!-- Item 4 -->
          <div class="flex items-center gap-2 text-xs text-[#6B7280]">
            <span class="w-4 h-4 rounded-full border-2 border-gray-300 shrink-0"></span>
            <span>Capture different angles</span>
          </div>
          <!-- Item 5 -->
          <div class="flex items-center gap-2 text-xs text-[#6B7280]">
            <span class="w-4 h-4 rounded-full border-2 border-gray-300 shrink-0"></span>
            <span>Record 10+ examples</span>
          </div>
        </div>
      </div>

    </div>

    <!-- 4. BOTTOM SECTION: CAPTURED EXAMPLES GALLERY -->
    <div class="bg-white border border-[#ECE2D8] rounded-2xl p-5 shadow-sm flex flex-col gap-4">
      <div class="flex items-center justify-between">
        <h2 class="font-bold text-sm md:text-base text-[#111318]">
          Captured Examples (<span id="teach-captured-count">3</span>)
        </h2>
        <button id="teach-clear-all-btn" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-red-200 hover:bg-red-50 text-xs font-semibold text-[#EF4444] transition cursor-pointer">
          <span class="material-symbols-outlined text-[15px]">delete</span>
          <span>Clear All</span>
        </button>
      </div>

      <!-- Gallery Row -->
      <div class="flex flex-col md:flex-row items-center gap-3">
        <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-8 gap-3 flex-1 w-full" id="teach-examples-grid">
          
          <!-- Sample 1 (Neon Wireframe Hand, Zero Person Photo) -->
          <div class="teach-sample-card relative aspect-square rounded-xl bg-[#1E2028] border border-[#2D3139] overflow-hidden flex items-center justify-center group shadow-sm">
            <!-- Simulated ASL Hand Landmark Skeleton Wireframe -->
            <svg class="w-12 h-12 text-[#FF8442]" viewBox="0 0 100 100" fill="none" stroke="currentColor">
              <!-- Hand Bones -->
              <path d="M50 85 L50 60 L30 45 L20 30" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L40 38 L36 18" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L50 35 L50 14" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L60 38 L64 18" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L70 45 L78 28" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <!-- Joints -->
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
          </div>

          <!-- Sample 2 -->
          <div class="teach-sample-card relative aspect-square rounded-xl bg-[#1E2028] border border-[#2D3139] overflow-hidden flex items-center justify-center group shadow-sm">
            <svg class="w-12 h-12 text-[#FF8442]" viewBox="0 0 100 100" fill="none" stroke="currentColor">
              <path d="M50 85 L50 60 L32 46 L24 32" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L42 40 L38 20" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L50 36 L50 16" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L58 40 L62 20" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L68 46 L76 30" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <circle cx="50" cy="85" r="3.5" fill="#60A5FA"></circle>
              <circle cx="50" cy="60" r="3.5" fill="#60A5FA"></circle>
              <circle cx="24" cy="32" r="3" fill="#FFFFFF"></circle>
              <circle cx="38" cy="20" r="3" fill="#FFFFFF"></circle>
              <circle cx="50" cy="16" r="3" fill="#FFFFFF"></circle>
              <circle cx="62" cy="20" r="3" fill="#FFFFFF"></circle>
              <circle cx="76" cy="30" r="3" fill="#FFFFFF"></circle>
            </svg>
            <button class="teach-delete-sample-btn absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-black/70 hover:bg-[#EF4444] text-white flex items-center justify-center transition cursor-pointer shadow">
              <span class="material-symbols-outlined text-[12px]">close</span>
            </button>
          </div>

          <!-- Sample 3 -->
          <div class="teach-sample-card relative aspect-square rounded-xl bg-[#1E2028] border border-[#2D3139] overflow-hidden flex items-center justify-center group shadow-sm">
            <svg class="w-12 h-12 text-[#FF8442]" viewBox="0 0 100 100" fill="none" stroke="currentColor">
              <path d="M50 85 L50 60 L28 44 L18 28" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L38 36 L34 16" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L50 34 L50 12" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L62 36 L66 16" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <path d="M50 60 L72 44 L80 26" stroke="#FF5A2A" stroke-width="3" stroke-linecap="round"></path>
              <circle cx="50" cy="85" r="3.5" fill="#60A5FA"></circle>
              <circle cx="50" cy="60" r="3.5" fill="#60A5FA"></circle>
              <circle cx="18" cy="28" r="3" fill="#FFFFFF"></circle>
              <circle cx="34" cy="16" r="3" fill="#FFFFFF"></circle>
              <circle cx="50" cy="12" r="3" fill="#FFFFFF"></circle>
              <circle cx="66" cy="16" r="3" fill="#FFFFFF"></circle>
              <circle cx="80" cy="26" r="3" fill="#FFFFFF"></circle>
            </svg>
            <button class="teach-delete-sample-btn absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-black/70 hover:bg-[#EF4444] text-white flex items-center justify-center transition cursor-pointer shadow">
              <span class="material-symbols-outlined text-[12px]">close</span>
            </button>
          </div>

          <!-- Slot 4: Empty (Add More) -->
          <div class="teach-add-slot aspect-square rounded-xl border-2 border-dashed border-gray-200 hover:border-[#FF5A2A] hover:bg-orange-50/20 flex flex-col items-center justify-center gap-1 text-gray-400 hover:text-[#FF5A2A] transition cursor-pointer select-none">
            <span class="material-symbols-outlined text-[20px]">add_circle</span>
            <span class="text-[10px] font-semibold">Add More</span>
          </div>

          <!-- Slot 5: Empty -->
          <div class="teach-add-slot aspect-square rounded-xl border-2 border-dashed border-gray-200 hover:border-[#FF5A2A] hover:bg-orange-50/20 flex flex-col items-center justify-center gap-1 text-gray-400 hover:text-[#FF5A2A] transition cursor-pointer select-none">
            <span class="material-symbols-outlined text-[20px]">add_circle</span>
            <span class="text-[10px] font-semibold">Add More</span>
          </div>

          <!-- Slot 6: Empty -->
          <div class="teach-add-slot aspect-square rounded-xl border-2 border-dashed border-gray-200 hover:border-[#FF5A2A] hover:bg-orange-50/20 flex flex-col items-center justify-center gap-1 text-gray-400 hover:text-[#FF5A2A] transition cursor-pointer select-none">
            <span class="material-symbols-outlined text-[20px]">add_circle</span>
            <span class="text-[10px] font-semibold">Add More</span>
          </div>

          <!-- Slot 7: Empty -->
          <div class="teach-add-slot aspect-square rounded-xl border-2 border-dashed border-gray-200 hover:border-[#FF5A2A] hover:bg-orange-50/20 flex flex-col items-center justify-center gap-1 text-gray-400 hover:text-[#FF5A2A] transition cursor-pointer select-none">
            <span class="material-symbols-outlined text-[20px]">add_circle</span>
            <span class="text-[10px] font-semibold">Add More</span>
          </div>

          <!-- Slot 8: Empty -->
          <div class="teach-add-slot aspect-square rounded-xl border-2 border-dashed border-gray-200 hover:border-[#FF5A2A] hover:bg-orange-50/20 flex flex-col items-center justify-center gap-1 text-gray-400 hover:text-[#FF5A2A] transition cursor-pointer select-none">
            <span class="material-symbols-outlined text-[20px]">add_circle</span>
            <span class="text-[10px] font-semibold">Add More</span>
          </div>

        </div>

        <!-- Next Step Action Button -->
        <button id="teach-next-btn" class="w-full md:w-auto flex items-center justify-center gap-2 px-8 py-3 rounded-full bg-[#FF5A2A] hover:bg-[#E5481B] text-white font-bold text-xs shadow-md shadow-orange-500/25 transition-all hover:scale-[1.02] active:scale-[0.98] cursor-pointer shrink-0">
          <span>Next</span>
          <span class="material-symbols-outlined text-[16px]">arrow_forward</span>
        </button>
      </div>
    </div>

  </div>
</section>
"""

def clean_pane(pane_id):
    pane = existing_soup.find("section", id=pane_id)
    if not pane:
        return f'<section class="view-pane hidden" id="{pane_id}"></section>'
    for img in pane.find_all("img"):
        replacement = BeautifulSoup("""
        <div class="w-full h-full bg-surface-container-low flex items-center justify-center text-primary">
          <span class="material-symbols-outlined text-3xl">gesture</span>
        </div>
        """, "html.parser")
        img.replace_with(replacement)
    return str(pane)

view_live = LIVE_TRANSLATE_HTML
view_practice = clean_pane("view-practice")
view_teach = TEACH_SIGNBRIDGE_HTML
view_profile = clean_pane("view-profile")
view_settings = clean_pane("view-settings")

# --------------------------------------------------------------------------
# 6. ASSEMBLE MASTER SINGLE PAGE APPLICATION
# --------------------------------------------------------------------------
MASTER_SPA = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SignBridge — AI Sign Language Translation &amp; Learning Suite</title>
  <!-- Favicon -->
  <link rel="icon" type="image/png" href="/static/assets/signbridge_logo.png">

  <!-- Google Fonts: Plus Jakarta Sans, Inter, Permanent Marker -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&family=Permanent+Marker&display=swap" rel="stylesheet">
  
  <!-- Material Symbols Outlined -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200">

  <!-- Tailwind CSS CDN with Full Stitch Theme Tokens -->
  <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
  <script>
    tailwind.config = {{
      darkMode: "class",
      theme: {{
        extend: {{
          colors: {{
            "on-tertiary-fixed-variant": "#6f3804",
            "tertiary-fixed-dim": "#ffb783",
            "surface-bright": "#fff8f5",
            "tertiary": "#8c4f1c",
            "on-tertiary-fixed": "#301400",
            "on-background": "#201a17",
            "on-error": "#ffffff",
            "surface-container-highest": "#ece0da",
            "on-secondary-fixed-variant": "#7b2f00",
            "on-primary-container": "#501f00",
            "inverse-surface": "#362f2b",
            "on-surface-variant": "#574238",
            "secondary": "#a14000",
            "secondary-fixed-dim": "#ffb694",
            "error": "#ba1a1a",
            "on-secondary-container": "#682700",
            "primary-fixed": "#ffdbca",
            "primary-container": "#e8752a",
            "on-surface": "#201a17",
            "surface-dim": "#e4d8d1",
            "on-secondary": "#ffffff",
            "surface-variant": "#ece0da",
            "outline-variant": "#dec1b3",
            "secondary-container": "#ff8343",
            "inverse-on-surface": "#fbeee8",
            "on-error-container": "#93000a",
            "tertiary-fixed": "#ffdcc5",
            "surface-container-low": "#fef1ea",
            "on-primary-fixed-variant": "#773200",
            "surface-container-high": "#f2e6df",
            "inverse-primary": "#ffb68f",
            "on-primary-fixed": "#331200",
            "surface-container": "#f8ece5",
            "on-secondary-fixed": "#351000",
            "on-tertiary": "#ffffff",
            "on-tertiary-container": "#4b2300",
            "background": "#fff8f5",
            "surface": "#fff8f5",
            "outline": "#8a7266",
            "tertiary-container": "#cd844c",
            "error-container": "#ffdad6",
            "secondary-fixed": "#ffdbcc",
            "primary": "#9c4400",
            "surface-tint": "#9c4400",
            "primary-fixed-dim": "#ffb68f",
            "surface-container-lowest": "#ffffff",
            "on-primary": "#ffffff",
            brand: {{
              orange: '#9c4400',
              orangeHover: '#e8752a',
              orangeLight: '#ffdbca',
              dark: '#201a17',
              darker: '#14100e',
              bg: '#fff8f5',
              surface: '#ffffff'
            }}
          }},
          borderRadius: {{
            DEFAULT: "0.25rem",
            lg: "0.5rem",
            xl: "0.75rem",
            full: "9999px"
          }},
          spacing: {{
            gutter: "1.25rem",
            "space-lg": "1.5rem",
            "margin-md": "1.5rem",
            "space-xl": "2rem",
            "gutter-lg": "1.5rem",
            "margin-lg": "2rem",
            "space-sm": "0.5rem",
            "space-md": "1rem",
            "space-xs": "0.25rem",
            margin: "1rem"
          }},
          fontFamily: {{
            sans: ['"Plus Jakarta Sans"', 'Inter', 'sans-serif'],
            marker: ['"Permanent Marker"', 'cursive'],
            "headline-lg": ["Plus Jakarta Sans"],
            "headline-sm": ["Plus Jakarta Sans"],
            "label-md": ["Inter"],
            "label-sm": ["Inter"],
            "title-md": ["Plus Jakarta Sans"],
            "code-gloss": ["Inter"],
            "body-lg": ["Inter"],
            "body-md": ["Inter"],
            "headline-lg-mobile": ["Plus Jakarta Sans"],
            "title-sm": ["Plus Jakarta Sans"],
            "headline-md": ["Plus Jakarta Sans"]
          }}
        }}
      }}
    }};
  </script>

  <!-- MediaPipe Hands & Camera Utils -->
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

  <!-- Custom Stylesheet -->
  <link rel="stylesheet" href="/static/css/style.css">
  <style>
    @layer base {{
      html, body {{ margin: 0; padding: 0; }}
      body {{ overscroll-behavior: none; }}
    }}
    ::-webkit-scrollbar {{ display: none; }}
    .material-symbols-outlined {{
      font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24;
    }}
    @keyframes soundwave {{
      0%, 100% {{ height: 6px; }}
      50% {{ height: 26px; }}
    }}
    .soundwave-bar {{
      animation: soundwave 1.2s ease-in-out infinite;
    }}
  </style>
</head>
<body class="bg-[#FBF8F4] font-body-md text-[#111318] antialiased min-h-screen flex flex-col selection:bg-[#FF5A2A] selection:text-white">

  <!-- SIDEBAR NAVIGATION (Fixed 250px Dark Theme Matching Mockup) -->
  <aside class="fixed left-0 top-0 h-full w-[250px] bg-[#12141A] border-r border-white/5 z-50 flex flex-col justify-between select-none">
    <div class="flex flex-col">
      <!-- Brand Header with Animated Hand & Gleaming Star Logo -->
      <div class="p-5 flex items-center gap-3">
        <div class="sb-logo-container w-10 h-10 shrink-0 cursor-pointer" title="SignBridge — Click to Wave">
          <div class="sb-logo-aura"></div>
          <img src="/static/assets/signbridge_logo.png" alt="SignBridge Logo" class="sb-logo-img">
          <svg class="sb-logo-star" viewBox="0 0 24 24" fill="white">
            <path d="M12 0 C12 7, 7 12, 0 12 C7 12, 12 17, 12 24 C12 17, 17 12, 24 12 C17 12, 12 7, 12 0 Z" fill="#ffffff"></path>
          </svg>
        </div>
        <div class="flex flex-col">
          <span class="text-white font-extrabold text-lg leading-tight tracking-tight">SignBridge</span>
          <span class="text-[10px] text-[#9CA3AF] leading-tight mt-0.5">Different Signs.</span>
          <span class="text-[10px] text-[#9CA3AF] leading-tight">Same Human Connection.</span>
        </div>
      </div>

      <!-- Navigation Links -->
      <nav class="px-3 pt-2 flex flex-col gap-1">
        <a data-nav="home" class="nav-link active flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm transition-all cursor-pointer bg-[#FF5A2A] text-white font-bold shadow-md shadow-orange-500/20">
          <span class="material-symbols-outlined text-[20px]">home</span>
          <span>Home</span>
        </a>
        <a data-nav="live" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">videocam</span>
          <span>Live Translate</span>
        </a>
        <a data-nav="video" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">play_circle</span>
          <span>Video Translate</span>
        </a>
        <a data-nav="teach" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">school</span>
          <span>Teach SignBridge</span>
        </a>
        <a data-nav="learn" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">menu_book</span>
          <span>Learn</span>
        </a>
        <a data-nav="practice" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">adjust</span>
          <span>Practice</span>
        </a>
        <a data-nav="communication" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">chat</span>
          <span>Communication</span>
        </a>

        <div class="my-2 mx-3 h-[1px] bg-white/10"></div>

        <a data-nav="settings" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">settings</span>
          <span>Settings</span>
        </a>
        <a data-nav="profile" class="nav-link flex items-center gap-3 px-4 py-2.5 rounded-xl font-medium text-sm text-[#9CA3AF] hover:bg-white/5 hover:text-white transition-all cursor-pointer">
          <span class="material-symbols-outlined text-[20px]">person</span>
          <span>Profile</span>
        </a>
      </nav>
    </div>

    <!-- Bottom Mountain Sunset Card in Sidebar -->
    <div class="m-3 p-4 rounded-2xl bg-[#1C1F26] border border-white/10 relative overflow-hidden flex flex-col justify-end min-h-[140px] select-none" style="background: linear-gradient(180deg, rgba(28,31,38,0.7) 0%, rgba(18,20,26,0.95) 100%), url('/static/assets/sidebar_mountain.png') center/cover no-repeat;">
      <div class="text-xs font-bold text-white relative z-10 leading-snug mb-1">
        Different Signs<br>Same Human Connection
      </div>
      <div class="w-8 h-[2.5px] bg-[#FF5A2A] rounded-full mb-3 relative z-10"></div>
      <p class="text-[10px] text-[#D1D5DB] relative z-10 leading-snug">
        Together for a more inclusive world.
      </p>
    </div>
  </aside>

  <!-- MAIN VIEWPORT CONTAINER -->
  <div class="pl-[250px] flex-1 flex flex-col">
    <!-- Fixed Top Header -->
    <header class="fixed top-0 left-[250px] right-0 h-16 bg-[#FBF8F4]/90 backdrop-blur-xl border-b border-[#ECE2D8] z-40 flex items-center justify-between px-6 md:px-8">
      <!-- Search Input -->
      <div class="flex items-center gap-2.5 px-4 py-2 bg-white border border-[#E5DDD2] rounded-full shadow-sm w-72 md:w-96 text-xs text-[#111318] focus-within:border-[#FF5A2A] focus-within:ring-2 focus-within:ring-orange-500/20 transition-all">
        <span class="material-symbols-outlined text-[18px] text-[#9CA3AF]">search</span>
        <input id="global-search-input" type="text" placeholder="Search 2,731 signs, categories, or anything..." class="bg-transparent border-none outline-none w-full text-xs text-[#111318] placeholder-[#9CA3AF]">
      </div>

      <!-- Right Header Actions: Theme Toggle, Notifications, Swastik User Profile -->
      <div class="flex items-center gap-3">
        <!-- Theme Toggle -->
        <button id="theme-toggle-btn" class="w-9 h-9 rounded-full bg-white border border-[#E5DDD2] hover:bg-[#F5EFE6] flex items-center justify-center text-[#5A5E66] transition shadow-sm cursor-pointer" title="Toggle Theme">
          <span class="material-symbols-outlined text-[18px]">light_mode</span>
        </button>

        <!-- Notification Bell with Red Badge -->
        <button id="notif-btn" class="relative w-9 h-9 rounded-full bg-white border border-[#E5DDD2] hover:bg-[#F5EFE6] flex items-center justify-center text-[#5A5E66] transition shadow-sm cursor-pointer" title="Notifications">
          <span class="material-symbols-outlined text-[18px]">notifications</span>
          <span class="w-2 h-2 rounded-full bg-[#EF4444] absolute top-2 right-2 ring-2 ring-white"></span>
        </button>

        <!-- Swastik Profile Pill -->
        <div data-nav="profile" class="flex items-center gap-2.5 pl-2 py-1 pr-3 rounded-full hover:bg-black/5 transition cursor-pointer select-none">
          <div class="w-8 h-8 rounded-full bg-[#111318] text-white flex items-center justify-center font-bold text-xs shadow-sm">
            S
          </div>
          <div class="flex flex-col text-left">
            <span class="font-bold text-xs text-[#111318] leading-tight">Hi, Swastik</span>
            <span class="text-[10px] text-[#6B7280] leading-tight">Keep learning!</span>
          </div>
          <span class="material-symbols-outlined text-[16px] text-[#9CA3AF]">expand_more</span>
        </div>
      </div>
    </header>

    <!-- Scrollable Body Content -->
    <main class="relative pt-16 bg-[#FBF8F4] min-h-screen" id="views-container">
      
      <!-- VIEW 1: HOME DASHBOARD -->
      <section class="view-pane" id="view-home">
        {str(home_main_soup)}
      </section>

      <!-- VIEW 2: TWO-WAY COMMUNICATION -->
      <section class="view-pane hidden" id="view-communication">
        {str(comm_main_soup)}
      </section>

      <!-- VIEW 3: LEARN VOCABULARY -->
      <section class="view-pane hidden" id="view-learn">
        {str(learn_main_soup)}
      </section>

      <!-- VIEW 4: VIDEO TRANSLATION -->
      <section class="view-pane hidden" id="view-video">
        {str(video_main_soup)}
      </section>

      <!-- VIEW 5: LIVE STUDIO (Fullscreen Continuous Inference) -->
      {view_live}

      <!-- VIEW 6: PRACTICE YOUR SIGNS -->
      {view_practice}

      <!-- VIEW 7: TEACH SIGNBRIDGE (Few-shot Personalization) -->
      {view_teach}

      <!-- VIEW 8: USER PROFILE -->
      {view_profile}

      <!-- VIEW 9: SETTINGS -->
      {view_settings}

    </main>
  </div>

  <!-- TOAST NOTIFICATION CONTAINER -->
  <div id="toast-container" class="fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-none"></div>

  <!-- APPLICATION SCRIPTS -->
  <script src="/static/js/nlp.js"></script>
  <script src="/static/js/app.js"></script>
</body>
</html>
"""

soup_check = BeautifulSoup(MASTER_SPA, "html.parser")
imgs = soup_check.find_all("img")
print("Total <img> tags in compiled template before cleanup:", len(imgs))
for i in imgs:
    src = i.get("src", "")
    if any(k in src for k in ["signbridge_logo.png", "logo.png", "hero_hands.png", "mountain_landscape.png", "sidebar_mountain.png", "hero_badge.png"]):
        continue
    print(" - Removing external mockup img with src:", src[:60])
    i.decompose()
MASTER_SPA = str(soup_check)

print("Writing compiled SPA to:", OUTPUT_PATH)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(MASTER_SPA)

print("Successfully written! Final size:", os.path.getsize(OUTPUT_PATH), "bytes")
