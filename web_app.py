from __future__ import annotations
import os
import threading
import queue
import time
import subprocess
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, Response, jsonify
import re

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(APP_DIR, "out")

app = Flask(__name__)
progress_q: dict[str, queue.Queue[str]] = {}

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TED Clip Finder</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root { 
      --bg: #0a0a0a; 
      --panel: #1a1a1a; 
      --text: #ffffff; 
      --muted: #888888; 
      --brand: #6366f1; 
      --brand2: #8b5cf6; 
      --border: #333333; 
      --success: #22c55e;
      --error: #ef4444;
      --accent: #f59e0b;
      --accent2: #ec4899;
      --accent3: #06b6d4;
      --gradient-bg: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
      --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
      --gradient-success: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
      --gradient-accent: linear-gradient(135deg, #f59e0b 0%, #ec4899 100%);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: 'Inter', system-ui, -apple-system, sans-serif; 
      background: var(--gradient-bg);
      color: var(--text); 
      line-height: 1.6;
      min-height: 100vh;
    }
    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 40px 20px;
    }
    header { 
      text-align: center;
      margin-bottom: 60px;
      padding: 80px 40px;
      background: var(--gradient-primary);
      border-radius: 32px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
    }
    header::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(circle at 30% 20%, rgba(139, 92, 246, 0.3) 0%, transparent 50%),
                  radial-gradient(circle at 70% 80%, rgba(245, 158, 11, 0.2) 0%, transparent 50%);
      pointer-events: none;
    }
    h1 { 
      font-size: clamp(2.5rem, 6vw, 4rem); 
      margin-bottom: 24px;
      font-weight: 800;
      color: white;
      position: relative;
      z-index: 1;
      letter-spacing: -0.02em;
    }
    .accent { 
      background: var(--gradient-accent);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .subtitle { 
      color: rgba(255,255,255,0.9); 
      font-size: 1.25rem;
      max-width: 700px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
      font-weight: 400;
      line-height: 1.6;
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 40px;
      margin-bottom: 60px;
    }
    .card { 
      background: rgba(26, 26, 26, 0.9); 
      backdrop-filter: blur(20px);
      border-radius: 24px; 
      padding: 32px; 
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 20px 40px rgba(0,0,0,0.3);
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }
    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: var(--gradient-primary);
      opacity: 0.5;
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 30px 60px rgba(99, 102, 241, 0.2);
      border-color: rgba(99, 102, 241, 0.3);
    }
    .card h3 {
      font-size: 1.5rem;
      margin-bottom: 24px;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
    }
    .form-group {
      margin-bottom: 24px;
    }
    label { 
      display: block; 
      font-weight: 600; 
      margin-bottom: 12px; 
      color: var(--text);
      font-size: 1rem;
      letter-spacing: -0.01em;
    }
    input[type="text"], input[type="file"], select { 
      width: 100%; 
      padding: 16px 20px; 
      border-radius: 12px; 
      border: 1px solid rgba(255, 255, 255, 0.1); 
      background: rgba(0, 0, 0, 0.3); 
      color: var(--text);
      font-size: 16px;
      transition: all 0.3s ease;
      backdrop-filter: blur(10px);
    }
    input[type="text"]:focus, input[type="file"]:focus, select:focus {
      outline: none;
      border-color: var(--brand);
      box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
      background: rgba(0, 0, 0, 0.5);
    }
    input[type="text"]::placeholder {
      color: var(--muted);
    }
    .row { 
      display: grid; 
      grid-template-columns: 1fr 1fr;
      gap: 16px; 
    }
    .checkbox-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 16px;
    }
    .checkbox-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .checkbox-group input[type="checkbox"] {
      width: auto;
    }
    .btn { 
      background: var(--gradient-primary); 
      color: white; 
      border: none; 
      padding: 18px 32px; 
      border-radius: 16px; 
      font-weight: 700; 
      cursor: pointer; 
      font-size: 16px;
      transition: all 0.3s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      position: relative;
      overflow: hidden;
      letter-spacing: -0.01em;
    }
    .btn::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
      transition: left 0.5s;
    }
    .btn:hover::before {
      left: 100%;
    }
    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 20px 40px rgba(99, 102, 241, 0.4);
    }
    .btn-primary { background: var(--gradient-primary); }
    .btn-success { background: var(--gradient-success); }
    .btn-danger { background: linear-gradient(135deg, var(--error), #dc2626); }
    .btn-secondary { 
      background: rgba(255, 255, 255, 0.1); 
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.2);
      backdrop-filter: blur(10px);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.2);
      box-shadow: 0 10px 20px rgba(255, 255, 255, 0.1);
    }
    .log { 
      background: rgba(0, 0, 0, 0.6); 
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.1); 
      border-radius: 16px; 
      padding: 20px; 
      height: 300px; 
      overflow-y: auto; 
      font-family: 'JetBrains Mono', 'Fira Code', monospace; 
      font-size: 14px;
      line-height: 1.5;
      backdrop-filter: blur(10px);
    }
    .clips-section {
      background: rgba(26, 26, 26, 0.9);
      backdrop-filter: blur(20px);
      border-radius: 24px;
      padding: 32px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 20px 40px rgba(0,0,0,0.3);
      position: relative;
      overflow: hidden;
    }
    .clips-section::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: var(--gradient-accent);
      opacity: 0.5;
    }
    .clips-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 32px;
    }
    .clips-header h3 {
      font-size: 1.75rem;
      font-weight: 800;
      color: var(--text);
      margin: 0;
    }
    .clips-controls {
      display: flex;
      gap: 16px;
    }
    .clips-grid { 
      display: grid; 
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); 
      gap: 28px; 
      align-items: start;
    }
    .clip-card {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      overflow: hidden;
      transition: all 0.4s ease;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      height: fit-content;
      backdrop-filter: blur(15px);
      position: relative;
    }
    .clip-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: var(--gradient-primary);
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    .clip-card:hover::before {
      opacity: 1;
    }
    .clip-card:hover {
      transform: translateY(-6px);
      box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
      border-color: rgba(99, 102, 241, 0.4);
    }
    .clip-video {
      width: 100%;
      aspect-ratio: 16/9;
      background: #000;
      object-fit: cover;
    }
    .clip-info {
      padding: 24px;
    }
    .clip-title {
      font-weight: 700;
      margin-bottom: 16px;
      color: var(--text);
      font-size: 1.05rem;
      line-height: 1.4;
      letter-spacing: -0.01em;
    }
    .clip-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 20px;
    }
    .clip-actions .btn {
      padding: 12px 16px;
      font-size: 0.9rem;
      font-weight: 700;
      text-align: center;
      min-height: 44px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border-radius: 12px;
    }
    .clip-actions .btn-success {
      background: var(--gradient-success);
      color: white;
      border: none;
    }
    .clip-actions .btn-secondary {
      background: rgba(255, 255, 255, 0.1);
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.2);
      backdrop-filter: blur(10px);
    }
    .clip-actions .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    .clip-actions .btn-success:hover {
      box-shadow: 0 8px 20px rgba(34, 197, 94, 0.4);
    }
    .status {
      padding: 16px 24px;
      border-radius: 16px;
      margin-bottom: 24px;
      font-weight: 600;
      display: none;
      border: 1px solid;
      backdrop-filter: blur(10px);
    }
    .status.success {
      background: rgba(34, 197, 94, 0.15);
      color: #22c55e;
      border-color: rgba(34, 197, 94, 0.3);
    }
    .status.error {
      background: rgba(239, 68, 68, 0.15);
      color: #ef4444;
      border-color: rgba(239, 68, 68, 0.3);
    }
    .empty-state {
      text-align: center;
      padding: 60px 20px;
      color: var(--muted);
    }
    .empty-state-icon {
      font-size: 4rem;
      margin-bottom: 20px;
      opacity: 0.6;
    }
    .empty-state p {
      font-size: 1.1rem;
      font-weight: 500;
    }
    @media (max-width: 768px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
      .clips-grid {
        grid-template-columns: 1fr;
      }
      .clips-header {
        flex-direction: column;
        gap: 16px;
        align-items: stretch;
      }
      .clips-controls {
        justify-content: stretch;
      }
      .row, .checkbox-row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Transform Videos Into <span class="accent">Perfect Clips</span></h1>
      <p class="subtitle">AI-powered video analysis that extracts the best speaker-focused moments from any TED talk or educational video. No slides, no audience - just pure content.</p>
    </header>

    <div id="status" class="status"></div>

    <div class="main-grid">
      <div class="card">
        <h3>🎬 Video Input</h3>
        <form id="form" enctype="multipart/form-data">
          <div class="form-group">
            <label>YouTube Link</label>
            <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." />
          </div>
          
          <div class="form-group">
            <label>Or Upload Video File</label>
            <input name="file" type="file" accept="video/*" />
          </div>
          
          <div class="form-group">
            <label>⚙️ Settings</label>
            <div class="row">
              <div>
                <label>Number of Clips</label>
                <select name="num_clips">
                  <option value="5" selected>5 clips</option>
                  <option value="3">3 clips</option>
                  <option value="10">10 clips</option>
                </select>
              </div>
              <div>
                <label>Duration (seconds)</label>
                <select name="clip_duration">
                  <option value="30" selected>30 seconds</option>
                  <option value="20">20 seconds</option>
                  <option value="45">45 seconds</option>
                </select>
              </div>
            </div>
          </div>

          <div class="form-group">
            <div class="checkbox-row">
              <div class="checkbox-group">
                <input type="checkbox" name="strict" id="strict" checked />
                <label for="strict">Strict person-only</label>
              </div>
              <div class="checkbox-group">
                <input type="checkbox" name="avoid_slides" id="avoid_slides" checked />
                <label for="avoid_slides">Avoid slides</label>
              </div>
            </div>
          </div>

          <button type="submit" class="btn btn-primary" style="width: 100%;">
            🚀 Create Clips
          </button>
        </form>
      </div>
      
      <div class="card">
        <h3>⚡ Live Progress</h3>
        <div id="log" class="log">Ready to process your video...</div>
      </div>
    </div>

    <div class="clips-section">
      <div class="clips-header">
        <h3>✨ Generated Clips</h3>
        <div class="clips-controls">
          <button id="refreshClips" onclick="backfillClips()" class="btn btn-success">
            🔄 Refresh
          </button>
          <button onclick="clearClips()" class="btn btn-danger">
            🗑️ Clear All
          </button>
        </div>
      </div>
      <div id="clips" class="clips-grid">
        <div class="empty-state">
          <div class="empty-state-icon">🎬</div>
          <p>Clips will appear here after processing...</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    const logEl = document.getElementById('log');
    const clipsEl = document.getElementById('clips');
    const form = document.getElementById('form');
    const statusEl = document.getElementById('status');

    function showStatus(message, type) {
      statusEl.textContent = message;
      statusEl.className = `status ${type}`;
      statusEl.style.display = 'block';
      setTimeout(() => statusEl.style.display = 'none', 5000);
    }

    function appendLog(msg) { 
      logEl.textContent += msg + '\\n'; 
      logEl.scrollTop = logEl.scrollHeight; 
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      logEl.textContent = '';
      clipsEl.innerHTML = '<p style="color: var(--muted); margin: 0;">Processing...</p>';
      
      const fd = new FormData(form);
      const url = fd.get('url');
      const file = fd.get('file');
      
      if (!url && (!file || !file.name)) {
        showStatus('Please provide a YouTube URL or upload a video file', 'error');
        return;
      }

      try {
        const res = await fetch('/start', { method: 'POST', body: fd });
        const data = await res.json();
        const job_id = data.job_id;
        
        appendLog('🎬 Job ' + job_id + ' started...');
        showStatus('Processing started! Check the progress log.', 'success');
        
        // Clear existing clips when starting new job
        clipsEl.innerHTML = `
          <div class="empty-state">
            <div class="empty-state-icon">⏳</div>
            <p>Processing video...</p>
          </div>
        `;
        
        const es = new EventSource(`/events/${job_id}`);
        es.onmessage = (ev) => {
          const msg = ev.data;
          if (msg.startsWith('CLIP:')) {
            const url = msg.slice(5);
            const filename = decodeURIComponent(url.split('/').pop());
            clearEmptyState();
            addClipCard(url, filename);
          } else if (msg === 'DONE') {
            es.close();
            appendLog('✅ Processing complete!');
            showStatus('All clips generated successfully!', 'success');
            backfillClips();
          } else {
            appendLog(msg);
          }
        };
        es.onerror = () => { 
          appendLog('⚠️ Connection lost. Processing may still continue...');
        };
      } catch(err) { 
        appendLog('❌ Failed to start: ' + err);
        showStatus('Failed to start processing: ' + err.message, 'error');
      }
    });

    // Backfill any clips if SSE missed them or arrived after completion
    async function backfillClips(){
      const refreshBtn = document.getElementById('refreshClips');
      refreshBtn.textContent = '🔄 Loading...';
      refreshBtn.disabled = true;
      
      try{
        const res = await fetch('/latest-clips?minutes=2&max=10');
        const data = await res.json();
        console.log('Fetched clips:', data);
        
        if(Array.isArray(data.clips) && data.clips.length > 0){
          clearEmptyState();
          
          let addedCount = 0;
          data.clips.forEach(url => {
            // Avoid duplicates by checking if URL already present
            const exists = Array.from(clipsEl.querySelectorAll('.clip-card')).some(card => 
              card.querySelector('video')?.src === location.origin + url
            );
            if(exists) return;
            
            const filename = decodeURIComponent(url.split('/').pop());
            addClipCard(url, filename);
            addedCount++;
          });
          
          showStatus(`Found ${data.clips.length} clips (${addedCount} new)`, 'success');
        } else {
          if (clipsEl.children.length === 0 || clipsEl.querySelector('.empty-state')) {
            showEmptyState();
          }
          showStatus('No clips found', 'error');
        }
      }catch(e){ 
        console.error('Error fetching clips:', e);
        showStatus('Error loading clips: ' + e.message, 'error');
      } finally {
        refreshBtn.textContent = '🔄 Refresh Clips';
        refreshBtn.disabled = false;
      }
    }
    
    // Helper functions for clip management
    function clearEmptyState() {
      const emptyState = clipsEl.querySelector('.empty-state');
      if (emptyState) {
        emptyState.remove();
      }
    }
    
    function showEmptyState() {
      clipsEl.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🎬</div>
          <p>Clips will appear here after processing...</p>
        </div>
      `;
    }
    
    function addClipCard(url, filename) {
      // Create the card structure with better semantic HTML
      const card = document.createElement('div');
      card.className = 'clip-card';
      card.setAttribute('data-url', url);
      
      // Video element with better controls
      const video = document.createElement('video');
      video.src = url;
      video.controls = true;
      video.preload = 'metadata';
      video.className = 'clip-video';
      video.setAttribute('controlsList', 'nodownload');
      video.setAttribute('disablePictureInPicture', 'true');
      
      // Info container
      const info = document.createElement('div');
      info.className = 'clip-info';
      
      // Title with better truncation
      const title = document.createElement('div');
      title.className = 'clip-title';
      title.textContent = filename.replace(/\.(mp4|avi|mov|mkv)$/i, '');
      title.setAttribute('title', filename);
      
      // Action buttons container
      const actions = document.createElement('div');
      actions.className = 'clip-actions';
      
      // Download button with better styling
      const downloadBtn = document.createElement('a');
      downloadBtn.href = url;
      downloadBtn.download = filename;
      downloadBtn.className = 'btn btn-success';
      downloadBtn.innerHTML = '<span>📥</span> Download';
      downloadBtn.setAttribute('aria-label', `Download ${filename}`);
      
      // Open button with better styling
      const openBtn = document.createElement('a');
      openBtn.href = url;
      openBtn.target = '_blank';
      openBtn.rel = 'noopener noreferrer';
      openBtn.className = 'btn btn-secondary';
      openBtn.innerHTML = '<span>🔗</span> Open';
      openBtn.setAttribute('aria-label', `Open ${filename} in new tab`);
      
      // Assemble the structure
      actions.appendChild(downloadBtn);
      actions.appendChild(openBtn);
      info.appendChild(title);
      info.appendChild(actions);
      card.appendChild(video);
      card.appendChild(info);
      
      // Add with smooth animation
      clipsEl.appendChild(card);
      
      // Trigger fade-in animation
      setTimeout(() => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.4s ease';
        setTimeout(() => {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, 10);
      }, 10);
    }
    
    // Clear all clips from display
    function clearClips() {
      showEmptyState();
      showStatus('Clips cleared from display', 'success');
    }
    
    // Try backfill on page load and after each submit
    backfillClips();
  </script>
</body>
</html>
"""


def _enqueue(q: queue.Queue[str], message: str) -> None:
    try:
        q.put_nowait(message)
    except Exception:
        pass


def _run_tool(job_id: str, url: str | None, filepath: str | None, params: dict[str, str]) -> None:
    q = progress_q[job_id]
    os.makedirs(OUT_DIR, exist_ok=True)

    cmd = ["python", os.path.join(APP_DIR, "tool.py")]
    if filepath:
        cmd += ["--input-file", filepath]
    elif url:
        cmd.append(url)
    else:
        _enqueue(q, "❌ No input provided")
        _enqueue(q, "DONE")
        return

    # Common args
    cmd += [
        "--output-dir", OUT_DIR,
        "--num-clips", params.get("num_clips", "5"),
        "--clip-duration", params.get("clip_duration", "30"),
        "--detector", "hog",
        "--strict-person-only",
        "--strict-min-ratio", "0.6",
        "--avoid-slides",
        "--slide-max-ratio", "0.2",
    ]

    _enqueue(q, "🚀 STARTED")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    start_ts = time.time()
    published: set[str] = set()
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=env) as p:
            for line in p.stdout or []:
                line = line.rstrip()
                if not line:
                    continue
                _enqueue(q, line)
                # Robustly extract saved clip path
                m = re.search(r"Saved\s+(.+?\.mp4)\b", line)
                if m:
                    clip_path = m.group(1).strip('"')
                    try:
                        abs_path = clip_path if os.path.isabs(clip_path) else os.path.abspath(os.path.join(APP_DIR, clip_path))
                        abs_path = os.path.normpath(abs_path)
                        rel_path = os.path.relpath(abs_path, OUT_DIR).replace('\\','/')
                        url_path = url_for('serve_file', path=rel_path)
                        _enqueue(q, f"CLIP:{url_path}")
                        published.add(abs_path)
                    except Exception as e:
                        _enqueue(q, f"WARN: could not publish clip path: {e}")
        # Fallback: scan output directory for recent mp4s in this run
        try:
            for root, _, files in os.walk(OUT_DIR):
                for f in files:
                    if not f.lower().endswith('.mp4'):
                        continue
                    abs_p = os.path.normpath(os.path.join(root, f))
                    if abs_p in published:
                        continue
                    try:
                        if os.path.getmtime(abs_p) >= start_ts - 5:
                            rel_path = os.path.relpath(abs_p, OUT_DIR).replace('\\','/')
                            url_path = url_for('serve_file', path=rel_path)
                            _enqueue(q, f"CLIP:{url_path}")
                    except Exception:
                        pass
        except Exception:
            pass
        _enqueue(q, "DONE")
    except Exception as e:
        _enqueue(q, f"❌ ERROR: {e}")
        _enqueue(q, "DONE")


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE)


@app.route('/favicon.ico', methods=["GET"])
def favicon():
    return '', 204


@app.route("/start", methods=["POST"])
def start():
    url = request.form.get("url") or None
    num_clips = request.form.get("num_clips", "5")
    clip_duration = request.form.get("clip_duration", "30")

    uploaded_path = None
    f = request.files.get("file")
    if f and f.filename:
        os.makedirs(OUT_DIR, exist_ok=True)
        uploaded_path = os.path.join(OUT_DIR, f.filename)
        f.save(uploaded_path)

    job_id = str(int(time.time() * 1000))
    progress_q[job_id] = queue.Queue()

    t = threading.Thread(target=_run_tool, args=(job_id, url, uploaded_path, {"num_clips": num_clips, "clip_duration": clip_duration}), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/events/<job_id>", methods=["GET"])
def events(job_id: str):
    q = progress_q.get(job_id)
    if q is None:
        return ("", 404)

    def gen():
        yield ": SSE connection established\n\n"
        while True:
            try:
                msg = q.get(timeout=120)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {msg}\n\n"
            if msg == "DONE":
                break
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(gen(), mimetype='text/event-stream', headers=headers)


@app.route("/out/<path:path>", methods=["GET"])
def serve_file(path: str):
    return send_from_directory(OUT_DIR, path, as_attachment=False)


@app.route("/latest-clips", methods=["GET"])
def latest_clips():
    """Return recent clips in the output directory so the UI can backfill if SSE misses events."""
    max_items = int(request.args.get("max", 20))
    minutes = float(request.args.get("minutes", 2))  # Only last 2 minutes by default
    threshold = time.time() - minutes * 60.0
    found: list[str] = []
    try:
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                if not f.lower().endswith(".mp4"):
                    continue
                # Skip full video files, only get clip files
                if not ("_clips" in root and f.startswith("clip_")):
                    continue
                abs_p = os.path.join(root, f)
                try:
                    if os.path.getmtime(abs_p) >= threshold:
                        rel = os.path.relpath(abs_p, OUT_DIR).replace('\\','/')
                        found.append(url_for('serve_file', path=rel))
                except Exception:
                    continue
    except Exception:
        pass
    # Sort newest first and cap
    found.sort(reverse=True)
    
    # Add cache-busting headers
    response = jsonify({"clips": found[:max_items]})
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7864))
    app.run(host="0.0.0.0", port=port, debug=False)