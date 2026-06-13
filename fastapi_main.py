import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from feature_extractor import extract_mel_spectrogram
import librosa

# ---------- MODEL ----------
MODEL_PATH = "models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/samples", StaticFiles(directory="my_samples"), name="samples")

# ---------- UTILS ----------
def get_waveform_plot(audio_path, sr=8000):
    y, _ = librosa.load(audio_path, sr=sr)
    x = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color='#00f5ff', width=1.5)
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, color='rgba(255,255,255,0.3)'),
        yaxis=dict(showgrid=False, zeroline=False, color='rgba(255,255,255,0.3)'),
    )
    return fig.to_html(full_html=False)

def predict_logic(file_path):
    x = extract_mel_spectrogram(file_path)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    top_indices = preds.argsort()[-3:][::-1]
    top_values = preds[top_indices]
    top_results = [(int(i), float(v)*100) for i, v in zip(top_indices, top_values)]
    predicted_class = top_results[0][0]
    confidence = top_results[0][1]
    return predicted_class, confidence, top_results

def generate_result_html(file_path, filename):
    predicted_class, confidence, top_results = predict_logic(file_path)
    top_html = "".join([
        f"""<div class='bar-row'>
              <span class='bar-label'>{digit}</span>
              <div class='bar-track'>
                <div class='bar-fill' style='width:{conf:.1f}%;background:{"#00f5ff" if i==0 else "#7c3aed" if i==1 else "#f59e0b"}'></div>
              </div>
              <span class='bar-pct'>{conf:.1f}%</span>
            </div>"""
        for i, (digit, conf) in enumerate(top_results)
    ])
    waveform_div = get_waveform_plot(file_path)
    conf_color = "#00f5ff" if confidence > 80 else "#f59e0b" if confidence > 50 else "#ff4444"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Munaf AI — Result</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

    :root {{
      --cyan: #00f5ff;
      --violet: #7c3aed;
      --gold: #f59e0b;
      --bg: #000008;
      --glass: rgba(255,255,255,0.04);
      --border: rgba(0,245,255,0.15);
    }}

    body {{
      background: var(--bg);
      color: white;
      font-family: 'Rajdhani', sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* ── STARS ── */
    .stars {{ position: fixed; inset: 0; z-index: 0; pointer-events: none; }}
    .star {{
      position: absolute;
      border-radius: 50%;
      background: white;
      animation: twinkle var(--d) ease-in-out infinite alternate;
    }}
    @keyframes twinkle {{ from {{ opacity: 0.1; }} to {{ opacity: 1; }} }}

    /* ── NAV ── */
    nav {{
      position: fixed; top: 0; left: 0; right: 0; z-index: 100;
      padding: 18px 40px;
      display: flex; align-items: center; justify-content: space-between;
      backdrop-filter: blur(24px);
      background: rgba(0,0,8,0.6);
      border-bottom: 1px solid rgba(0,245,255,0.08);
    }}
    .nav-logo {{
      font-family: 'Orbitron', sans-serif;
      font-size: 1.1rem; font-weight: 900;
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      letter-spacing: 0.1em;
    }}
    .nav-back {{
      font-family: 'Rajdhani', sans-serif;
      font-size: 0.85rem; font-weight: 600;
      color: rgba(0,245,255,0.7);
      text-decoration: none; letter-spacing: 0.12em;
      display: flex; align-items: center; gap: 8px;
      transition: color 0.3s;
    }}
    .nav-back:hover {{ color: var(--cyan); }}
    .nav-back::before {{ content: '←'; font-size: 1rem; }}

    /* ── MAIN ── */
    main {{
      position: relative; z-index: 1;
      min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
      padding: 120px 20px 60px;
    }}

    .result-container {{
      width: 100%; max-width: 680px;
      display: flex; flex-direction: column; gap: 20px;
    }}

    /* ── BIG DIGIT CARD ── */
    .digit-hero {{
      position: relative; overflow: hidden;
      background: var(--glass);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 50px 40px;
      text-align: center;
      backdrop-filter: blur(20px);
      animation: slideUp 0.7s cubic-bezier(.22,.68,0,1.2) both;
    }}
    @keyframes slideUp {{
      from {{ opacity: 0; transform: translateY(60px) scale(0.95); }}
      to   {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    .digit-hero::before {{
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 50% 0%, rgba(0,245,255,0.08), transparent 70%);
      pointer-events: none;
    }}
    .digit-eyebrow {{
      font-family: 'Rajdhani', sans-serif;
      font-size: 0.8rem; font-weight: 600;
      letter-spacing: 0.35em;
      color: rgba(0,245,255,0.5);
      text-transform: uppercase;
      margin-bottom: 16px;
    }}
    .digit-number {{
      font-family: 'Orbitron', sans-serif;
      font-size: 9rem; font-weight: 900;
      line-height: 1;
      background: linear-gradient(180deg, #ffffff 0%, var(--cyan) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 0 40px rgba(0,245,255,0.4));
      margin-bottom: 20px;
    }}
    .confidence-badge {{
      display: inline-flex; align-items: center; gap: 10px;
      padding: 10px 24px;
      border-radius: 100px;
      border: 1px solid {conf_color}44;
      background: {conf_color}11;
      font-family: 'Orbitron', sans-serif;
      font-size: 0.95rem;
      color: {conf_color};
      letter-spacing: 0.08em;
    }}
    .conf-dot {{
      width: 8px; height: 8px;
      border-radius: 50%;
      background: {conf_color};
      box-shadow: 0 0 8px {conf_color};
      animation: pulse 1.5s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 1; }}
      50% {{ transform: scale(1.6); opacity: 0.5; }}
    }}

    /* ── ROW CARDS ── */
    .row-cards {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}

    .card {{
      background: var(--glass);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 28px 24px;
      backdrop-filter: blur(20px);
      animation: slideUp 0.7s cubic-bezier(.22,.68,0,1.2) both;
    }}
    .card:nth-child(1) {{ animation-delay: 0.1s; }}
    .card:nth-child(2) {{ animation-delay: 0.2s; }}

    .card-title {{
      font-family: 'Rajdhani', sans-serif;
      font-size: 0.75rem; font-weight: 600;
      letter-spacing: 0.3em;
      color: rgba(255,255,255,0.3);
      text-transform: uppercase;
      margin-bottom: 20px;
    }}

    /* ── BARS ── */
    .bar-row {{
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 14px;
    }}
    .bar-label {{
      font-family: 'Orbitron', sans-serif;
      font-size: 1rem; font-weight: 700;
      color: white; width: 20px; text-align: center;
    }}
    .bar-track {{
      flex: 1; height: 6px;
      background: rgba(255,255,255,0.06);
      border-radius: 3px; overflow: hidden;
    }}
    .bar-fill {{
      height: 100%; border-radius: 3px;
      animation: barIn 1s cubic-bezier(.22,.68,0,1.2) both;
      box-shadow: 0 0 8px currentColor;
    }}
    @keyframes barIn {{
      from {{ width: 0 !important; }}
    }}
    .bar-pct {{
      font-family: 'Rajdhani', sans-serif;
      font-size: 0.85rem; font-weight: 600;
      color: rgba(255,255,255,0.5);
      width: 44px; text-align: right;
    }}

    /* ── AUDIO ── */
    .audio-card {{
      animation: slideUp 0.7s 0.3s cubic-bezier(.22,.68,0,1.2) both;
    }}
    audio {{
      width: 100%;
      filter: invert(1) hue-rotate(180deg);
      border-radius: 8px;
      margin-bottom: 20px;
    }}
    .waveform-wrap {{ border-radius: 12px; overflow: hidden; }}

    /* ── SCANLINE ── */
    .scanline {{
      position: fixed; inset: 0; z-index: 9999;
      background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.03) 0px, rgba(0,0,0,0.03) 1px,
        transparent 1px, transparent 2px
      );
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div class="scanline"></div>
  <div class="stars" id="stars"></div>

  <nav>
    <span class="nav-logo">MUNAF·AI</span>
    <a href="/" class="nav-back">NEW PREDICTION</a>
  </nav>

  <main>
    <div class="result-container">

      <div class="digit-hero">
        <div class="digit-eyebrow">Recognized Digit</div>
        <div class="digit-number">{predicted_class}</div>
        <div class="confidence-badge">
          <span class="conf-dot"></span>
          {confidence:.2f}% Confidence
        </div>
      </div>

      <div class="row-cards">
        <div class="card">
          <div class="card-title">Top Predictions</div>
          {top_html}
        </div>

        <div class="card audio-card">
          <div class="card-title">Audio Playback</div>
          <audio controls>
            <source src="/{filename}">
          </audio>
          <div class="card-title" style="margin-top:16px">Waveform</div>
          <div class="waveform-wrap">{waveform_div}</div>
        </div>
      </div>

    </div>
  </main>

  <script>
    // Generate stars
    const starsEl = document.getElementById('stars');
    for (let i = 0; i < 180; i++) {{
      const s = document.createElement('div');
      s.className = 'star';
      const size = Math.random() * 2 + 0.5;
      s.style.cssText = `
        left:${{Math.random()*100}}%;top:${{Math.random()*100}}%;
        width:${{size}}px;height:${{size}}px;
        opacity:${{Math.random()*0.7+0.1}};
        --d:${{(Math.random()*3+1.5).toFixed(1)}}s;
        animation-delay:${{(Math.random()*4).toFixed(1)}}s;
      `;
      starsEl.appendChild(s);
    }}
  </script>
</body>
</html>"""


# ---------- HOME UI ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Munaf AI — Voice Recognition</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap" rel="stylesheet">

  <style>
    /* ═══════════════════════════════════════
       BASE RESET + CSS VARS
    ═══════════════════════════════════════ */
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

    :root {
      --cyan:   #00f5ff;
      --violet: #7c3aed;
      --gold:   #f59e0b;
      --rose:   #f43f5e;
      --bg:     #000008;
      --glass:  rgba(255,255,255,0.04);
      --border: rgba(0,245,255,0.12);
      --border2: rgba(124,58,237,0.2);
    }

    html { scroll-behavior: smooth; }

    body {
      background: var(--bg);
      color: white;
      font-family: 'Rajdhani', sans-serif;
      overflow-x: hidden;
      cursor: none;
    }

    /* ═══════════════════════════════════════
       CUSTOM CURSOR
    ═══════════════════════════════════════ */
    .cursor {
      position: fixed; z-index: 9999;
      pointer-events: none;
      mix-blend-mode: difference;
    }
    .cursor-dot {
      width: 8px; height: 8px;
      background: var(--cyan);
      border-radius: 50%;
      transform: translate(-50%,-50%);
      transition: transform 0.1s;
    }
    .cursor-ring {
      width: 40px; height: 40px;
      border: 1px solid rgba(0,245,255,0.5);
      border-radius: 50%;
      transform: translate(-50%,-50%);
      transition: transform 0.15s, width 0.3s, height 0.3s, opacity 0.3s;
    }

    /* ═══════════════════════════════════════
       PARALLAX STAR LAYERS
    ═══════════════════════════════════════ */
    .parallax-layer {
      position: fixed; inset: 0; z-index: 0;
      pointer-events: none;
      will-change: transform;
    }
    .star {
      position: absolute;
      border-radius: 50%;
      background: white;
      animation: twinkle var(--d) ease-in-out infinite alternate;
    }
    @keyframes twinkle { from { opacity: 0.05; } to { opacity: 1; } }

    /* Nebula blobs */
    .nebula {
      position: fixed;
      border-radius: 50%;
      pointer-events: none;
      filter: blur(80px);
      z-index: 0;
      animation: nebulaDrift 20s ease-in-out infinite alternate;
    }
    @keyframes nebulaDrift {
      from { transform: translate(0,0) scale(1); }
      to   { transform: translate(40px,-30px) scale(1.1); }
    }

    /* Scanlines overlay */
    .scanlines {
      position: fixed; inset: 0; z-index: 2;
      background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.025) 0px,
        rgba(0,0,0,0.025) 1px,
        transparent 1px, transparent 3px
      );
      pointer-events: none;
    }

    /* ═══════════════════════════════════════
       NAV
    ═══════════════════════════════════════ */
    nav {
      position: fixed; top: 0; left: 0; right: 0; z-index: 100;
      padding: 20px 48px;
      display: flex; align-items: center; justify-content: space-between;
      backdrop-filter: blur(30px);
      background: rgba(0,0,8,0.5);
      border-bottom: 1px solid rgba(0,245,255,0.06);
      transition: background 0.4s;
    }
    .nav-logo {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.05rem; font-weight: 900;
      letter-spacing: 0.15em;
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .nav-links {
      display: flex; gap: 36px; list-style: none;
    }
    .nav-links a {
      font-size: 0.78rem; font-weight: 600;
      letter-spacing: 0.2em;
      color: rgba(255,255,255,0.35);
      text-decoration: none; text-transform: uppercase;
      transition: color 0.3s;
    }
    .nav-links a:hover { color: var(--cyan); }

    /* ═══════════════════════════════════════
       SECTIONS (full-page)
    ═══════════════════════════════════════ */
    section {
      position: relative; z-index: 10;
      min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
      flex-direction: column;
      padding: 120px 24px 80px;
    }

    /* ═══════════════════════════════════════
       SECTION 1 — HERO
    ═══════════════════════════════════════ */
    .hero { overflow: hidden; }

    .hero-eyebrow {
      font-size: 0.72rem; font-weight: 600;
      letter-spacing: 0.45em; text-transform: uppercase;
      color: rgba(0,245,255,0.5);
      margin-bottom: 24px;
      display: flex; align-items: center; gap: 14px;
    }
    .hero-eyebrow::before, .hero-eyebrow::after {
      content: '';
      display: block; height: 1px; width: 40px;
      background: rgba(0,245,255,0.3);
    }

    .hero-title {
      font-family: 'Orbitron', sans-serif;
      font-size: clamp(3rem, 8vw, 7rem);
      font-weight: 900;
      line-height: 0.95;
      text-align: center;
      letter-spacing: -0.02em;
    }
    .hero-title .line-1 {
      display: block;
      background: linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.7) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-title .line-2 {
      display: block;
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 0 30px rgba(0,245,255,0.3));
    }
    .hero-title .line-3 {
      display: block;
      font-size: 0.45em;
      letter-spacing: 0.3em;
      color: rgba(255,255,255,0.2);
      font-weight: 400;
      margin-top: 10px;
    }

    .hero-sub {
      margin-top: 36px;
      font-size: 1.1rem; font-weight: 400;
      color: rgba(255,255,255,0.35);
      letter-spacing: 0.04em;
      max-width: 480px;
      text-align: center;
      line-height: 1.7;
    }

    .hero-cta {
      margin-top: 50px;
      display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;
    }
    .btn-primary {
      position: relative; overflow: hidden;
      padding: 16px 44px;
      border-radius: 100px;
      border: none; cursor: none;
      font-family: 'Orbitron', sans-serif;
      font-size: 0.78rem; font-weight: 700;
      letter-spacing: 0.2em;
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      color: white;
      text-decoration: none;
      transition: transform 0.3s, box-shadow 0.3s;
      box-shadow: 0 0 30px rgba(0,245,255,0.3);
    }
    .btn-primary:hover {
      transform: translateY(-3px) scale(1.03);
      box-shadow: 0 0 60px rgba(0,245,255,0.5);
    }
    .btn-primary::after {
      content: '';
      position: absolute; inset: 0;
      background: linear-gradient(135deg, rgba(255,255,255,0.2), transparent);
    }
    .btn-ghost {
      padding: 15px 40px;
      border-radius: 100px;
      border: 1px solid rgba(0,245,255,0.25);
      font-family: 'Rajdhani', sans-serif;
      font-size: 0.85rem; font-weight: 600;
      letter-spacing: 0.15em;
      color: rgba(0,245,255,0.7);
      text-decoration: none;
      transition: all 0.3s; cursor: none;
    }
    .btn-ghost:hover {
      background: rgba(0,245,255,0.06);
      border-color: var(--cyan);
      color: var(--cyan);
    }

    /* Scroll indicator */
    .scroll-hint {
      position: absolute; bottom: 40px;
      display: flex; flex-direction: column; align-items: center; gap: 10px;
      font-size: 0.72rem; letter-spacing: 0.25em;
      color: rgba(255,255,255,0.2);
      animation: bobDown 2s ease-in-out infinite;
    }
    @keyframes bobDown {
      0%,100% { transform: translateY(0); }
      50% { transform: translateY(8px); }
    }
    .scroll-arrow {
      width: 20px; height: 20px;
      border-right: 1px solid rgba(0,245,255,0.3);
      border-bottom: 1px solid rgba(0,245,255,0.3);
      transform: rotate(45deg);
    }

    /* ═══════════════════════════════════════
       SECTION 2 — STORY (SCROLLYTELLING)
    ═══════════════════════════════════════ */
    .story-section {
      padding-top: 80px; padding-bottom: 80px;
    }

    .story-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 2px;
      width: 100%; max-width: 1000px;
    }

    .story-step {
      position: relative;
      padding: 50px 36px;
      text-align: center;
      border: 1px solid transparent;
      border-radius: 2px;
      background: var(--glass);
      backdrop-filter: blur(10px);
      transition: border-color 0.4s, background 0.4s;
      /* Faux 3D on hover */
      transform-style: preserve-3d;
      perspective: 800px;
    }
    .story-step:hover {
      border-color: var(--border);
      background: rgba(0,245,255,0.03);
    }

    .step-num {
      font-family: 'Orbitron', sans-serif;
      font-size: 4.5rem; font-weight: 900;
      color: rgba(0,245,255,0.06);
      position: absolute; top: 20px; right: 24px;
      line-height: 1;
      transition: color 0.4s;
    }
    .story-step:hover .step-num { color: rgba(0,245,255,0.12); }

    .step-icon {
      width: 56px; height: 56px;
      border-radius: 16px;
      background: var(--glass);
      border: 1px solid var(--border);
      display: flex; align-items: center; justify-content: center;
      margin: 0 auto 24px;
      font-size: 1.5rem;
      position: relative; z-index: 1;
    }

    .step-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 0.9rem; font-weight: 700;
      letter-spacing: 0.12em;
      color: white; margin-bottom: 12px;
    }

    .step-desc {
      font-size: 0.9rem; line-height: 1.65;
      color: rgba(255,255,255,0.35);
      font-weight: 400;
    }

    .step-divider {
      display: flex; align-items: center; justify-content: center;
      font-size: 1.2rem; color: rgba(0,245,255,0.2);
      position: absolute; right: -20px; top: 50%;
      transform: translateY(-50%);
      z-index: 2;
    }

    /* ═══════════════════════════════════════
       SECTION 3 — UPLOAD
    ═══════════════════════════════════════ */
    .upload-section { gap: 40px; }

    .section-label {
      font-family: 'Orbitron', sans-serif;
      font-size: 0.7rem; font-weight: 700;
      letter-spacing: 0.4em;
      color: rgba(0,245,255,0.4);
      text-transform: uppercase;
      text-align: center;
      margin-bottom: 8px;
    }
    .section-heading {
      font-family: 'Orbitron', sans-serif;
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 900;
      text-align: center;
      background: linear-gradient(135deg, white, rgba(255,255,255,0.5));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      line-height: 1.1;
    }

    .upload-card {
      width: 100%; max-width: 520px;
      padding: 50px 44px;
      border-radius: 28px;
      background: var(--glass);
      border: 1px solid var(--border);
      backdrop-filter: blur(30px);
      position: relative; overflow: hidden;
      /* Faux 3D */
      transform-style: preserve-3d;
      perspective: 1000px;
    }
    .upload-card::before {
      content: '';
      position: absolute; top: 0; left: 0; right: 0; height: 1px;
      background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    }
    .upload-card::after {
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 50% -20%, rgba(0,245,255,0.05), transparent 60%);
      pointer-events: none;
    }

    .drop-zone {
      border: 2px dashed rgba(0,245,255,0.2);
      border-radius: 16px;
      padding: 44px 20px;
      text-align: center;
      transition: all 0.3s;
      cursor: none; position: relative;
    }
    .drop-zone:hover, .drop-zone.drag-over {
      border-color: rgba(0,245,255,0.6);
      background: rgba(0,245,255,0.03);
    }

    .drop-icon {
      font-size: 2.8rem; margin-bottom: 16px;
      display: block;
    }
    .drop-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 0.9rem; font-weight: 700;
      color: rgba(255,255,255,0.7);
      margin-bottom: 8px;
    }
    .drop-hint {
      font-size: 0.82rem;
      color: rgba(255,255,255,0.25);
    }
    .drop-hint span {
      color: var(--cyan); cursor: none;
    }

    input[type="file"] {
      position: absolute; inset: 0;
      opacity: 0; cursor: none;
      width: 100%; height: 100%;
    }
    #file-name {
      margin-top: 14px;
      font-size: 0.82rem;
      color: rgba(0,245,255,0.6);
      text-align: center;
      min-height: 20px;
    }

    .submit-btn {
      width: 100%;
      margin-top: 24px;
      padding: 17px;
      border-radius: 14px;
      border: none; cursor: none;
      font-family: 'Orbitron', sans-serif;
      font-size: 0.82rem; font-weight: 700;
      letter-spacing: 0.2em;
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      color: white;
      position: relative; overflow: hidden;
      transition: transform 0.3s, box-shadow 0.3s;
      box-shadow: 0 8px 32px rgba(0,245,255,0.2);
    }
    .submit-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 16px 48px rgba(0,245,255,0.35);
    }
    .submit-btn::before {
      content: '';
      position: absolute; top: 0; left: -100%;
      width: 100%; height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
      transition: left 0.6s;
    }
    .submit-btn:hover::before { left: 100%; }

    /* ═══════════════════════════════════════
       SECTION 4 — SAMPLE DEMO
    ═══════════════════════════════════════ */
    .demo-section { gap: 40px; }

    .digits-label {
      font-size: 0.82rem;
      color: rgba(255,255,255,0.25);
      text-align: center;
      margin-top: -24px;
      letter-spacing: 0.05em;
    }

    .digits-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 14px;
      max-width: 600px; width: 100%;
    }

    .digit-btn {
      aspect-ratio: 1;
      border-radius: 18px;
      background: var(--glass);
      border: 1px solid rgba(255,255,255,0.07);
      display: flex; align-items: center; justify-content: center;
      font-family: 'Orbitron', sans-serif;
      font-size: 1.8rem; font-weight: 900;
      color: rgba(255,255,255,0.5);
      text-decoration: none;
      transition: all 0.3s;
      position: relative; overflow: hidden;
      /* 3D tilt target */
      transform-style: preserve-3d;
      will-change: transform;
    }
    .digit-btn::before {
      content: '';
      position: absolute; inset: 0;
      background: linear-gradient(135deg, rgba(0,245,255,0.08), rgba(124,58,237,0.08));
      opacity: 0; transition: opacity 0.3s;
    }
    .digit-btn:hover {
      border-color: rgba(0,245,255,0.4);
      color: var(--cyan);
      transform: translateY(-6px) scale(1.06);
      box-shadow: 0 20px 48px rgba(0,245,255,0.2), 0 0 0 1px rgba(0,245,255,0.1);
    }
    .digit-btn:hover::before { opacity: 1; }

    /* ═══════════════════════════════════════
       FOOTER
    ═══════════════════════════════════════ */
    footer {
      position: relative; z-index: 10;
      padding: 50px 48px;
      border-top: 1px solid rgba(255,255,255,0.04);
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 20px;
    }
    .footer-logo {
      font-family: 'Orbitron', sans-serif;
      font-size: 0.85rem; font-weight: 900;
      letter-spacing: 0.15em;
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .footer-copy {
      font-size: 0.78rem;
      color: rgba(255,255,255,0.2);
      letter-spacing: 0.08em;
    }

    /* ═══════════════════════════════════════
       SCROLL-TRIGGERED REVEALS
    ═══════════════════════════════════════ */
    .reveal {
      opacity: 0;
      transform: translateY(50px);
      transition: opacity 0.9s cubic-bezier(.22,.68,0,1), transform 0.9s cubic-bezier(.22,.68,0,1);
    }
    .reveal.delay-1 { transition-delay: 0.1s; }
    .reveal.delay-2 { transition-delay: 0.2s; }
    .reveal.delay-3 { transition-delay: 0.3s; }
    .reveal.delay-4 { transition-delay: 0.4s; }

    .reveal.active {
      opacity: 1;
      transform: translateY(0);
    }

    .reveal-left {
      opacity: 0;
      transform: translateX(-60px);
      transition: opacity 0.9s cubic-bezier(.22,.68,0,1), transform 0.9s cubic-bezier(.22,.68,0,1);
    }
    .reveal-left.active { opacity: 1; transform: translateX(0); }
    .reveal-right {
      opacity: 0;
      transform: translateX(60px);
      transition: opacity 0.9s cubic-bezier(.22,.68,0,1), transform 0.9s cubic-bezier(.22,.68,0,1);
    }
    .reveal-right.active { opacity: 1; transform: translateX(0); }

    /* ═══════════════════════════════════════
       GLITCH EFFECT (hero title on load)
    ═══════════════════════════════════════ */
    @keyframes glitchIn {
      0%   { clip-path: inset(80% 0 0 0); transform: skewX(-8deg) translateX(-10px); }
      20%  { clip-path: inset(0 0 80% 0); transform: skewX(4deg) translateX(6px); }
      40%  { clip-path: inset(40% 0 30% 0); transform: skewX(0deg) translateX(-4px); }
      60%  { clip-path: inset(0 0 0 0); transform: skewX(2deg); }
      80%  { clip-path: inset(0 0 0 0); transform: skewX(-1deg); }
      100% { clip-path: inset(0 0 0 0); transform: skewX(0deg) translateX(0); }
    }
    .hero-title .line-1 { animation: glitchIn 1.2s 0.1s both; }
    .hero-title .line-2 { animation: glitchIn 1.2s 0.35s both; }
    .hero-eyebrow { animation: fadeSlideUp 1s 0.6s both; }
    .hero-sub     { animation: fadeSlideUp 1s 0.8s both; }
    .hero-cta     { animation: fadeSlideUp 1s 1s both; }

    @keyframes fadeSlideUp {
      from { opacity: 0; transform: translateY(30px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* Responsive */
    @media (max-width: 700px) {
      .story-grid { grid-template-columns: 1fr; }
      .digits-grid { grid-template-columns: repeat(5, 1fr); gap: 10px; }
      .upload-card { padding: 32px 24px; }
      nav { padding: 16px 20px; }
      footer { padding: 40px 24px; }
    }
  </style>
</head>
<body>

  <!-- Custom cursor -->
  <div class="cursor" id="cursorDot" style="position:fixed;z-index:9999;pointer-events:none;">
    <div class="cursor-dot" id="dot"></div>
  </div>
  <div class="cursor" id="cursorRing" style="position:fixed;z-index:9998;pointer-events:none;">
    <div class="cursor-ring" id="ring"></div>
  </div>

  <!-- Scanlines -->
  <div class="scanlines"></div>

  <!-- Parallax star layers -->
  <div class="parallax-layer" id="layer1" data-speed="0.02"></div>
  <div class="parallax-layer" id="layer2" data-speed="0.05"></div>
  <div class="parallax-layer" id="layer3" data-speed="0.1"></div>

  <!-- Nebula glow blobs -->
  <div class="nebula" style="width:600px;height:400px;left:-100px;top:10%;
    background:radial-gradient(circle,rgba(0,245,255,0.06),transparent 70%);"></div>
  <div class="nebula" style="width:700px;height:500px;right:-150px;top:40%;
    background:radial-gradient(circle,rgba(124,58,237,0.07),transparent 70%);
    animation-delay:-10s;animation-duration:25s;"></div>
  <div class="nebula" style="width:500px;height:500px;left:30%;bottom:5%;
    background:radial-gradient(circle,rgba(245,158,11,0.04),transparent 70%);
    animation-delay:-5s;animation-duration:30s;"></div>

  <!-- ────────────────────────────────── NAV -->
  <nav>
    <span class="nav-logo">MUNAF·AI</span>
    <ul class="nav-links">
      <li><a href="#hero">Home</a></li>
      <li><a href="#upload">Upload</a></li>
      <li><a href="#demo">Demo</a></li>
    </ul>
  </nav>

  <!-- ────────────────────────────────── SECTION 1: HERO -->
  <section class="hero" id="hero">

    <div class="hero-eyebrow">Neural Voice Recognition System</div>

    <h1 class="hero-title">
      <span class="line-1">HEAR</span>
      <span class="line-2">THE AI</span>
      <span class="line-3">Transcending Sound Into Intelligence</span>
    </h1>

    <p class="hero-sub">
      Upload any spoken digit and watch our deep learning model decode it in
      milliseconds — with confidence scores and waveform visualization.
    </p>

    <div class="hero-cta">
      <a href="#upload" class="btn-primary">Start Predicting</a>
      <a href="#demo" class="btn-ghost">Try Live Demo</a>
    </div>

    <div class="scroll-hint">
      <span>SCROLL</span>
      <div class="scroll-arrow"></div>
    </div>
  </section>

  <!-- ────────────────────────────────── SECTION 2: SCROLLYTELLING -->
  <section class="story-section" id="how">
    <div class="section-label reveal">How It Works</div>
    <h2 class="section-heading reveal delay-1" style="margin-bottom:60px">
      Three Steps to<br>Sonic Intelligence
    </h2>

    <div class="story-grid">
      <div class="story-step reveal-left">
        <span class="step-num">01</span>
        <div class="step-icon">🎙️</div>
        <div class="step-title">Capture</div>
        <p class="step-desc">
          Record or upload a .wav file of any spoken digit from 0 through 9.
          The system accepts any sample rate.
        </p>
      </div>

      <div class="story-step reveal delay-2">
        <div style="position:absolute;top:50%;left:-1px;width:2px;height:60px;
          background:linear-gradient(var(--cyan),var(--violet));transform:translateY(-50%);"></div>
        <div style="position:absolute;top:50%;right:-1px;width:2px;height:60px;
          background:linear-gradient(var(--cyan),var(--violet));transform:translateY(-50%);"></div>
        <span class="step-num">02</span>
        <div class="step-icon">🧠</div>
        <div class="step-title">Analyse</div>
        <p class="step-desc">
          Mel-spectrogram features are extracted and fed to a TensorFlow
          CNN trained on thousands of audio samples.
        </p>
      </div>

      <div class="story-step reveal-right">
        <span class="step-num">03</span>
        <div class="step-icon">⚡</div>
        <div class="step-title">Predict</div>
        <p class="step-desc">
          Get the top-3 digit predictions with confidence percentages and
          an interactive waveform in real-time.
        </p>
      </div>
    </div>
  </section>

  <!-- ────────────────────────────────── SECTION 3: UPLOAD -->
  <section class="upload-section" id="upload">
    <div class="section-label reveal">Upload Your Audio</div>
    <h2 class="section-heading reveal delay-1">Drop It &amp; Discover</h2>
    <p class="reveal delay-2" style="color:rgba(255,255,255,0.3);font-size:0.95rem;
      text-align:center;margin-top:-20px;max-width:360px;line-height:1.7;">
      Supports .wav files. Our model predicts the spoken digit with stunning accuracy.
    </p>

    <div class="upload-card reveal delay-3" id="uploadCard">
      <form action="/predict" method="post" enctype="multipart/form-data">
        <div class="drop-zone" id="dropZone">
          <input type="file" name="file" id="fileInput" accept=".wav,audio/*" required>
          <span class="drop-icon">🎵</span>
          <div class="drop-title">Drop your audio here</div>
          <div class="drop-hint">or <span>browse files</span> · WAV format</div>
        </div>
        <div id="file-name"></div>
        <button type="submit" class="submit-btn">⚡ &nbsp; ANALYSE AUDIO</button>
      </form>
    </div>
  </section>

  <!-- ────────────────────────────────── SECTION 4: DEMO -->
  <section class="demo-section" id="demo">
    <div class="section-label reveal">Instant Demo</div>
    <h2 class="section-heading reveal delay-1">Tap Any Digit</h2>
    <p class="digits-label reveal delay-2">
      Test with pre-loaded sample audio — no upload needed
    </p>

    <div class="digits-grid">
      <a href="/sample/0" class="digit-btn reveal delay-1">0</a>
      <a href="/sample/1" class="digit-btn reveal delay-2">1</a>
      <a href="/sample/2" class="digit-btn reveal delay-2">2</a>
      <a href="/sample/3" class="digit-btn reveal delay-3">3</a>
      <a href="/sample/4" class="digit-btn reveal delay-3">4</a>
      <a href="/sample/5" class="digit-btn reveal delay-3">5</a>
      <a href="/sample/6" class="digit-btn reveal delay-4">6</a>
      <a href="/sample/7" class="digit-btn reveal delay-4">7</a>
      <a href="/sample/8" class="digit-btn reveal delay-4">8</a>
      <a href="/sample/9" class="digit-btn reveal delay-4">9</a>
    </div>
  </section>

  <!-- ────────────────────────────────── FOOTER -->
  <footer>
    <span class="footer-logo">MUNAF·AI</span>
    <span class="footer-copy">AI Voice Recognition System · Built with FastAPI &amp; TensorFlow</span>
  </footer>

  <script>
  (function() {
    /* ── STARS ── */
    const speeds = [0.02, 0.05, 0.1];
    const counts = [80, 50, 30];
    ['layer1','layer2','layer3'].forEach((id, li) => {
      const layer = document.getElementById(id);
      for (let i = 0; i < counts[li]; i++) {
        const s = document.createElement('div');
        s.className = 'star';
        const size = Math.random() * (li * 1.5 + 0.8) + 0.4;
        s.style.cssText = `
          left:${Math.random()*100}%;
          top:${Math.random()*100}%;
          width:${size}px;height:${size}px;
          opacity:${(Math.random()*0.6+0.1).toFixed(2)};
          --d:${(Math.random()*3+2).toFixed(1)}s;
          animation-delay:${(Math.random()*5).toFixed(1)}s;
        `;
        layer.appendChild(s);
      }
    });

    /* ── PARALLAX ON SCROLL ── */
    const layers = document.querySelectorAll('.parallax-layer');
    window.addEventListener('scroll', () => {
      const y = window.scrollY;
      layers.forEach(l => {
        const speed = parseFloat(l.dataset.speed);
        l.style.transform = `translateY(${y * speed}px)`;
      });
    }, { passive: true });

    /* ── CUSTOM CURSOR ── */
    const dot  = document.getElementById('cursorDot');
    const ring = document.getElementById('cursorRing');
    let mx = 0, my = 0, rx = 0, ry = 0;
    document.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });
    function animCursor() {
      dot.style.left  = mx + 'px';
      dot.style.top   = my + 'px';
      rx += (mx - rx) * 0.14;
      ry += (my - ry) * 0.14;
      ring.style.left = rx + 'px';
      ring.style.top  = ry + 'px';
      requestAnimationFrame(animCursor);
    }
    animCursor();

    // Cursor grow on interactive elements
    document.querySelectorAll('a,button,input,.digit-btn,.drop-zone').forEach(el => {
      el.addEventListener('mouseenter', () => {
        document.getElementById('ring').style.cssText +=
          'width:60px;height:60px;opacity:0.5;';
      });
      el.addEventListener('mouseleave', () => {
        document.getElementById('ring').style.cssText =
          document.getElementById('ring').style.cssText
            .replace(/width:[^;]+;/,'').replace(/height:[^;]+;/,'').replace(/opacity:[^;]+;/,'');
      });
    });

    /* ── FAUX 3D TILT on upload card ── */
    const card = document.getElementById('uploadCard');
    if (card) {
      card.addEventListener('mousemove', e => {
        const rect = card.getBoundingClientRect();
        const cx = rect.left + rect.width  / 2;
        const cy = rect.top  + rect.height / 2;
        const dx = (e.clientX - cx) / rect.width;
        const dy = (e.clientY - cy) / rect.height;
        card.style.transform =
          `perspective(1000px) rotateY(${dx * 10}deg) rotateX(${-dy * 10}deg) scale(1.02)`;
      });
      card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1000px) rotateY(0) rotateX(0) scale(1)';
        card.style.transition = 'transform 0.6s cubic-bezier(.22,.68,0,1)';
      });
      card.addEventListener('mouseenter', () => {
        card.style.transition = 'transform 0.1s';
      });
    }

    /* ── FAUX 3D TILT on digit buttons ── */
    document.querySelectorAll('.digit-btn').forEach(btn => {
      btn.addEventListener('mousemove', e => {
        const r = btn.getBoundingClientRect();
        const dx = (e.clientX - r.left - r.width /2) / r.width;
        const dy = (e.clientY - r.top  - r.height/2) / r.height;
        btn.style.transform =
          `perspective(400px) rotateY(${dx*20}deg) rotateX(${-dy*20}deg) translateY(-6px) scale(1.08)`;
      });
      btn.addEventListener('mouseleave', () => {
        btn.style.transform = '';
        btn.style.transition = 'transform 0.5s cubic-bezier(.22,.68,0,1)';
      });
      btn.addEventListener('mouseenter', () => {
        btn.style.transition = 'transform 0.08s';
      });
    });

    /* ── SCROLL-TRIGGERED REVEAL (IntersectionObserver) ── */
    const revealEls = document.querySelectorAll('.reveal,.reveal-left,.reveal-right');
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    revealEls.forEach(el => observer.observe(el));

    /* ── FILE INPUT display ── */
    document.getElementById('fileInput').addEventListener('change', function() {
      const name = this.files[0] ? this.files[0].name : '';
      document.getElementById('file-name').textContent = name
        ? '📁 ' + name : '';
    });

    /* ── DRAG & DROP visual ── */
    const dz = document.getElementById('dropZone');
    if (dz) {
      ['dragenter','dragover'].forEach(ev =>
        dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag-over'); }));
      ['dragleave','drop'].forEach(ev =>
        dz.addEventListener(ev, () => dz.classList.remove('drag-over')));
    }

    /* ── STORY STEPS parallax on scroll ── */
    const steps = document.querySelectorAll('.story-step');
    window.addEventListener('scroll', () => {
      steps.forEach((step, i) => {
        const rect = step.getBoundingClientRect();
        const center = window.innerHeight / 2;
        const dist = (rect.top + rect.height/2 - center) / center;
        const ty = dist * -12;
        if (Math.abs(dist) < 1.5) {
          step.style.transform = `translateY(${ty}px)`;
        }
      });
    }, { passive: true });

  })();
  </script>
</body>
</html>"""


# ---------- ROUTES ----------
@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return generate_result_html(file_path, f"uploads/{file.filename}")

@app.get("/sample/{digit}", response_class=HTMLResponse)
def sample_test(digit: int):
    file_path = f"my_samples/{digit}.wav"
    return generate_result_html(file_path, f"samples/{digit}.wav")

# ---------- RUN ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
