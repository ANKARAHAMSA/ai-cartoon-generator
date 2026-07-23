<div align="center">

# 🎨 TOONIFY
### Turn your photo into stunning cartoon art — powered by Stable Diffusion

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-toonify--studio.vercel.app-7c3aed?style=for-the-badge)](https://toonify-studio.vercel.app)
[![Backend](https://img.shields.io/badge/⚡_API-toonify--backend.onrender.com-10b981?style=for-the-badge)](https://toonify-backend-8mup.onrender.com/health)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

![TOONIFY Banner](https://img.shields.io/badge/Stable_Diffusion-img2img-purple?style=flat-square)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?style=flat-square&logo=vercel)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat-square)

</div>

---

## ✨ Features

- 🖼️ **Upload any photo** — portraits, selfies, group shots
- 🎨 **12 unique art styles** — Disney, Anime, Ghibli, Arcane, Pixar, Comic, Van Gogh, Simpsons, Cyberpunk, Watercolor, Claymation, Pencil Sketch
- 🎚️ **Style Strength Slider** — fine-tune from subtle (40%) to full transformation (90%)
- 👤 **IP-Adapter Face Lock** — preserve face identity across styles
- ⚖️ **Before/After Compare Mode** — drag to reveal the transformation
- 🔮 **Custom LoRA styles** — Arcane Painterly & 3D Claymation trained styles
- 🌐 **Fully deployed** — Vercel frontend + Render backend

---

## 🖥️ Live App

**Frontend:** [toonify-studio.vercel.app](https://toonify-studio.vercel.app)  
**API Health:** [toonify-backend-8mup.onrender.com/health](https://toonify-backend-8mup.onrender.com/health)  
**API Docs:** [toonify-backend-8mup.onrender.com/docs](https://toonify-backend-8mup.onrender.com/docs)

> ⚠️ The backend runs on Render's free tier — first request after inactivity may take ~50s to wake up.

---

## 🎨 Art Styles

| Style | Emoji | Description |
|-------|-------|-------------|
| Modern Disney | 🏰 | Vibrant, expressive Disney animation |
| Anime | ⛩️ | Clean Makoto Shinkai aesthetic |
| Studio Ghibli | 🌿 | Soft, painterly Miyazaki-style |
| Arcane Painterly | 🔮 | Custom LoRA — dramatic neon lighting |
| 3D Claymation | 🧱 | Custom LoRA — tactile clay texture |
| Pixar 3D | ✨ | Warm Pixar-style 3D render |
| Comic Book | 💥 | Bold Marvel comic ink outlines |
| Van Gogh | 🌻 | Post-impressionist swirling brushstrokes |
| Simpsons | 🟡 | Classic yellow skin flat cartoon |
| Cyberpunk | 🌆 | Neon-lit blade runner atmosphere |
| Watercolor | 🎨 | Soft flowing paper watercolor |
| Pencil Sketch | ✏️ | Fine graphite crosshatching |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TOONIFY Stack                         │
├─────────────┬──────────────────┬───────────────────────┤
│   FRONTEND  │     BACKEND      │    INFERENCE           │
│   Vercel    │     Render       │  HuggingFace Spaces    │
│             │                  │  (optional GPU)        │
│  React 18   │   FastAPI 0.111  │   Gradio + SD          │
│  Vite 5     │   Uvicorn        │   nitrosocke/mo-di     │
│  CSS3       │   Pillow (mock)  │   T4 GPU               │
│             │   gradio_client  │                        │
└─────────────┴──────────────────┴───────────────────────┘
         ↕ /api/* proxy              ↕ HF_SPACES_URL
    vercel.json rewrites         gradio_client.predict()
```

**Data Flow:**
```
User Photo → Vercel Frontend → Render Backend → HF Spaces (GPU) → Cartoon Image
                                     ↓ (if HF unavailable)
                               Mock PIL filter → Cartoon Image
```

---

## 🛠️ Tech Stack

### Frontend
- **React 18** + **Vite 5** — fast SPA
- **Vanilla CSS** — custom design system with glassmorphism, dark mode
- **Google Fonts** — Outfit + Inter
- **Axios** — API client with upload progress

### Backend
- **FastAPI** — async Python REST API
- **Uvicorn** — ASGI server
- **Pillow** — mock cartoon filters (no GPU)
- **diffusers** — Stable Diffusion img2img (GPU mode)
- **gradio_client** — HF Spaces proxy client

### AI/ML
- **Stable Diffusion 1.5** — base model
- **nitrosocke/mo-di-diffusion** — Disney-style fine-tune
- **IP-Adapter** — face identity preservation
- **Custom LoRA** — Arcane & Claymation styles (trained in Colab)

### Infra
- **Vercel** — frontend hosting (CDN edge)
- **Render** — backend hosting (free tier)
- **HuggingFace Spaces** — GPU inference (optional T4)
- **GitHub Actions** — auto-deploy on push

---

## 🚀 Local Development

### Prerequisites
- Node.js 20+
- Python 3.12+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/ANKARAHAMSA/TOONIFY.git
cd TOONIFY
```

### 2. Start the backend (Mock Mode — no GPU needed)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
MOCK_MODE=true uvicorn main:app --reload
# Backend running at http://localhost:8000
```

### 3. Start the frontend
```bash
cd frontend
npm install
npm run dev
# Frontend running at http://localhost:5173
```

Open [http://localhost:5173](http://localhost:5173) — the Vite proxy forwards `/api/*` to the backend automatically.

---

## 🔧 Environment Variables

### Backend (Render / local)
| Variable | Default | Description |
|----------|---------|-------------|
| `MOCK_MODE` | `false` | Use PIL filters instead of SD (no GPU) |
| `MODEL_PATH` | `./mo-di-diffusion` | HuggingFace model ID or local path |
| `USE_IP_ADAPTER` | `false` | Enable IP-Adapter face lock |
| `IP_ADAPTER_SCALE` | `0.7` | Face lock strength (0.0–1.0) |
| `HF_SPACES_URL` | `` | HF Space ID for GPU inference proxy |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `PORT` | `8000` | Server port |

### Frontend (Vercel)
| Variable | Description |
|----------|-------------|
| `VITE_API_BASE` | Override API base URL (optional) |

---

## 🌐 Deployment

### Frontend → Vercel
1. Import `ANKARAHAMSA/TOONIFY` on [vercel.com](https://vercel.com)
2. Set **Root Directory** → `frontend`
3. Click Deploy — auto-detects Vite

### Backend → Render
1. New Web Service → connect `ANKARAHAMSA/TOONIFY`
2. **Language:** Python | **Root Directory:** `backend`
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add env vars: `MOCK_MODE=true`, `PYTHON_VERSION=3.12.6`

### Real GPU Inference → HuggingFace Spaces
1. Create new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Upload `spaces/app.py`, `spaces/requirements.txt`, `spaces/README.md`
3. Set hardware to **T4 Small**
4. Add `HF_SPACES_URL=your-username/toonify` to Render env vars

---

## 🧠 Custom LoRA Training

Train your own Arcane-style LoRA using the provided Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANKARAHAMSA/TOONIFY/blob/main/notebooks/lora_training_complete.ipynb)

**Training images:** `training_images/arcane/` (30 Arcane-style 512×512 PNGs)  
**Trigger word:** `arcane_toonify style`  
**Base model:** `runwayml/stable-diffusion-v1-5`

---

## 📁 Project Structure

```
TOONIFY/
├── frontend/                  # React + Vite app
│   ├── src/
│   │   ├── App.jsx            # Main app with state management
│   │   ├── api.js             # Axios API wrapper
│   │   ├── components/
│   │   │   ├── StrengthSlider.jsx   # Style strength control
│   │   │   └── CompareSlider.jsx    # Before/After comparison
│   │   └── index.css          # Design system + animations
│   └── vercel.json            # Vercel rewrite config
│
├── backend/                   # FastAPI backend
│   ├── main.py                # API routes
│   ├── pipeline.py            # SD pipeline + 12 styles + HF client
│   ├── models.py              # Pydantic schemas
│   ├── requirements.txt       # Pinned dependencies
│   └── runtime.txt            # Python 3.12.6
│
├── spaces/                    # HuggingFace Spaces inference app
│   ├── app.py                 # Gradio SD inference app
│   ├── requirements.txt
│   └── README.md
│
├── notebooks/
│   └── lora_training_complete.ipynb  # 9-cell LoRA training
│
├── training_images/
│   └── arcane/               # 30 Arcane training images + metadata
│
└── scripts/
    └── generate_placeholders.py
```

---

## 🛣️ Roadmap

- [x] Stable Diffusion img2img pipeline
- [x] 12 art styles (7 base + 2 LoRA + 3 new)
- [x] IP-Adapter face identity preservation
- [x] Style Strength Slider
- [x] Before/After Compare Mode
- [x] Vercel + Render deployment
- [x] HuggingFace Spaces GPU inference
- [x] Custom LoRA training pipeline
- [ ] User gallery (Firebase)
- [ ] Social sharing
- [ ] Credits / monetization (Stripe)
- [ ] Mobile app (React Native)
- [ ] Video cartoonization

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using Stable Diffusion, FastAPI, and React<br>
<a href="https://toonify-studio.vercel.app">toonify-studio.vercel.app</a>
</div>
