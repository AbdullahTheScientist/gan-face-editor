<div align="center">

<br/>

<pre>
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    ██████╗  █████╗ ███╗   ██╗    ███████╗ █████╗  ██████╗║
║   ██╔════╝ ██╔══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝║
║   ██║  ███╗███████║██╔██╗ ██║    █████╗  ███████║██║     ║
║   ██║   ██║██╔══██║██║╚██╗██║    ██╔══╝  ██╔══██║██║     ║
║   ╚██████╔╝██║  ██║██║ ╚████║    ██║     ██║  ██║╚██████╗║
║    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝║
║                                                           ║
║              E D I T O R  ──  v 1 . 0 . 0                ║
╚═══════════════════════════════════════════════════════════╝
</pre>

<br/>

</div>

## 🎥 Click below to Watch Demo

<div align="center">
  <a href="https://www.youtube.com/watch?v=fXN4hCyXGgQ">
    <img src="https://img.youtube.com/vi/fXN4hCyXGgQ/0.jpg" width="600"/>
  </a>
</div>
**A FastAPI-powered web interface for real-time GAN facial attribute manipulation**  
*Built on InterFaceGAN · StyleGAN · PGGAN*



[![Python](https://img.shields.io/badge/Python-3.8%2B-c8f135?style=flat-square&logo=python&logoColor=black)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-6af0d8?style=flat-square&logo=fastapi&logoColor=black)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ff5e6c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-f0b16a?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-c8f135?style=flat-square)]()

<br/>

</div>

---

## ✦ Overview

**GAN Face Editor** is a full-stack web application that exposes the power of [InterFaceGAN](https://github.com/genforce/interfacegan) through a sleek browser UI. Load a pre-trained GAN, sample random face images, then push and pull facial attributes — age, gender, smile, pose, eyeglasses — in real-time using intuitive sliders.

```
Browser UI  ──►  FastAPI Backend  ──►  GAN Generator  ──►  Boundary Vectors
    ▲                                                              │
    └─────────────── Base64 PNG ◄────────────────────────────────┘
```

> **No Jupyter. No Google Colab.** Just a clean web UI you can run anywhere.

---

## ✦ Features

| Feature | Description |
|---|---|
| 🧠 **Multi-model Support** | Switch between `stylegan_ffhq`, `stylegan_celebahq`, and `pggan_celebahq` |
| 🔀 **W / Z Latent Space** | Toggle between Z-space and disentangled W-space for StyleGAN |
| 🎲 **Seeded Sampling** | Reproducible face generation with a numeric seed (0–1000) |
| 🖼️ **Batch Synthesis** | Generate 1–8 faces side-by-side in a single pass |
| 🎛️ **5 Attribute Sliders** | Age · Eyeglasses · Gender · Pose · Smile (range: −3.0 → +3.0) |
| ⚡ **Auto-apply on Release** | Edits fire automatically when you release a slider |
| 📐 **Configurable Viz Size** | Choose output resolution from 64 px to 512 px |
| 📋 **Live Activity Log** | Timestamped log of every API call and result |
| 🌐 **Zero-reload Architecture** | Everything happens over REST — no page refreshes |

---

## ✦ Project Structure

```
gan-face-editor/
│
├── main.py                   ← FastAPI application (all endpoints)
├── index.html                ← Single-file frontend (HTML + CSS + JS)
├── requirements.txt          ← Python dependencies
│
├── models/
│   ├── model_settings.py     ← MODEL_POOL registry
│   ├── pggan_generator.py    ← PGGAN wrapper
│   └── stylegan_generator.py ← StyleGAN wrapper
│
├── boundaries/               ← Pre-trained attribute boundary vectors (.npy)
│   ├── stylegan_ffhq_age_w_boundary.npy
│   ├── stylegan_ffhq_eyeglasses_w_boundary.npy
│   ├── stylegan_ffhq_gender_w_boundary.npy
│   ├── stylegan_ffhq_pose_w_boundary.npy
│   ├── stylegan_ffhq_smile_w_boundary.npy
│   └── ...                   ← (same pattern for other models / Z-space)
│
└── utils/
    └── manipulator.py        ← linear_interpolate utility
```

---

## ✦ Installation

### 1 · Clone the repository

```bash
git clone https://github.com/your-username/gan-face-editor.git
cd gan-face-editor
```

### 2 · Create a virtual environment *(recommended)*

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt contents</b></summary>

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart
numpy
opencv-python-headless
Pillow
torch
```

</details>

### 4 · Download pre-trained models & boundaries

Follow the [InterFaceGAN instructions](https://github.com/genforce/interfacegan#pretrained-models) to download:

- Pre-trained GAN weights → place in `models/pretrain/`
- Boundary vectors → place in `boundaries/`

---

## ✦ Running the App

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser at:

```
http://localhost:8000
```

> The root `/` route serves `index.html` automatically. You can also open `index.html` directly as a local file and point the **API Endpoint** field at any running server.

---

## ✦ API Reference

All endpoints accept and return JSON. Base URL: `http://localhost:8000`

### `GET /status`
Returns current server state.

```json
{
  "model_loaded": true,
  "model_name": "stylegan_ffhq",
  "latent_space_type": "W",
  "has_samples": true
}
```

---

### `GET /models`
Lists all available models and latent space options.

```json
{
  "models": ["pggan_celebahq", "stylegan_celebahq", "stylegan_ffhq"],
  "latent_space_types": ["Z", "W"]
}
```

---

### `POST /load_model`
Loads a GAN model and its boundary vectors into memory.

**Request body:**
```json
{
  "model_name": "stylegan_ffhq",
  "latent_space_type": "W"
}
```

**Response:**
```json
{
  "status": "ok",
  "model": "stylegan_ffhq",
  "latent_space": "W"
}
```

---

### `POST /sample`
Samples random latent codes and synthesizes face images.

**Request body:**
```json
{
  "num_samples": 4,
  "noise_seed": 42
}
```

**Response:**
```json
{
  "status": "ok",
  "image": "<base64-encoded PNG>",
  "num_samples": 4
}
```

---

### `POST /edit`
Applies attribute offsets to the cached latent codes and returns edited faces.

**Request body:**
```json
{
  "age":        1.5,
  "eyeglasses": 0.0,
  "gender":    -1.0,
  "pose":       0.5,
  "smile":      2.0,
  "viz_size":   256
}
```

**Response:**
```json
{
  "status": "ok",
  "image": "<base64-encoded PNG>"
}
```

> All attribute values are clamped to **[−3.0, +3.0]**. `viz_size` must be between **64** and **512**.

---

## ✦ Workflow

```
1. Open the UI  →  2. Set API URL  →  3. Load Model
       ↓
4. Set num_samples + seed  →  5. Generate Samples
       ↓
6. Adjust attribute sliders  →  7. View edited faces side-by-side
       ↓
8. Change seed / model / latent space  →  repeat
```

---

## ✦ Supported Models

| Model | GAN Type | Resolution | Notes |
|---|---|---|---|
| `stylegan_ffhq` | StyleGAN | 1024 × 1024 | FFHQ dataset · best quality |
| `stylegan_celebahq` | StyleGAN | 1024 × 1024 | CelebA-HQ dataset |
| `pggan_celebahq` | PGGAN | 1024 × 1024 | CelebA-HQ dataset · Z-space only |

> W-space manipulation is only available for StyleGAN models. PGGAN automatically falls back to Z-space.

---

## ✦ Attribute Ranges

| Attribute | Min | Max | Effect at +3 | Effect at −3 |
|---|---|---|---|---|
| `age` | −3.0 | +3.0 | Older | Younger |
| `eyeglasses` | −3.0 | +3.0 | With glasses | Without glasses |
| `gender` | −3.0 | +3.0 | More masculine | More feminine |
| `pose` | −3.0 | +3.0 | Turned right | Turned left |
| `smile` | −3.0 | +3.0 | Smiling | Neutral / frowning |

---

## ✦ Troubleshooting

**`Model not loaded` error**  
→ Make sure you've clicked **Load Model** before sampling or editing.

**`No latent codes` error**  
→ Click **Generate Samples** before applying edits.

**Boundary `.npy` file not found**  
→ Confirm your `boundaries/` directory contains the correct files for the selected model and latent space type. W-space files end in `_w_boundary.npy`.

**CORS error in browser**  
→ The FastAPI backend has CORS enabled for all origins by default. If you're behind a reverse proxy, ensure it forwards the `Origin` header.

**Slow inference**  
→ Enable GPU support by ensuring PyTorch detects CUDA: `python -c "import torch; print(torch.cuda.is_available())"`. The generators automatically use the available device.

---

## ✦ Tech Stack

```
Backend          Frontend
────────         ────────
FastAPI          Vanilla HTML / CSS / JS
Uvicorn          Syne + DM Mono (Google Fonts)
PyTorch          CSS custom properties
NumPy            Fetch API (no framework)
OpenCV           Base64 image rendering
Pillow
```

---

## ✦ Credits & References

- **InterFaceGAN** — [Shen et al., CVPR 2020](https://arxiv.org/abs/1907.10786) · [GitHub](https://github.com/genforce/interfacegan)
- **StyleGAN** — [Karras et al., CVPR 2019](https://arxiv.org/abs/1812.04948)
- **PGGAN** — [Karras et al., ICLR 2018](https://arxiv.org/abs/1710.10196)
- **FFHQ Dataset** — [Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset)
- **CelebA-HQ Dataset** — [Liu et al.](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---

## ✦ License

```
MIT License — free to use, modify, and distribute.
See LICENSE for full terms.
```

---

<div align="center">

*Built with ♥ using FastAPI + InterFaceGAN*

</div>
