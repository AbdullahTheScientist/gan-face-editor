import io
import base64
import numpy as np
import cv2
import PIL.Image
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.manipulator import linear_interpolate

app = FastAPI(title="GAN Face Editor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── State ────────────────────────────────────────────────────────────────────
_generator = None
_boundaries = {}
_latent_codes = None
_model_name = None
_latent_space_type = None

ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']


# ─── Helpers ──────────────────────────────────────────────────────────────────
def build_generator(model_name: str):
    gan_type = MODEL_POOL[model_name]['gan_type']
    if gan_type == 'pggan':
        return PGGANGenerator(model_name)
    elif gan_type == 'stylegan':
        return StyleGANGenerator(model_name)
    raise ValueError(f"Unknown gan_type: {gan_type}")


def sample_codes(generator, num: int, latent_space_type: str = 'Z', seed: int = 0):
    np.random.seed(seed)
    codes = generator.easy_sample(num)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
        codes = generator.get_value(generator.model.mapping(codes))
    return codes


def images_to_base64(images: np.ndarray, col: int, viz_size: int = 256) -> str:
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col
    fused = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)
    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y, x = i * viz_size, j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused[y:y + viz_size, x:x + viz_size] = image
    pil = PIL.Image.fromarray(fused)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Schemas ──────────────────────────────────────────────────────────────────
class LoadModelRequest(BaseModel):
    model_name: Literal['pggan_celebahq', 'stylegan_celebahq', 'stylegan_ffhq'] = 'stylegan_ffhq'
    latent_space_type: Literal['Z', 'W'] = 'W'


class SampleRequest(BaseModel):
    num_samples: int = Field(4, ge=1, le=8)
    noise_seed: int = Field(0, ge=0, le=1000)


class EditRequest(BaseModel):
    age: float = Field(0.0, ge=-3.0, le=3.0)
    eyeglasses: float = Field(0.0, ge=-3.0, le=3.0)
    gender: float = Field(0.0, ge=-3.0, le=3.0)
    pose: float = Field(0.0, ge=-3.0, le=3.0)
    smile: float = Field(0.0, ge=-3.0, le=3.0)
    viz_size: int = Field(256, ge=64, le=512)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()


@app.get("/models")
async def list_models():
    return {
        "models": list(MODEL_POOL.keys()),
        "latent_space_types": ["Z", "W"]
    }


@app.post("/load_model")
async def load_model(req: LoadModelRequest):
    global _generator, _boundaries, _latent_codes, _model_name, _latent_space_type
    try:
        _model_name = req.model_name
        _latent_space_type = req.latent_space_type
        _generator = build_generator(_model_name)
        _boundaries = {}
        for attr_name in ATTRS:
            boundary_name = f'{_model_name}_{attr_name}'
            if _generator.gan_type == 'stylegan' and _latent_space_type == 'W':
                path = f'boundaries/{boundary_name}_w_boundary.npy'
            else:
                path = f'boundaries/{boundary_name}_boundary.npy'
            _boundaries[attr_name] = np.load(path)
        _latent_codes = None
        return {"status": "ok", "model": _model_name, "latent_space": _latent_space_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sample")
async def sample(req: SampleRequest):
    global _latent_codes
    if _generator is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load_model first.")
    try:
        _latent_codes = sample_codes(_generator, req.num_samples, _latent_space_type, req.noise_seed)
        synthesis_kwargs = {'latent_space_type': 'W'} if (_generator.gan_type == 'stylegan' and _latent_space_type == 'W') else {}
        images = _generator.easy_synthesize(_latent_codes, **synthesis_kwargs)['image']
        img_b64 = images_to_base64(images, col=req.num_samples)
        return {"status": "ok", "image": img_b64, "num_samples": req.num_samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit")
async def edit(req: EditRequest):
    if _generator is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")
    if _latent_codes is None:
        raise HTTPException(status_code=400, detail="No latent codes. Call /sample first.")
    try:
        attr_values = {
            'age': req.age,
            'eyeglasses': req.eyeglasses,
            'gender': req.gender,
            'pose': req.pose,
            'smile': req.smile,
        }
        new_codes = _latent_codes.copy()
        for attr_name in ATTRS:
            new_codes += _boundaries[attr_name] * attr_values[attr_name]

        synthesis_kwargs = {'latent_space_type': 'W'} if (_generator.gan_type == 'stylegan' and _latent_space_type == 'W') else {}
        new_images = _generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
        num_samples = new_images.shape[0]
        img_b64 = images_to_base64(new_images, col=num_samples, viz_size=req.viz_size)
        return {"status": "ok", "image": img_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    return {
        "model_loaded": _generator is not None,
        "model_name": _model_name,
        "latent_space_type": _latent_space_type,
        "has_samples": _latent_codes is not None,
    }