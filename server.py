"""
Skeuomorphic Harmonizer — DiffSinger Cloud API
------------------------------------------------
Wraps OpenVPI's DiffSingerMiniEngine in a small FastAPI service so the
browser-based Harmony Deck mixer can request rendered SATB parts over HTTP.

This file assumes you have already run the setup in DEPLOY.md:
  - cloned https://github.com/openvpi/DiffSingerMiniEngine into this directory
  - placed an acoustic model checkpoint in assets/acoustic/
  - placed a matching NSF-HiFiGAN vocoder in assets/vocoder/
  - confirmed both paths in configs/default.yaml

Run with:
  python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
"""

import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Native MiniEngine inference call — comes from the cloned OpenVPI repo,
# not from this file. If your checkout exposes a different entrypoint,
# adjust this import to match (e.g. `from inference import DiffSingerPipeline`).
from synthesis import DiffSingerPipeline

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Skeuomorphic Harmonizer AI Engine")

# The mixer UI is a static HTML file that may be opened from a completely
# different origin (local file, a teacher's own site, etc). Loosen CORS for
# development; lock allow_origins down to your real app's origin before
# putting this anywhere production-facing — wide-open "*" plus credentials
# is a real attack surface, not a formality.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/download", StaticFiles(directory=OUTPUT_DIR), name="download")

pipeline = None


@app.on_event("startup")
def load_pipeline():
    """
    Loads model weights once at process start, not per-request — diffusion
    acoustic models are large enough that per-request loading would dominate
    your latency budget entirely.
    """
    global pipeline
    pipeline = DiffSingerPipeline(config_path="configs/default.yaml")


class HarmonyRequest(BaseModel):
    lyrics: str = Field(..., min_length=1, description='Syllables to sing, e.g. "Ha le lu jah"')
    notes: List[str] = Field(..., min_length=1, description='Note names per syllable, e.g. ["A4","A4","F4","G4"]')
    durations: List[float] = Field(..., min_length=1, description="Seconds per note, same length as notes")

    @field_validator("durations")
    @classmethod
    def positive_durations(cls, v):
        if any(d <= 0 for d in v):
            raise ValueError("all durations must be positive")
        return v


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/synthesize")
async def synthesize_harmony(data: HarmonyRequest):
    if len(data.notes) != len(data.durations):
        raise HTTPException(status_code=400, detail="notes and durations must be the same length")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model still loading — retry shortly")

    try:
        filename = f"{uuid.uuid4()}.wav"
        output_path = OUTPUT_DIR / filename

        pipeline.infer(
            lyrics=data.lyrics,
            notes=data.notes,
            durations=data.durations,
            output_path=str(output_path),
        )

        return {"status": "success", "audio_url": f"/download/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
