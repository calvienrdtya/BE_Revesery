"""
api/index.py — FastAPI READ-ONLY untuk deploy Vercel (serverless).

Beda dari main.py: deployment ini TIDAK melatih model dan TIDAK menerima
upload. Ia cuma menyajikan snapshot hasil pipeline yang sudah dihitung
sebelumnya (api/data/result.snapshot.json), di-bundle langsung ke deployment.

Kenapa terpisah dari main.py:
Vercel serverless function filesystem-nya ephemeral (reset tiap cold start,
gak dibagi antar instance) dan tiap invocation bisa berhenti begitu response
selesai — training model (GridSearchCV + SHAP) dan persist result.json ke
disk lokal (pola yang dipakai main.py + pipeline.py) gak reliable di sana.
Untuk backend penuh (training + upload), jalankan main.py di host yang
mendukung disk persisten, mis. Docker + volume (lihat Dockerfile) di
Railway/Render/Fly.io.

File ini sengaja TIDAK import pipeline.py/config.py/pandas/numpy/sklearn/shap
supaya bundle serverless-nya kecil & cepat cold-start — lihat api/requirements.txt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DATA_DIR = Path(__file__).parent / "data"
RESULT_SNAPSHOT_PATH = DATA_DIR / "result.snapshot.json"
RESEARCH_PATH = DATA_DIR / "research.json"


def _load_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


# Dimuat sekali per cold start, tetap di memori selama instance hangat.
_pipeline_data: dict | None = _load_json(RESULT_SNAPSHOT_PATH)
_research: dict | None = _load_json(RESEARCH_PATH)

app = FastAPI(
    title="Revesery Churn API (read-only)",
    description=(
        "Backend snapshot untuk Dashboard Analisis Customer Churn Revesery Store. "
        "Menyajikan hasil pipeline yang sudah dihitung sebelumnya — tidak mendukung "
        "upload/retrain (lihat main.py untuk backend penuh)."
    ),
    version="1.1.0-readonly",
)

# Selalu izinkan origin FE production, apa pun isi env CORS_ORIGINS di Vercel —
# supaya dashboard tidak ikut ke-block kalau env var lupa/belum diisi benar.
_DEFAULT_ORIGINS = ["https://fe-revesery.vercel.app"]
_env_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
_allow_origins = _env_origins + [o for o in _DEFAULT_ORIGINS if o not in _env_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,  # read-only publik, tak ada auth cookie — aman dipasangkan wildcard.
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _get_data() -> dict:
    if _pipeline_data is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Snapshot data tidak tersedia. Deployment read-only ini butuh "
                "api/data/result.snapshot.json ter-bundle saat deploy."
            ),
        )
    return _pipeline_data


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — ops
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "mode": "read-only"}


@app.get("/api/status")
async def status():
    return {
        "ready": _pipeline_data is not None,
        "running": False,
        "error": None,
        "no_data_yet": _pipeline_data is None,
        "mode": "read-only-snapshot",
    }


@app.get("/api/research")
async def research():
    if _research is None:
        raise HTTPException(status_code=404, detail="Metadata riset tidak tersedia.")
    return JSONResponse(_research)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — data pipeline (snapshot)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
async def metrics():
    return JSONResponse(_get_data()["metrics"])


@app.get("/api/dataset")
async def dataset():
    return JSONResponse(_get_data()["dataset"])


@app.get("/api/eda")
async def eda():
    return JSONResponse(_get_data()["eda"])


@app.get("/api/feature_importance")
async def feature_importance():
    return JSONResponse(_get_data()["feature_importance"])


@app.get("/api/shap")
async def shap_data():
    return JSONResponse(_get_data()["shap"])


@app.get("/api/insight")
async def insight():
    return JSONResponse(_get_data()["insight"])


@app.get("/api/retention")
async def retention():
    return JSONResponse(_get_data()["retention"])


@app.post("/api/upload")
async def upload_disabled():
    raise HTTPException(
        status_code=501,
        detail=(
            "Upload/retrain tidak didukung di deployment read-only (Vercel). "
            "Jalankan backend penuh (main.py) di host dengan disk persisten "
            "untuk mengunggah data baru."
        ),
    )
