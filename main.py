"""
main.py — FastAPI entry point untuk Revesery Churn Dashboard backend.

Jalankan:  uvicorn main:app --reload --port 8000
Config lewat environment / .env (lihat .env.example + config.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from pipeline import (
    DataValidationError,
    load_dataframe,
    load_result,
    run_pipeline,
    save_result,
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("revesery.api")

# ─────────────────────────────────────────────────────────────────────────────
# State global
# ─────────────────────────────────────────────────────────────────────────────
_pipeline_data: dict[str, Any] | None = None
_pipeline_running = False
_pipeline_error: str | None = None
_error_status = 500          # HTTP status untuk error terakhir (422 validasi, else 500)
_no_data_yet = True          # True sampai ada data (upload / hasil tersimpan)
_result_mtime: float | None = None  # mtime result.json yang sedang di-memori

# Metadata riset — statik, tak bergantung pipeline. Dibaca sekali.
_research: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Lock training lintas worker (single-host)
# ponytail: filesystem lock, cukup untuk single-host; naik ke Redis kalau multi-host.
# ─────────────────────────────────────────────────────────────────────────────
_LOCK_STALE_SEC = 900  # lock >15 mnt dianggap crash, boleh diambil alih


def _acquire_lock() -> bool:
    config.ensure_data_dir()
    try:
        fd = os.open(config.LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - config.LOCK_PATH.stat().st_mtime
        except OSError:
            return False
        if age > _LOCK_STALE_SEC:
            logger.warning("Lock training basi (%.0fs), diambil alih.", age)
            return True
        return False


def _release_lock() -> None:
    try:
        config.LOCK_PATH.unlink()
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
async def _run_pipeline_async(source=None):
    global _pipeline_data, _pipeline_running, _pipeline_error, _error_status
    global _no_data_yet, _result_mtime

    if not _acquire_lock():
        logger.info("Worker lain sedang training, lewati (akan sync via mtime).")
        return

    _pipeline_running = True
    _pipeline_error = None
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_pipeline, source)
        _pipeline_data = result
        _no_data_yet = False
        save_result(result)
        try:
            _result_mtime = config.RESULT_PATH.stat().st_mtime
        except OSError:
            _result_mtime = None
    except FileNotFoundError:
        # Belum ada data — bukan error, hanya belum diunggah.
        _no_data_yet = True
        _pipeline_data = None
    except DataValidationError as e:
        logger.warning("Data tidak valid: %s", e)
        _pipeline_error = str(e)
        _error_status = 422
    except Exception as e:  # noqa: BLE001
        logger.exception("Pipeline gagal")  # traceback ke log, tak hilang
        _pipeline_error = str(e)
        _error_status = 500
    finally:
        _pipeline_running = False
        _release_lock()


def _load_persisted() -> bool:
    global _pipeline_data, _no_data_yet, _result_mtime
    r = load_result()
    if r is not None:
        _pipeline_data = r
        _no_data_yet = False
        try:
            _result_mtime = config.RESULT_PATH.stat().st_mtime
        except OSError:
            _result_mtime = None
        logger.info("Hasil pipeline dimuat dari disk (tanpa retrain).")
        return True
    return False


def _load_research() -> None:
    global _research
    if config.RESEARCH_PATH.exists():
        try:
            _research = json.loads(config.RESEARCH_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Gagal baca research.json: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_research()
    if not _load_persisted():
        # Tak ada hasil tersimpan; coba latih dari xlsx yang ada (non-blocking).
        asyncio.create_task(_run_pipeline_async(None))
    yield


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Revesery Churn API",
    description="Backend REST API untuk Dashboard Analisis Customer Churn Revesery Store.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────
async def require_api_key(x_api_key: str | None = Header(default=None)):
    if config.API_KEY:
        if x_api_key != config.API_KEY:
            raise HTTPException(status_code=401, detail="API key tidak valid.")
    else:
        logger.warning("Upload tanpa proteksi API key (set API_KEY untuk production).")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _maybe_reload() -> None:
    """Sinkron multi-worker: reload result.json bila di disk lebih baru dari memori.
    ponytail: file-mtime sync; naik ke Redis kalau multi-host."""
    global _pipeline_data, _result_mtime, _no_data_yet
    try:
        m = config.RESULT_PATH.stat().st_mtime
    except OSError:
        return
    if _result_mtime is None or m > _result_mtime:
        r = load_result()
        if r is not None:
            _pipeline_data = r
            _result_mtime = m
            _no_data_yet = False
            logger.info("result.json lebih baru, di-reload.")


def _get_data() -> dict:
    _maybe_reload()
    if _pipeline_error and not _pipeline_running:
        raise HTTPException(status_code=_error_status, detail=_pipeline_error)
    if _no_data_yet and not _pipeline_running:
        raise HTTPException(
            status_code=503,
            detail="Belum ada data yang diunggah. Silakan unggah file Transaction_Revesery.xlsx.",
        )
    if _pipeline_running or _pipeline_data is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline masih berjalan, coba beberapa saat lagi.",
        )
    return _pipeline_data


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — ops
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    _maybe_reload()
    return {
        "ready": _pipeline_data is not None,
        "running": _pipeline_running,
        "error": _pipeline_error,
        "no_data_yet": _no_data_yet,
    }


@app.get("/api/research")
async def research():
    # Statik, tersedia sebelum upload — tak lewat gate _get_data().
    if _research is None:
        raise HTTPException(status_code=404, detail="Metadata riset tidak tersedia.")
    return JSONResponse(_research)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — data pipeline
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


@app.post("/api/upload", dependencies=[Depends(require_api_key)])
async def upload_excel(file: UploadFile = File(...)):
    """Upload Transaction_Revesery.xlsx → validasi → pipeline dijalankan ulang."""
    global _pipeline_error, _no_data_yet, _error_status

    if not file.filename or not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="File harus berformat .xlsx")

    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar (maks {config.MAX_UPLOAD_MB} MB)",
        )

    if _pipeline_running:
        raise HTTPException(status_code=409, detail="Pipeline sedang berjalan, tunggu sebentar.")

    # Validasi skema sinkron di trust boundary → 422 langsung (bukan 500 saat training).
    # ponytail: baca Excel 2x (validasi + pipeline); file skripsi kecil, abaikan.
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_dataframe, contents)
    except DataValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Persist file mentah supaya restart tak kehilangan data.
    config.ensure_data_dir()
    config.UPLOAD_PATH.write_bytes(contents)

    _no_data_yet = False
    _pipeline_error = None
    _error_status = 500
    asyncio.create_task(_run_pipeline_async(source=contents))
    return {"message": "Upload berhasil, pipeline sedang dijalankan.", "running": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, log_level=config.LOG_LEVEL.lower())
