"""
config.py — Konfigurasi backend dari environment variable.

Semua tuning knob production di sini (tanpa dependency tambahan, cukup os.getenv).
Baca sekali saat import; override lewat env / .env (lihat .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path


def _csv(val: str) -> list[str]:
    return [x.strip() for x in val.split(",") if x.strip()]


# ── Server ──────────────────────────────────────────────────────────────────
PORT: int = int(os.getenv("PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ── CORS ────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list[str] = _csv(
    os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173",
    )
)

# ── Auth ────────────────────────────────────────────────────────────────────
# Kosong = upload TIDAK terproteksi (dev). Diset = wajib header X-API-Key (prod).
API_KEY: str = os.getenv("API_KEY", "").strip()

# ── Storage ─────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(Path(__file__).parent / "data")))
RESULT_PATH: Path = DATA_DIR / "result.json"
UPLOAD_PATH: Path = DATA_DIR / "latest.xlsx"
LOCK_PATH: Path = DATA_DIR / ".training.lock"
RESEARCH_PATH: Path = DATA_DIR / "research.json"

# ── Upload ──────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

# ── ML knobs (kalibrasi domain) ─────────────────────────────────────────────
# Pelanggan dianggap churn bila langganan terakhir kadaluarsa > CHURN_DAYS hari
# sebelum tanggal snapshot dataset.
CHURN_DAYS: int = int(os.getenv("CHURN_DAYS", "60"))
# Sebelum tanggal ini, semua transaksi diasumsikan langganan 1 bulan.
MULTIBULAN_CUTOFF: str = os.getenv("MULTIBULAN_CUTOFF", "2026-03-07")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
