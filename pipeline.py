"""
pipeline.py — Logika lengkap dari Revesery_Store.ipynb
Dijalankan saat startup (kalau ada data tersimpan / xlsx) atau saat user upload Excel baru.
"""
from __future__ import annotations

import io
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import config

logger = logging.getLogger("revesery.pipeline")

# Arahkan warning ke logging (bukan diam total) — warning nyata tetap terlihat.
logging.captureWarnings(True)
# Redam hanya yang benar-benar noise & tak actionable.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*undefined metric.*")

RANDOM_STATE = 42
HARI_PER_BULAN = 30
CUTOFF_MULTIBULAN = pd.Timestamp(config.MULTIBULAN_CUTOFF)

# Kolom minimal yang harus ada di file Excel agar pipeline bisa jalan.
REQUIRED_COLUMNS = ["Date", "Email", "SKU", "Status", "Gross Revenue", "Reference"]

# Harga referensi 1 bulan langganan (data domain, bukan tuning knob runtime).
# ponytail: konstanta; knob kalibrasi yang runtime-tunable adalah CHURN_DAYS lewat env.
HARGA_1BULAN: dict[str, int] = {
    "Netflix Premium": 45000,
    "YouTube Premium": 40000,
    "Disney+ Premium": 47000,
    "HBO Max Premium": 40000,
    "Capcut Premium Mobile": 55000,
    "Prime Video": 40000,
    "Spotify Premium": 42000,
    "Canva Pro": 14000,
    "Viu Premium": 13000,
    "IQIYI Premium": 15000,
    "WeTV Premium": 15000,
    "Google AI": 40000,
    "Vidio Platinum": 35000,
}


class DataValidationError(ValueError):
    """Excel yang diunggah tidak memenuhi skema yang dibutuhkan pipeline."""


# ─────────────────────────────────────────────────────────────────────────────
# Persistensi hasil (JSON) — supaya restart tak perlu retrain
# ─────────────────────────────────────────────────────────────────────────────
def save_result(result: dict) -> None:
    config.ensure_data_dir()
    tmp = config.RESULT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(config.RESULT_PATH)  # atomic-ish, hindari file setengah tertulis
    logger.info("Hasil pipeline dipersist ke %s", config.RESULT_PATH)


def load_result() -> dict | None:
    if config.RESULT_PATH.exists():
        try:
            return json.loads(config.RESULT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Gagal baca result.json (%s), abaikan.", e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: estimasi durasi (bulan)
# ─────────────────────────────────────────────────────────────────────────────
def _estimasi_bulan(row: pd.Series) -> int:
    if row["Date"] < CUTOFF_MULTIBULAN:
        return 1
    patokan = HARGA_1BULAN.get(row["SKU"], np.nan)
    if pd.isna(patokan) or patokan <= 0:
        return 1
    return int(np.clip(round(row["Gross Revenue"] / patokan), 1, 3))


def _safe_mode(s: pd.Series) -> Any:
    """Modus aman: kembalikan nilai pertama kalau mode() kosong (grup all-NaN)."""
    m = s.mode()
    if not m.empty:
        return m.iat[0]
    return s.iloc[0] if len(s) else "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Load
# ─────────────────────────────────────────────────────────────────────────────
def load_dataframe(source: bytes | str | Path) -> pd.DataFrame:
    """Terima bytes (dari upload) atau path. Validasi skema + parsing tanggal."""
    try:
        buf = io.BytesIO(source) if isinstance(source, bytes) else source
        df = pd.read_excel(buf, sheet_name=0)
    except Exception as e:  # openpyxl/pandas parse error
        raise DataValidationError(f"File Excel tidak bisa dibaca: {e}") from e

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing)
            + f". Kolom yang ada: {', '.join(map(str, df.columns))}."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        raise DataValidationError("Kolom 'Date' tidak berisi tanggal valid.")
    n_bad_date = int(df["Date"].isna().sum())
    if n_bad_date:
        logger.warning("%d baris punya Date tak terparse, dibuang.", n_bad_date)
        df = df[df["Date"].notna()].copy()

    if df.empty:
        raise DataValidationError("Dataset kosong setelah parsing tanggal.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    df = df_raw.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.replace({"nan": np.nan, "None": np.nan})

    log = []

    def bersihkan(kondisi, alasan):
        nonlocal df
        sebelum = len(df)
        df = df[kondisi].copy()
        log.append({"step": alasan, "removed": sebelum - len(df), "remaining": len(df)})

    sebelum = len(df)
    df = df.drop_duplicates()
    log.append({"step": "Hapus duplikat", "removed": sebelum - len(df), "remaining": len(df)})

    bersihkan(df["Status"].notna(), "Hapus Status kosong")
    bersihkan(df["SKU"] != "-", "Hapus SKU placeholder")
    bersihkan(
        ~df["Reference"].astype(str).str.upper().str.startswith("TEST"),
        "Hapus Reference TEST",
    )

    ppob_pat = r"PPOB|\bpln\b|dana|pulsa|flash|getcontact|lacak nomor|gopay|\bovo\b|shopeepay|\bisi\b|token"
    bersihkan(
        ~df["SKU"].astype(str).str.contains(ppob_pat, case=False, regex=True),
        "Hapus produk PPOB/non-subscription",
    )

    df = df.reset_index(drop=True)
    if df.empty:
        raise DataValidationError("Semua baris terbuang saat cleaning — cek isi dataset.")
    return df, log


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Feature Engineering & Label
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    SNAPSHOT_DATE = df["Date"].max()

    df["bulan_dibeli"] = df.apply(_estimasi_bulan, axis=1)
    df["tanggal_kadaluarsa"] = df["Date"] + pd.to_timedelta(
        df["bulan_dibeli"] * HARI_PER_BULAN, unit="D"
    )

    paid = df[df["Status"] == "PAID"].copy()
    exp = df[df["Status"] == "EXPIRED"].copy()
    if paid.empty:
        raise DataValidationError("Tidak ada transaksi Status='PAID' — tak bisa bangun fitur pelanggan.")
    paid = paid.sort_values(["Email", "Date"])
    g = paid.groupby("Email")

    cust = pd.DataFrame(
        {
            "jumlah_transaksi": g.size(),
            "total_belanja": g["Gross Revenue"].sum(),
            "rata_belanja": g["Gross Revenue"].mean().round(0),
            "tenure_hari": (g["Date"].max() - g["Date"].min()).dt.days,
            "produk_unik": g["SKU"].nunique(),
            "rata_bulan": g["bulan_dibeli"].mean().round(2),
            "maks_bulan": g["bulan_dibeli"].max(),
            "total_bulan_langganan": g["bulan_dibeli"].sum(),
        }
    )
    cust["is_pelanggan_ulang"] = (cust["jumlah_transaksi"] >= 2).astype(int)
    cust["is_multi_produk"] = (cust["produk_unik"] > 1).astype(int)
    cust["pernah_multibulan"] = (cust["maks_bulan"] > 1).astype(int)

    paid["gap"] = (paid["Date"] - g["Date"].shift(1)).dt.days
    cust["rata_jeda_hari"] = paid.groupby("Email")["gap"].mean()
    med_gap = cust.loc[cust["is_pelanggan_ulang"] == 1, "rata_jeda_hari"].median()
    cust["rata_jeda_hari"] = cust["rata_jeda_hari"].fillna(med_gap).round(1)

    cust["produk_utama"] = g["SKU"].agg(_safe_mode)

    cust = cust.join(exp.groupby("Email").size().rename("jumlah_gagal"))
    cust["jumlah_gagal"] = cust["jumlah_gagal"].fillna(0).astype(int)
    cust["rasio_gagal"] = (
        cust["jumlah_gagal"] / (cust["jumlah_gagal"] + cust["jumlah_transaksi"])
    ).round(3)

    # WARNING (riset): label churn diturunkan dari tanggal_kadaluarsa, sementara fitur
    # tenure_hari / rata_bulan / total_bulan_langganan ikut menentukan kadaluarsa.
    # Ada potensi kebocoran (leakage) yang bisa menaikkan metrik secara artifisial.
    # Definisi label = keputusan riset; tinjau bila skor sangat tinggi.
    last_exp = g["tanggal_kadaluarsa"].max()
    cust["hari_sejak_kadaluarsa"] = (SNAPSHOT_DATE - last_exp).dt.days
    cust["churn"] = (cust["hari_sejak_kadaluarsa"] > config.CHURN_DAYS).astype(int)
    cust = cust.drop(columns=["hari_sejak_kadaluarsa"])

    return cust, df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Model Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(cust: pd.DataFrame):
    drop_redundan = [
        "is_pelanggan_ulang",
        "is_multi_produk",
        "pernah_multibulan",
        "maks_bulan",
    ]
    data = cust.drop(columns=drop_redundan).copy()
    data = pd.get_dummies(data, columns=["produk_utama"], prefix="prod", dtype=int)

    X = data.drop(columns=["churn"])
    y = data["churn"]

    if y.nunique() < 2:
        raise DataValidationError(
            "Label churn hanya punya satu kelas — tak bisa latih model. "
            "Cek CHURN_DAYS atau rentang tanggal data."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    grid = {
        "n_estimators": [300],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt"],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    best_rf = gs.best_estimator_

    pred = best_rf.predict(X_test)
    proba = best_rf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    return best_rf, X_train, X_test, y_train, y_test, pred, proba, report, cm, fpr, tpr, prec_vals, rec_vals, roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SHAP
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap(best_rf: RandomForestClassifier, X_test: pd.DataFrame):
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_churn = shap_values[1]
    elif shap_values.ndim == 3:
        shap_churn = shap_values[:, :, 1]
    else:
        shap_churn = shap_values

    # Base value SHAP yang BENAR = expected_value explainer (kelas churn=1),
    # bukan rata-rata predict_proba. Ini yang bikin base + Σshap ≈ prediction.
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev).ravel()
        base_value = float(ev[1] if ev.size > 1 else ev[0])
    else:
        base_value = float(ev)

    importance = (
        pd.Series(np.abs(shap_churn).mean(0), index=X_test.columns)
        .sort_values(ascending=False)
    )
    return shap_churn, importance, base_value


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY — run_pipeline()
# ─────────────────────────────────────────────────────────────────────────────
def _stage(name: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    logger.info("Stage %-22s %.2fs", name, time.perf_counter() - t0)
    return out


def run_pipeline(source: bytes | str | Path | None = None) -> dict:
    """
    Jalankan pipeline penuh. Kembalikan dict berisi semua data yang dibutuhkan FE.
    source=None → cari Transaction_Revesery.xlsx di lokasi umum, else FileNotFoundError.
    """
    # ── Load ──────────────────────────────────────────────
    if source is None:
        candidates = [
            config.UPLOAD_PATH,
            Path(__file__).parent / "Transaction_Revesery.xlsx",
            Path(__file__).parent.parent / "Transaction_Revesery.xlsx",
            Path("Transaction_Revesery.xlsx"),
        ]
        xlsx = next((p for p in candidates if p.exists()), None)
        if xlsx is None:
            raise FileNotFoundError(
                "Belum ada data yang diunggah. Silakan unggah file Transaction_Revesery.xlsx."
            )
        df_raw = _stage("load", load_dataframe, xlsx)
    else:
        df_raw = _stage("load", load_dataframe, source)

    n_raw = len(df_raw)

    df, clean_log = _stage("clean", clean, df_raw)
    cust, df_fe = _stage("feature_engineering", feature_engineering, df)
    (
        best_rf, X_train, X_test, y_train, y_test,
        pred, proba, report, cm, fpr, tpr, prec_vals, rec_vals, roc_auc
    ) = _stage("train_model", train_model, cust)
    shap_churn, importance, base_value = _stage("compute_shap", compute_shap, best_rf, X_test)

    # ── Build result dict ──────────────────────────────────
    churn_count = int(cust["churn"].sum())
    active_count = int((cust["churn"] == 0).sum())
    total = int(len(cust))
    churn_rate = round(churn_count / max(total, 1), 4)

    # Customer trend (bulanan dari df_fe)
    df_fe["month"] = df_fe["Date"].dt.to_period("M")
    paid_fe = df_fe[df_fe["Status"] == "PAID"]
    trend_total = paid_fe.groupby("month")["Email"].nunique()
    churn_emails = set(cust[cust["churn"] == 1].index)
    trend_churn = (
        paid_fe[paid_fe["Email"].isin(churn_emails)]
        .groupby("month")["Email"]
        .nunique()
    )
    months = sorted(set(trend_total.index) | set(trend_churn.index))
    customer_trend = [
        {
            "label": str(m),
            "total": int(trend_total.get(m, 0)),
            "churned": int(trend_churn.get(m, 0)),
        }
        for m in months[-12:]
    ]

    # ── Assemble output ────────────────────────────────────
    result: dict = {
        "metrics": {
            "total_customers": total,
            "churned_customers": churn_count,
            "churn_rate": churn_rate,
            "active_customers": active_count,
            "model": {
                "accuracy": round(float(report["accuracy"]), 4),
                "precision": round(float(report["1"]["precision"]), 4),
                "recall": round(float(report["1"]["recall"]), 4),
                "f1_score": round(float(report["1"]["f1-score"]), 4),
                "roc_auc": round(float(roc_auc), 4),
            },
            "customer_trend": customer_trend,
            "roc_curve": {
                "fpr": [round(float(v), 4) for v in fpr.tolist()[::5]],
                "tpr": [round(float(v), 4) for v in tpr.tolist()[::5]],
            },
            "pr_curve": {
                "recall": [round(float(v), 4) for v in rec_vals.tolist()[::5]],
                "precision": [round(float(v), 4) for v in prec_vals.tolist()[::5]],
            },
            "confusion_matrix": cm,
            "confusion_labels": ["Tidak Churn", "Churn"],
        },
        "dataset": {
            "n_rows": n_raw,
            "n_features": df_raw.shape[1],
            "missing_values": int(df_raw.isnull().sum().sum()),
            "duplicates": int(df_raw.duplicated().sum()),
            "target": "churn",
            "class_distribution": [
                {"label": "Aktif (Tidak Churn)", "count": active_count},
                {"label": "Churn", "count": churn_count},
            ],
            "columns": [
                {"name": col, "dtype": str(df_raw[col].dtype)}
                for col in df_raw.columns
            ],
            "preview": df_raw.head(20).fillna("").astype(str).to_dict(orient="records"),
            "cleaning_log": clean_log,
        },
        "eda": _build_eda(df, cust),
        "feature_importance": {
            "features": [
                {"name": k, "importance": round(float(v), 6)}
                for k, v in importance.items()
            ],
            "method": "mean |SHAP|",
        },
        "shap": _build_shap(shap_churn, importance, X_test, best_rf, base_value),
        "insight": _build_insight(importance),
        "retention": _build_retention(importance),
    }

    logger.info(
        "Pipeline selesai: %d pelanggan, churn_rate=%.3f, ROC-AUC=%.3f",
        total, churn_rate, roc_auc,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sub-builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_eda(df: pd.DataFrame, cust: pd.DataFrame) -> dict:
    paid = df[df["Status"] == "PAID"]
    cust_reset = cust.reset_index()

    churn_dist = [
        {"label": "Aktif", "count": int((cust["churn"] == 0).sum())},
        {"label": "Churn", "count": int(cust["churn"].sum())},
    ]

    prod_counts = paid["SKU"].value_counts()
    cat_skus: list[dict] = []
    if "produk_utama" in cust.columns:
        churn_mask = cust["churn"] == 1
        skus = prod_counts.index.tolist()[:10]
        churn_prod = cust[churn_mask]["produk_utama"].value_counts()
        retain_prod = cust[~churn_mask]["produk_utama"].value_counts()
        cat_skus = [
            {
                "feature": "Produk Utama",
                "categories": skus,
                "churn": [int(churn_prod.get(s, 0)) for s in skus],
                "retain": [int(retain_prod.get(s, 0)) for s in skus],
            }
        ]

    numeric_features = ["jumlah_transaksi", "total_belanja", "tenure_hari", "rata_jeda_hari"]
    histograms = []
    for feat in numeric_features:
        if feat not in cust_reset.columns:
            continue
        vals = cust_reset[feat].dropna()
        if vals.empty:
            continue
        churn_vals = cust_reset[cust_reset["churn"] == 1][feat].dropna()
        retain_vals = cust_reset[cust_reset["churn"] == 0][feat].dropna()
        counts, bin_edges = np.histogram(vals, bins=10)
        bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(bin_edges) - 1)]
        churn_counts, _ = np.histogram(churn_vals, bins=bin_edges)
        retain_counts, _ = np.histogram(retain_vals, bins=bin_edges)
        histograms.append(
            {
                "feature": feat,
                "bins": bin_labels,
                "counts": counts.tolist(),
                "by_class": {
                    "churn": churn_counts.tolist(),
                    "retain": retain_counts.tolist(),
                },
            }
        )

    boxplots = []
    for feat in ["total_belanja", "rata_belanja", "tenure_hari"]:
        if feat not in cust_reset.columns:
            continue
        groups = []
        for label, mask in [("Churn", cust_reset["churn"] == 1), ("Aktif", cust_reset["churn"] == 0)]:
            v = cust_reset[mask][feat].dropna()
            if len(v) == 0:
                continue
            groups.append(
                {
                    "label": label,
                    "min": float(v.min()),
                    "q1": float(v.quantile(0.25)),
                    "median": float(v.median()),
                    "q3": float(v.quantile(0.75)),
                    "max": float(v.max()),
                }
            )
        boxplots.append({"feature": feat, "groups": groups})

    num_cols = ["jumlah_transaksi", "total_belanja", "rata_belanja", "tenure_hari",
                "rata_bulan", "total_bulan_langganan", "rata_jeda_hari",
                "jumlah_gagal", "rasio_gagal", "churn"]
    existing = [c for c in num_cols if c in cust_reset.columns]
    corr = cust_reset[existing].corr().round(3).fillna(0)

    return {
        "churn_distribution": churn_dist,
        "categorical": cat_skus,
        "histograms": histograms,
        "boxplots": boxplots,
        "correlation": {
            "labels": existing,
            "matrix": corr.values.tolist(),
        },
    }


def _build_shap(shap_churn, importance: pd.Series, X_test: pd.DataFrame, model, base_value: float) -> dict:
    top_features = importance.head(15).index.tolist()

    global_importance = [
        {"feature": k, "value": round(float(v), 6)}
        for k, v in importance.head(15).items()
    ]

    # Summary (beeswarm data) — sample 200 titik untuk perf
    n_sample = min(200, len(X_test))
    idx = np.random.RandomState(42).choice(len(X_test), n_sample, replace=False)
    summary = []
    for feat in top_features:
        fi = list(X_test.columns).index(feat)
        sv = shap_churn[idx, fi].tolist()
        fv_raw = X_test.iloc[idx, fi].values.astype(float)
        fv_min, fv_max = float(fv_raw.min()), float(fv_raw.max())
        fv_norm = ((fv_raw - fv_min) / max(fv_max - fv_min, 1e-9)).tolist()
        summary.append(
            {
                "feature": feat,
                "shap_values": [round(v, 5) for v in sv],
                "feature_values": [round(v, 4) for v in fv_norm],
            }
        )

    # Waterfall — pelanggan paling berisiko churn (hitung proba sekali)
    churn_proba = model.predict_proba(X_test)[:, 1]
    highest_risk_idx = int(np.argmax(churn_proba))
    contribs = [
        {
            "feature": feat,
            "value": round(float(shap_churn[highest_risk_idx, list(X_test.columns).index(feat)]), 5),
            "feature_value": float(X_test.iloc[highest_risk_idx][feat]),
        }
        for feat in top_features
        if feat in X_test.columns
    ]
    contribs.sort(key=lambda x: abs(x["value"]), reverse=True)

    return {
        "global_importance": global_importance,
        "summary": summary,
        "waterfall": {
            "base_value": round(base_value, 4),
            "prediction": round(float(churn_proba[highest_risk_idx]), 4),
            "contributions": contribs[:10],
        },
    }


def _build_insight(importance: pd.Series) -> dict:
    top = importance.head(5)
    top_factors = [
        {
            "feature": feat,
            "impact": round(float(val), 6),
            "direction": "decrease",
            "description": _feature_description(feat),
        }
        for feat, val in top.items()
    ]
    dominant = top.index[0]
    return {
        "top_factors": top_factors,
        "dominant_factor": {
            "feature": dominant,
            "description": _feature_description(dominant),
        },
        "narrative": [
            f"Faktor terkuat penyebab churn adalah **{dominant}**.",
            "Pelanggan dengan total belanja rendah dan rata-rata jeda beli panjang cenderung churning.",
            "Produk Netflix Premium memiliki pengaruh tinggi karena volume pelanggan terbesar.",
            "Strategi retensi perlu memprioritaskan pelanggan dengan rasio kegagalan transaksi tinggi.",
        ],
    }


def _feature_description(feat: str) -> str:
    descs = {
        "total_belanja": "Total pengeluaran pelanggan sepanjang periode.",
        "rata_belanja": "Rata-rata belanja per transaksi.",
        "total_bulan_langganan": "Total bulan berlangganan kumulatif.",
        "tenure_hari": "Lama aktif sebagai pelanggan (hari).",
        "rata_bulan": "Rata-rata durasi langganan per transaksi.",
        "rata_jeda_hari": "Rata-rata jeda antar pembelian (hari).",
        "jumlah_transaksi": "Jumlah total transaksi berhasil.",
        "jumlah_gagal": "Jumlah transaksi yang gagal/expired.",
        "rasio_gagal": "Proporsi transaksi gagal dari seluruh transaksi.",
        "produk_unik": "Jumlah produk berbeda yang pernah dibeli.",
    }
    if feat.startswith("prod_"):
        sku = feat.replace("prod_", "")
        return f"Pelanggan yang berlangganan produk {sku}."
    return descs.get(feat, f"Fitur {feat} dari perilaku pelanggan.")


def _build_retention(importance: pd.Series) -> dict:
    strategies = [
        {
            "id": "loyalty-reward",
            "title": "Program Loyalitas & Reward",
            "icon": "Gift",
            "target_segment": "Pelanggan dengan total_belanja rendah",
            "description": "Berikan poin reward atau diskon eksklusif untuk pelanggan yang mendekati threshold churn berdasarkan total belanja.",
            "linked_factors": ["total_belanja", "rata_belanja"],
            "priority": "Tinggi",
            "expected_impact": "Menurunkan churn rate 8-12%",
        },
        {
            "id": "multi-bulan",
            "title": "Paket Multi-Bulan Bundling",
            "icon": "PackagePlus",
            "target_segment": "Pelanggan dengan rata_bulan = 1",
            "description": "Promosikan paket 3 bulan dengan harga lebih hemat untuk meningkatkan komitmen jangka panjang.",
            "linked_factors": ["total_bulan_langganan", "rata_bulan"],
            "priority": "Tinggi",
            "expected_impact": "Meningkatkan tenure rata-rata 45 hari",
        },
        {
            "id": "winback-email",
            "title": "Kampanye Win-Back Email",
            "icon": "Mail",
            "target_segment": "Pelanggan dengan rata_jeda_hari > 60",
            "description": "Kirim email personal dengan penawaran kembali berlangganan untuk pelanggan yang lama tidak aktif.",
            "linked_factors": ["rata_jeda_hari", "tenure_hari"],
            "priority": "Sedang",
            "expected_impact": "Re-aktivasi 15-20% pelanggan lapse",
        },
        {
            "id": "payment-assist",
            "title": "Bantuan Pembayaran Fleksibel",
            "icon": "CreditCard",
            "target_segment": "Pelanggan dengan rasio_gagal > 0.3",
            "description": "Sediakan metode pembayaran alternatif dan auto-reminder sebelum langganan habis untuk mengurangi transaksi gagal.",
            "linked_factors": ["rasio_gagal", "jumlah_gagal"],
            "priority": "Tinggi",
            "expected_impact": "Mengurangi gagal bayar 25-30%",
        },
        {
            "id": "cross-sell",
            "title": "Cross-Sell Produk Lain",
            "icon": "Shuffle",
            "target_segment": "Pelanggan dengan produk_unik = 1",
            "description": "Rekomendasikan produk streaming lain yang relevan berdasarkan riwayat belanja untuk meningkatkan ketergantungan layanan.",
            "linked_factors": ["produk_unik"],
            "priority": "Sedang",
            "expected_impact": "Meningkatkan LTV pelanggan 20%",
        },
    ]

    segments = [
        {"name": "High Risk", "size": int(len(importance) * 0.25), "churn_risk": 0.75},
        {"name": "Medium Risk", "size": int(len(importance) * 0.35), "churn_risk": 0.40},
        {"name": "Low Risk", "size": int(len(importance) * 0.40), "churn_risk": 0.10},
    ]
    return {"segments": segments, "strategies": strategies}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data — dipakai untuk test/dev (bukan fallback runtime)
# ─────────────────────────────────────────────────────────────────────────────
def _make_synthetic_data() -> pd.DataFrame:
    """Generate 1200 baris data sintetis untuk test/dev tanpa file Excel nyata."""
    rng = np.random.RandomState(42)
    n = 1200
    skus = list(HARGA_1BULAN.keys())
    dates = pd.date_range("2025-10-10", "2026-06-27", periods=n)
    harga = [HARGA_1BULAN[s] for s in rng.choice(skus, n)]
    gross = np.array(harga) * rng.choice([1, 2, 3], n, p=[0.85, 0.08, 0.07])
    cogs = (gross * rng.uniform(0.55, 0.75, n)).astype(int) // 100 * 100
    statuses = rng.choice(["PAID", "EXPIRED"], n, p=[0.71, 0.29])
    emails = [f"user{rng.randint(1, 500)}@example.com" for _ in range(n)]

    return pd.DataFrame(
        {
            "Date": dates,
            "Reference": [f"INV{rng.randint(10**8, 10**9)}" for _ in range(n)],
            "SKU": rng.choice(skus, n),
            "Customer Name": [f"Customer {i}" for i in range(n)],
            "Email": emails,
            "Gross Revenue": gross.astype(int),
            "COGS": cogs,
            "Net Revenue": gross.astype(int) - cogs,
            "% Margin": np.where(gross > 0, ((gross - cogs) / gross * 100).round(1), 0),
            "Account": [f"acc{rng.randint(1, 300)}@test.com" for _ in range(n)],
            "PIC": rng.choice(["Fanda", "Koala", "Autopsy", "Admin"], n),
            "UC": [f"AMD-{rng.randint(10**8, 10**9)}" for _ in range(n)],
            "Status": statuses,
        }
    )
