"""
test_smoke.py — smoke test end-to-end pipeline pakai data sintetis.

Jalankan:  pytest backend/test_smoke.py   atau   python backend/test_smoke.py
Tak butuh file Excel nyata; _make_synthetic_data() bikin datanya.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from pipeline import (
    DataValidationError,
    _make_synthetic_data,
    clean,
    compute_shap,
    feature_engineering,
    load_dataframe,
    run_pipeline,
    train_model,
)

RESULT_KEYS = {
    "metrics", "dataset", "eda", "feature_importance", "shap", "insight", "retention",
}


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _raises(exc, fn, *args) -> None:
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"{exc.__name__} tidak dilempar")


def test_run_pipeline_shape():
    result = run_pipeline(_xlsx_bytes(_make_synthetic_data()))

    assert set(result.keys()) == RESULT_KEYS

    m = result["metrics"]["model"]
    for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert 0.0 <= m[k] <= 1.0

    assert result["feature_importance"]["features"], "importance kosong"
    w = result["shap"]["waterfall"]
    assert {"base_value", "prediction", "contributions"} <= set(w)
    assert result["metrics"]["total_customers"] > 0


def test_shap_reconciliation():
    """base_value + Σ shap harus ≈ predict_proba (bukti expected_value benar)."""
    df, _ = clean(load_dataframe(_xlsx_bytes(_make_synthetic_data())))
    cust, _ = feature_engineering(df)
    model, _Xtr, X_test, *_ = train_model(cust)
    shap_churn, _importance, base = compute_shap(model, X_test)

    proba = model.predict_proba(X_test)[:, 1]
    recon = base + shap_churn.sum(axis=1)
    assert np.allclose(recon, proba, atol=1e-4), (
        f"rekonsiliasi SHAP gagal: maks selisih {np.abs(recon - proba).max():.2e}"
    )


def test_missing_column_rejected():
    bad = _make_synthetic_data().drop(columns=["Email"])
    _raises(DataValidationError, load_dataframe, _xlsx_bytes(bad))


def test_bad_dates_rejected():
    bad = _make_synthetic_data()
    bad["Date"] = "bukan-tanggal"
    _raises(DataValidationError, load_dataframe, _xlsx_bytes(bad))


if __name__ == "__main__":
    test_run_pipeline_shape()
    test_shap_reconciliation()
    test_missing_column_rejected()
    test_bad_dates_rejected()
    print("OK — semua smoke test lolos.")
