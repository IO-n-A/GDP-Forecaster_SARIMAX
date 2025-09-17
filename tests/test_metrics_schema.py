import pandas as pd
import numpy as np

import types
from pathlib import Path

import pytest


REQUIRED_COLS = [
    "mode",
    "country",
    "n",
    "train_len",
    "test_len",
    "model",
    "ME",
    "MAE",
    "RMSE",
    "MAPE",
    "sMAPE",
    "median_APE",
    "MASE",
    "TheilU1",
    "TheilU2",
    "DM_t",
    "DM_p",
]


def test_metrics_schema_and_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import forecaster_SARIMAX as fs

    # Minimal quarterly series with some variation
    dates = pd.date_range("2012-03-31", periods=12, freq="QE")
    vals = np.linspace(100.0, 111.0, num=12)
    df = pd.DataFrame({"date": dates, "gdp": vals})
    series_csv = tmp_path / "gdp_US.csv"
    df.to_csv(series_csv, index=False)

    figs_dir = tmp_path / "figs"
    metrics_csv = tmp_path / "metrics.csv"

    def _optimize_stub(endog, exog, order_list, d, D, s):
        import pandas as _pd

        return _pd.DataFrame(
            {
                "(p,q,P,Q)": [(0, 0, 0, 0)],
                "AIC": [0.0],
            }
        )

    monkeypatch.setattr(fs, "optimize_sarimax", _optimize_stub, raising=True)

    args = types.SimpleNamespace(
        use_exog=False,
        target_transform="level",
        p_range="0",
        q_range="0",
        P_range="0",
        Q_range="0",
        aic_cache=None,
        intervals="80,95",
        multi_fold=False,
    )

    fs.run_endog_only(series_path=series_csv, figures_dir=figs_dir, metrics_csv_path=metrics_csv, args=args)

    assert metrics_csv.exists() and metrics_csv.stat().st_size > 0
    m = pd.read_csv(metrics_csv)
    # Expect at least SARIMA(best) + naive_last rows for the last run
    assert (m["model"] == "SARIMA(best)").any()
    assert (m["model"] == "naive_last").any()
    # Schema contains required columns
    for c in REQUIRED_COLS:
        assert c in m.columns