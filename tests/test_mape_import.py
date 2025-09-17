import pandas as pd
import numpy as np

import types
from pathlib import Path

import pytest


def test_mape_import_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Smoke test: ensure run_endog_only executes without NameError due to mape usage.
    We monkeypatch optimize_sarimax to return a trivial best-order result to avoid long grid search.
    """
    # Import after pytest is available
    import forecaster_SARIMAX as fs

    # Prepare a larger quarterly GDP CSV (20 observations to ensure sufficient data for diagnostics)
    dates = pd.date_range("2010-03-31", periods=20, freq="QE")
    vals = np.linspace(100.0, 120.0, num=20)
    df = pd.DataFrame({"date": dates, "gdp": vals})
    series_csv = tmp_path / "gdp_US.csv"
    df.to_csv(series_csv, index=False)

    # Figures and metrics outputs
    figs_dir = tmp_path / "figures"
    metrics_csv = tmp_path / "metrics.csv"

    # Monkeypatch optimize_sarimax to one trivial candidate to speed up
    def _optimize_stub(endog, exog, order_list, d, D, s):
        import pandas as _pd

        return _pd.DataFrame(
            {
                "(p,q,P,Q)": [(0, 0, 0, 0)],
                "AIC": [0.0],
            }
        )

    monkeypatch.setattr(fs, "optimize_sarimax", _optimize_stub, raising=True)

    # Minimal argparse-like args object
    args = types.SimpleNamespace(
        use_exog=False,
        target_transform="level",
        p_range="0",  # ignored by stub
        q_range="0",
        P_range="0",
        Q_range="0",
        aic_cache=None,
        intervals="80,95",
        multi_fold=False,
    )

    # Execute
    fs.run_endog_only(series_path=series_csv, figures_dir=figs_dir, metrics_csv_path=metrics_csv, args=args)

    # Assertions: metrics CSV should exist and be non-empty; figures dir created
    assert metrics_csv.exists() and metrics_csv.stat().st_size > 0
    assert figs_dir.exists()
