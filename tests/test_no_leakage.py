import pandas as pd
import numpy as np
from pathlib import Path
import types

import pytest


def test_no_leakage_rolling_equals_per_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verify that rolling_one_step_predictions matches a reconstruction using one_step_forecast_at,
    ensuring no leakage beyond t-1 is used for forecasting at t.
    """
    import forecaster_SARIMAX as fs

    # Construct a small quarterly series with trend + noise
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-03-31", periods=16, freq="QE")
    vals = 100.0 + np.linspace(0.0, 7.5, num=len(dates)) + rng.normal(0, 0.5, size=len(dates))
    series_csv = tmp_path / "gdp_US.csv"
    pd.DataFrame({"date": dates, "gdp": vals}).to_csv(series_csv, index=False)

    # Minimal args with level transform and narrow grid; stub grid search for speed
    def _optimize_stub(endog, exog, order_list, d, D, s):
        import pandas as _pd
        return _pd.DataFrame({"(p,q,P,Q)": [(0, 0, 0, 0)], "AIC": [0.0]})

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
        log_level="INFO",
    )

    # Load series and apply transform similarly to run_endog_only
    df = pd.read_csv(series_csv)
    df["date"] = pd.to_datetime(df["date"])
    endog_raw = pd.Series(df["gdp"].values, index=df["date"], name="gdp")
    endog, d_for_transform = fs._apply_target_transform(endog_raw, args.target_transform)
    exog_model = None
    d, D, s = d_for_transform, 0, 4

    # Emulate grid selection (stub returns (0,0,0,0))
    best_order = (0, 0, 0, 0)

    # Use same test/train split logic as run_endog_only (default 16 with adaptive shrink)
    TEST_LEN_DEFAULT = 16
    if len(endog) <= TEST_LEN_DEFAULT + 8:
        test_len = max(1, min(TEST_LEN_DEFAULT, len(endog) // 4))
    else:
        test_len = TEST_LEN_DEFAULT
    train_len = len(endog) - test_len

    coverage_levels = [80, 95]
    preds_roll, conf_roll = fs.rolling_one_step_predictions(
        endog=endog,
        exog_model=exog_model,
        best_order=best_order,
        d=d,
        D=D,
        s=s,
        coverage_levels=coverage_levels,
        start_index=train_len,
    )

    # Reconstruct predictions step-by-step using only data up to i-1
    preds_step = []
    for i in range(train_len, len(endog)):
        y_hat_i = fs.one_step_forecast_at(
            endog=endog,
            exog_model=exog_model,
            best_order=best_order,
            d=d,
            D=D,
            s=s,
            i=i,
        )
        preds_step.append(y_hat_i)

    # Assert sequences match closely
    assert len(preds_roll) == len(preds_step)
    assert np.allclose(preds_roll, preds_step, rtol=1e-10, atol=1e-10)