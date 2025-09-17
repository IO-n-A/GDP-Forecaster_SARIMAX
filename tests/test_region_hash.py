import pandas as pd
import numpy as np
from pathlib import Path
import types

import pytest


def test_file_tag_for_oecd_mapping():
    from fetchers.fetch_gdp import file_tag_for_oecd

    assert file_tag_for_oecd("USA") == "US"
    assert file_tag_for_oecd("CHN") == "CN"
    assert file_tag_for_oecd("EU27_2020") == "EU27_2020"
    assert file_tag_for_oecd("EA19") == "EA19"
    # Unknown passes through unchanged
    assert file_tag_for_oecd("XXX") == "XXX"


def test_region_forecast_hashes_differ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Run endog-only twice with different region series and ensure SARIMA forecast hashes differ.
    """
    import forecaster_SARIMAX as fs

    # Two simple quarterly series with different trajectories
    dates = pd.date_range("2015-03-31", periods=12, freq="QE")
    us_vals = np.linspace(100.0, 111.0, num=12)
    eu_vals = np.linspace(200.0, 207.0, num=12)  # different scale/trajectory

    us_csv = tmp_path / "gdp_US.csv"
    eu_csv = tmp_path / "gdp_EU27_2020.csv"
    pd.DataFrame({"date": dates, "gdp": us_vals}).to_csv(us_csv, index=False)
    pd.DataFrame({"date": dates, "gdp": eu_vals}).to_csv(eu_csv, index=False)

    metrics_csv = tmp_path / "metrics.csv"

    # Stub optimize_sarimax to force a single trivial candidate for speed
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

    # Run for US and EU
    fs.run_endog_only(series_path=us_csv, figures_dir=tmp_path / "figs_US", metrics_csv_path=metrics_csv, args=args)
    fs.run_endog_only(series_path=eu_csv, figures_dir=tmp_path / "figs_EU", metrics_csv_path=metrics_csv, args=args)

    # Load metrics and compare hashes for SARIMA(best)
    m = pd.read_csv(metrics_csv)
    m = m[m["model"] == "SARIMA(best)"].copy()
    assert {"US", "EU27_2020"} <= set(m["country"].unique())

    last_by_country = (
        m.assign(_ord_=range(len(m)))
        .sort_values("_ord_")
        .groupby("country", as_index=False)
        .tail(1)
    )

    us_hash = last_by_country[last_by_country["country"] == "US"]["hash_SARIMA"].iloc[0]
    eu_hash = last_by_country[last_by_country["country"] == "EU27_2020"]["hash_SARIMA"].iloc[0]

    assert isinstance(us_hash, str) and isinstance(eu_hash, str)
    assert us_hash != "" and eu_hash != ""
    assert us_hash != eu_hash