"""Tests for src.analysis.metrics with known answers."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import metrics


def test_wilson_known_values():
    # Reference values (Wilson 1927; e.g. statsmodels proportion_confint method="wilson").
    low, high = metrics.wilson_ci(81, 263)
    assert low == pytest.approx(0.2553, abs=1e-4)
    assert high == pytest.approx(0.3662, abs=1e-4)


def test_wilson_edge_cases():
    low, high = metrics.wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12) and 0 < high < 0.35
    low, high = metrics.wilson_ci(10, 10)
    assert high == pytest.approx(1.0, abs=1e-12) and 0.65 < low < 1


def test_wilson_matches_statsmodels():
    from statsmodels.stats.proportion import proportion_confint

    low, high = metrics.wilson_ci(np.array([5, 250]), np.array([40, 1000]))
    ref = [proportion_confint(k, n, method="wilson") for k, n in [(5, 40), (250, 1000)]]
    assert np.allclose(low, [r[0] for r in ref]) and np.allclose(high, [r[1] for r in ref])


def test_ksi_rate_table():
    df = pd.DataFrame(
        {
            "grp": ["a"] * 40 + ["b"] * 4 + [None] * 3,
            "ksi_adj": [1.0] * 10 + [0.5] * 2 + [0.0] * 28 + [1.0, 0, 0, 0] + [1.0] * 3,
            "ksi_raw": [1] * 10 + [0] * 30 + [1, 0, 0, 0] + [1] * 3,
        }
    )
    t = metrics.ksi_rate_table(df, "grp")
    assert t.loc["a", "n"] == 40 and t.loc["a", "ksi"] == 11.0
    assert t.loc["a", "rate"] == pytest.approx(11 / 40)
    assert t.loc["a", "rate_raw"] == pytest.approx(10 / 40)
    assert t.loc["a", "ci_low"] < t.loc["a", "rate"] < t.loc["a", "ci_high"]
    assert not t.loc["a", "small_n"] and t.loc["b", "small_n"]  # b: n=4 < 30
    assert len(t) == 2  # missing group excluded


def test_missing_table():
    df = pd.DataFrame({"x": [1, None, None, 4], "y": [1, 2, 3, 4]})
    t = metrics.missing_table(df)
    assert t.index[0] == "x" and t.loc["x", "missing_share"] == 0.5 and t.loc["y", "n_missing"] == 0
