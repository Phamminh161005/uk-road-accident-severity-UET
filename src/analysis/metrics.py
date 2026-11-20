"""Summary metrics: KSI rates with Wilson 95% confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054

# Groups below these sizes are flagged and should not be reported as rates (analysis_rules.md).
MIN_N = 30
MIN_KSI = 5


def wilson_ci(
    successes: float | np.ndarray, n: float | np.ndarray, z: float = Z_95
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a proportion.

    ``successes`` may be fractional (sum of the adjusted KSI probability); the interval is then an
    approximation that is accurate when ``n`` is large.
    """
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = successes / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def ksi_rate_table(
    df: pd.DataFrame,
    by: str | list[str],
    target: str = "ksi_adj",
    raw_target: str | None = "ksi_raw",
    min_n: int = MIN_N,
    min_ksi: int = MIN_KSI,
) -> pd.DataFrame:
    """KSI rate per group with Wilson 95% CI; rows with a missing group value are left out.

    Columns: n, ksi (sum of target), rate, ci_low, ci_high, [rate_raw], small_n.
    """
    keys = [by] if isinstance(by, str) else list(by)
    g = df.dropna(subset=keys).groupby(keys, observed=True)
    out = pd.DataFrame({"n": g.size(), "ksi": g[target].sum()})
    out["rate"] = out["ksi"] / out["n"]
    out["ci_low"], out["ci_high"] = wilson_ci(out["ksi"], out["n"])
    if raw_target:
        out["rate_raw"] = g[raw_target].mean()
    out["small_n"] = (out["n"] < min_n) | (out["ksi"] < min_ksi)
    return out


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of missing values per column, sorted descending."""
    share = df.isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"missing_share": share, "n_missing": df.isna().sum()[share.index]})
