"""Build small pre-aggregated tables for the Streamlit app (the app never reads row-level data).

Run: python -m src.etl.aggregate
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import AGG_DIR, PROCESSED_DIR

OVERVIEW_KEYS = ["collision_year", "month", "country"]


def overview_monthly(collision: pd.DataFrame) -> pd.DataFrame:
    """Collision-level counts per year x month x country.

    ksi_adj is summed (expected number of KSI collisions, DfT adjusted severity);
    ksi_raw is the recorded count, kept for sensitivity checks.
    """
    missing = collision[OVERVIEW_KEYS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} collisions without year/month/country")

    df = collision.assign(
        country=collision["country"].astype(str),
        fatal=(collision["collision_severity"].astype(str) == "Fatal").astype(int),
    )
    out = (
        df.groupby(OVERVIEW_KEYS, observed=True)
        .agg(
            collisions=("collision_index", "size"),
            casualties=("number_of_casualties", "sum"),
            fatal=("fatal", "sum"),
            ksi_adj=("ksi_adj", "sum"),
            ksi_raw=("ksi_raw", "sum"),
        )
        .reset_index()
        .astype({"collision_year": "int16", "month": "int8"})
        .sort_values(OVERVIEW_KEYS, ignore_index=True)
    )
    if out["collisions"].sum() != len(collision):
        raise ValueError("aggregation lost or duplicated collisions")
    return out


def main() -> None:
    collision = pd.read_parquet(PROCESSED_DIR / "collision_analysis.parquet")
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    overview = overview_monthly(collision)
    overview.to_parquet(AGG_DIR / "overview_monthly.parquet", index=False)

    report = {
        "overview_monthly": {
            "rows": len(overview),
            "collisions": int(overview["collisions"].sum()),
            "casualties": int(overview["casualties"].sum()),
            "ksi_adj": round(float(overview["ksi_adj"].sum()), 1),
        }
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
