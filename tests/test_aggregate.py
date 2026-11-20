"""Tests for src.etl.aggregate: web tables must keep every collision exactly once."""

import pandas as pd
import pytest

from src.etl import aggregate


@pytest.fixture
def collision() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collision_index": ["a", "b", "c", "d"],
            "collision_year": [2023, 2023, 2023, 2024],
            "month": [1, 1, 1, 2],
            "country": pd.Categorical(["England", "England", "Wales", "England"]),
            "collision_severity": ["Fatal", "Slight", "Serious", "Slight"],
            "number_of_casualties": [2, 1, 1, 3],
            "ksi_adj": [1.0, 0.2, 1.0, 0.1],
            "ksi_raw": [1, 0, 1, 0],
        }
    )


def test_overview_counts(collision):
    out = aggregate.overview_monthly(collision).set_index(aggregate.OVERVIEW_KEYS)
    eng = out.loc[(2023, 1, "England")]
    assert eng["collisions"] == 2 and eng["casualties"] == 3 and eng["fatal"] == 1
    assert eng["ksi_adj"] == pytest.approx(1.2) and eng["ksi_raw"] == 1
    assert out["collisions"].sum() == len(collision) and len(out) == 3


def test_overview_rejects_missing_keys(collision):
    collision.loc[0, "month"] = None
    with pytest.raises(ValueError, match="without year/month/country"):
        aggregate.overview_monthly(collision)
