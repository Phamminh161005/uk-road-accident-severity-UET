"""Tests for src.etl.clean on small synthetic tables (no raw files needed)."""

import numpy as np
import pandas as pd
import pytest

from src.etl import clean


def _codes(field: str, table: str, items: list[tuple[str, str, bool]]) -> list[dict]:
    return [
        {"table": table, "field": field, "code": c, "label": lbl, "is_missing": m, "note": None}
        for c, lbl, m in items
    ]


@pytest.fixture
def codebook() -> pd.DataFrame:
    rows = (
        _codes(
            "collision_severity",
            "collision",
            [("1", "Fatal", False), ("2", "Serious", False), ("3", "Slight", False)],
        )
        + _codes(
            "weather_conditions",
            "collision",
            [
                ("2", "Raining no high winds", False),
                ("1", "Fine no high winds", False),
                ("9", "Unknown", True),
                ("-1", "Data missing or out of range", True),
            ],
        )
        + _codes(
            "junction_detail_historic",
            "collision",
            [("1", "Roundabout", False), ("2", "Mini-roundabout", False)],
        )
        + _codes(
            "speed_limit",
            "collision",
            [("-1", "Data missing or out of range", True), ("99", "unknown (self reported)", True)],
        )
        + _codes(
            "collision_injury_based",
            "collision",
            [("0", "Based on severity reporting", False), ("1", "Based on Injury code", False)],
        )
    )
    return pd.DataFrame(rows)


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collision_index": ["2021A", "2021B", "2022C", "2023D"],
            "collision_year": ["2021", "2021", "2022", "2023"],
            "collision_severity": ["1", "2", "3", "3"],
            "weather_conditions": ["1", "2", "9", "-1"],
            "junction_detail_historic": ["1", "2", "1", "1"],
            "speed_limit": ["30", "-1", "99", "20"],
            "collision_injury_based": ["1", "1", "0", "0"],
            "collision_adjusted_severity_serious": ["0", "1", "0.25", "0.10"],
        }
    )


def test_to_numeric_sets_missing_codes_to_nan():
    s = clean.to_numeric(pd.Series(["30", "-1", "99", None]), {"-1", "99"})
    assert s.iloc[0] == 30 and s.iloc[1:].isna().all()


def test_to_bool():
    s = clean.to_bool(pd.Series(["1", "0", "-1", None]))
    assert s.tolist()[:2] == [True, False] and s.iloc[2:].isna().all()


def test_decode_orders_by_code_and_drops_missing(codebook):
    codes = codebook[codebook.field == "weather_conditions"]
    out = pd.Series(clean.decode(pd.Series(["2", "1", "9", "-1"]), codes))
    assert list(out.cat.categories) == ["Fine no high winds", "Raining no high winds"]
    assert out.tolist()[:2] == ["Raining no high winds", "Fine no high winds"]
    assert out.iloc[2:].isna().all()


def test_clean_table(raw, codebook):
    df = clean.clean_table(raw, "collision", codebook)
    assert "junction_detail_historic" not in df.columns
    assert df["speed_limit"].tolist()[0] == 30 and df["speed_limit"].isna().sum() == 2
    assert df["weather_conditions"].isna().sum() == 2
    assert df["collision_injury_based"].dtype == "boolean"
    assert str(df["collision_year"].dtype) == "Int64"


def test_clean_table_rejects_unknown_code(raw, codebook):
    raw.loc[0, "weather_conditions"] = "7"
    with pytest.raises(ValueError, match="Unknown codes"):
        clean.clean_table(raw, "collision", codebook)


def test_add_ksi(raw, codebook):
    df = clean.add_ksi(
        clean.clean_table(raw, "collision", codebook),
        "collision_severity",
        "collision_adjusted_severity_serious",
    )
    # Fatal -> 1 even though adjusted probability is 0; others use the adjusted probability.
    assert df["ksi_adj"].tolist() == [1.0, 1.0, 0.25, 0.10]
    assert df["ksi_raw"].tolist() == [1, 1, 0, 0]


def test_add_ksi_rejects_bad_severity():
    df = pd.DataFrame({"sev": ["Fatal", "Other"], "adj": [0.0, 0.5]})
    with pytest.raises(ValueError, match="Unexpected values"):
        clean.add_ksi(df, "sev", "adj")


def test_country_and_coordinates():
    df = pd.DataFrame(
        {
            "date": ["17/09/2023", "18/09/2023", "19/09/2023"],
            "time": ["08:30", "", "23:05"],
            "latitude": [51.5, 52.0, 10.0],  # last one outside GB
            "longitude": [-0.1, -3.5, 0.0],
            "collision_severity": ["Slight", "Slight", "Fatal"],
            "collision_adjusted_severity_serious": [0.2, 0.0, 0.0],
        }
    )
    df["time"] = df["time"].replace("", np.nan)
    out = clean.add_collision_fields(df, pd.Series(["1", "63", "99"]))
    assert out["country"].astype(str).tolist() == ["England", "Wales", "Scotland"]
    assert out["hour"].tolist()[0] == 8 and pd.isna(out["hour"].iloc[1])
    assert out["datetime"].iloc[0] == pd.Timestamp("2023-09-17 08:30")
    assert np.isnan(out["latitude"].iloc[2])


def test_check_table_rejects_duplicate_keys():
    df = pd.DataFrame({"collision_index": ["A", "A"], "collision_year": [2021, 2021]})
    with pytest.raises(ValueError, match="duplicated keys"):
        clean.check_table(df, "collision", raw_rows=2)


def test_check_table_rejects_years_outside_range():
    df = pd.DataFrame({"collision_index": ["A"], "collision_year": [2019]})
    with pytest.raises(ValueError, match="years outside"):
        clean.check_table(df, "collision", raw_rows=1)
