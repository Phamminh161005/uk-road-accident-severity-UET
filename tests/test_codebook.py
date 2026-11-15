"""Tests for src.etl.codebook using a tiny synthetic Data Guide (no raw files needed)."""

from pathlib import Path

import pandas as pd
import pytest

from src.etl import codebook as cb


@pytest.fixture
def guide_path(tmp_path: Path) -> Path:
    rows = [
        # table, field name, code/format, label, note
        ("collision", "collision_index", None, None, "unique id"),
        ("collision", "date", "(DD/MM/YYYY)", None, None),
        ("collision", "weather_conditions", 1, "Fine no high winds", None),
        ("collision", "weather_conditions", 2, "Raining no high winds", None),
        ("collision", "weather_conditions", 9, "Unknown", None),
        ("collision", "weather_conditions", -1, "Data missing or out of range", None),
        ("collision", "weather_conditions", 2, "Raining no high winds", None),  # exact repeat
        ("collision", "light_conditions", 1, "Daylight", None),
        ("collision", "light_conditions", 7, "Darkness - lighting unknown", None),
        ("collision", "enhanced_collision_severity", 1, "Fatal", None),  # aliased field
        ("collision", "enhanced_collision_severity", 3, "Serious", None),
        ("collision", "local_authority_highway", "E06000001", "Hartlepool", None),
        ("collision", "local_authority_highway", "E06000002", "Middlesbrough", None),
        ("casualty", "casualty_type", 0, "Pedestrian", None),
        ("casualty", "casualty_type", 23, "Electric motorcycle rider or passenger", "2011 spec"),
        ("casualty", "casualty_type", 23, "Personal Powered Transporter", "2024 spec"),  # conflict
        ("historical_revisions", "variable", None, None, None),
    ]
    df = pd.DataFrame(rows, columns=["table", "field name", "code/format", "label", "note"])
    path = tmp_path / "guide.xlsx"
    df.to_excel(path, sheet_name=cb.GUIDE_SHEET, index=False)
    return path


@pytest.fixture
def book(guide_path: Path) -> pd.DataFrame:
    return cb.build_codebook(guide_path)


def _label(book: pd.DataFrame, table: str, field: str, code: str) -> pd.Series:
    return book[(book.table == table) & (book.field == field) & (book.code == code)]


def test_only_real_codes_kept(book):
    assert "date" not in set(book.field)
    assert "collision_index" not in set(book.field)
    assert "historical_revisions" not in set(book.table)


def test_codes_are_unique_after_dedup_and_conflict(book):
    assert not book.duplicated(subset=["table", "field", "code"]).any()
    row = _label(book, "casualty", "casualty_type", "23")
    assert row.label.tolist() == ["Electric motorcycle rider or passenger"]


def test_supplementary_code_added(book):
    assert len(_label(book, "casualty", "casualty_type", "33")) == 1


def test_field_alias_applied(book):
    assert "enhanced_severity_collision" in set(book.field)
    assert "enhanced_collision_severity" not in set(book.field)


def test_missing_flags(book):
    assert _label(book, "collision", "weather_conditions", "-1").is_missing.item()
    assert _label(book, "collision", "weather_conditions", "9").is_missing.item()
    assert not _label(book, "collision", "weather_conditions", "2").is_missing.item()
    # "Darkness - lighting unknown" is a real category (still dark)
    assert not _label(book, "collision", "light_conditions", "7").is_missing.item()


def test_unknown_code_detected_and_raises(book):
    df = pd.DataFrame({"weather_conditions": ["1", "2", "8", "8", None]})
    unknown = cb.find_unknown_codes(df, "collision", book)
    assert unknown[["field", "code", "n_rows"]].values.tolist() == [["weather_conditions", "8", 2]]
    with pytest.raises(ValueError, match="Unknown codes"):
        cb.assert_known_codes(df, "collision", book)


def test_lenient_area_codes_do_not_raise(book):
    df = pd.DataFrame({"local_authority_highway": ["E06000001", "E06000099"]})
    unknown = cb.find_unknown_codes(df, "collision", book)
    assert unknown.lenient.all() and len(unknown) == 1
    cb.assert_known_codes(df, "collision", book)  # no error


def test_missing_rates(book):
    df = pd.DataFrame({"weather_conditions": ["1", "-1", "9", None]})
    rates = cb.missing_rates(df, "collision", book)
    assert rates.set_index("field").loc["weather_conditions", "missing_share"] == 0.75
