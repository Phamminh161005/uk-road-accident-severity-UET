"""Clean and decode the three STATS19 tables and create the KSI target variables.

Usage:
    python -m src.etl.clean   # writes data/processed/{collision,vehicle,casualty}.parquet + clean_report.json

Steps per table (see docs/decisions.md D-007, D-008):
    1. read raw CSV as text; fail on codes absent from the codebook
    2. drop superseded columns (*_historic, local_authority_district)
    3. numeric columns -> numbers, with "missing" codes (-1, 99 ...) set to NaN
    4. flag columns (0/1) -> boolean
    5. coded columns -> labelled categoricals ordered by code; missing codes -> NaN
    6. table-specific derived columns (datetime, country, KSI)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, YEARS
from src.etl.codebook import (
    CODEBOOK_PATH,
    LENIENT_FIELDS,
    RAW_FILES,
    TABLES,
    assert_known_codes,
    build_codebook,
    coded_fields,
)

# Columns kept as text identifiers (not decoded).
TEXT_FIELDS = {
    "collision_index",
    "collision_ref_no",
    "local_authority_ons_district",
    "local_authority_highway",
    "local_authority_highway_current",
    "lsoa_of_accident_location",
    "lsoa_of_driver",
    "lsoa_of_casualty",
    "generic_make_model",
}

NUMERIC_FIELDS = {
    "collision_year",
    "vehicle_reference",
    "casualty_reference",
    "location_easting_osgr",
    "location_northing_osgr",
    "longitude",
    "latitude",
    "number_of_vehicles",
    "number_of_casualties",
    "speed_limit",
    "first_road_number",
    "second_road_number",
    "age_of_driver",
    "engine_capacity_cc",
    "age_of_vehicle",
    "age_of_casualty",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
    "casualty_adjusted_severity_serious",
    "casualty_adjusted_severity_slight",
}

# 0/1 flags: stored as nullable booleans instead of long labels.
BOOL_FIELDS = {"escooter_flag", "collision_injury_based", "casualty_injury_based"}

# local_authority_district is deprecated since 2019 (99.96% missing); *_historic replaced by 2024 spec.
DROP_SUFFIX = "_historic"
DROP_FIELDS = {"local_authority_district"}

# Police force -> country (Data Guide labels). Scotland has a single force since 2013 (code 99);
# codes 91-98 are pre-2013 Scottish forces. All other codes are English forces.
WALES_FORCES = {60, 61, 62, 63}  # North Wales, Gwent, South Wales, Dyfed-Powys
SCOTLAND_FORCES = {91, 92, 93, 94, 95, 96, 97, 98, 99}

# Rough bounding box for Great Britain (WGS84).
GB_LAT = (49.8, 60.95)
GB_LON = (-8.7, 1.9)


def _field_codes(codebook: pd.DataFrame, table: str, field: str) -> pd.DataFrame:
    return codebook[(codebook["table"] == table) & (codebook["field"] == field)]


def to_numeric(series: pd.Series, missing_codes: set[str]) -> pd.Series:
    """Convert text to numbers, turning missing codes and blanks into NaN."""
    cleaned = series.where(~series.isin(missing_codes))
    return pd.to_numeric(cleaned, errors="raise")


def to_bool(series: pd.Series) -> pd.Series:
    """Map "1" -> True, "0" -> False, anything else (blank, -1) -> <NA>."""
    return series.map({"1": True, "0": False}).astype("boolean")


def decode(series: pd.Series, codes: pd.DataFrame) -> pd.Series:
    """Replace codes by labels as an ordered-by-code categorical; missing codes become NaN."""
    real = codes[~codes["is_missing"]].copy()
    real["order"] = pd.to_numeric(real["code"], errors="coerce")
    real = real.sort_values(["order", "code"])
    labels = pd.Series(real["label"].values, index=real["code"].values)
    categories = list(dict.fromkeys(real["label"]))  # unique, in code order
    return pd.Categorical(series.map(labels), categories=categories)


def clean_table(raw: pd.DataFrame, table: str, codebook: pd.DataFrame) -> pd.DataFrame:
    """Generic cleaning shared by all tables (steps 1-5 in the module docstring)."""
    assert_known_codes(raw, table, codebook)
    decodable = set(coded_fields(codebook)[table])
    out = {}
    for col in raw.columns:
        if col.endswith(DROP_SUFFIX) or col in DROP_FIELDS:
            continue
        codes = _field_codes(codebook, table, col)
        missing = set(codes.loc[codes["is_missing"], "code"])
        if col in BOOL_FIELDS:
            out[col] = to_bool(raw[col])
        elif col in NUMERIC_FIELDS:
            out[col] = to_numeric(raw[col], missing)
        elif col in TEXT_FIELDS or (table, col) in LENIENT_FIELDS:
            out[col] = raw[col].where(~raw[col].isin(missing)).astype("string")
        elif col in decodable:
            out[col] = decode(raw[col], codes)
        else:
            out[col] = raw[col].astype("string")
    df = pd.DataFrame(out, index=raw.index)
    for col in ("collision_year", "vehicle_reference", "casualty_reference"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


def add_ksi(df: pd.DataFrame, severity_col: str, adjusted_serious_col: str) -> pd.DataFrame:
    """Add ksi_raw (0/1 from recorded severity) and ksi_adj (DfT adjusted, in [0, 1]); D-007.

    Fatal records carry no adjustment probability (both columns are 0), so ksi_adj = 1 for Fatal.
    """
    severity = df[severity_col].astype(str)
    if not severity.isin(["Fatal", "Serious", "Slight"]).all():
        raise ValueError(f"Unexpected values in {severity_col}: {severity.unique()}")
    df["ksi_raw"] = severity.isin(["Fatal", "Serious"]).astype("int8")
    df["ksi_adj"] = np.where(severity == "Fatal", 1.0, df[adjusted_serious_col]).astype("float64")
    return df


def add_collision_fields(df: pd.DataFrame, force_code: pd.Series) -> pd.DataFrame:
    """Datetime, hour, month, country and coordinate checks for the collision table.

    ``force_code`` is the raw numeric police_force code (before decoding to labels).
    """
    date = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="raise")
    time = pd.to_datetime(df["time"], format="%H:%M", errors="coerce")
    df["date"] = date
    df["hour"] = time.dt.hour.astype("Int8")
    df["month"] = date.dt.month.astype("int8")
    df["datetime"] = date + pd.to_timedelta(time.dt.hour * 60 + time.dt.minute, unit="m")
    df = df.drop(columns=["time"])

    force = pd.to_numeric(force_code)
    df["country"] = pd.Categorical(
        np.select(
            [force.isin(WALES_FORCES), force.isin(SCOTLAND_FORCES)],
            ["Wales", "Scotland"],
            default="England",
        ),
        categories=["England", "Wales", "Scotland"],
    )

    in_gb = df["latitude"].between(*GB_LAT) & df["longitude"].between(*GB_LON)
    df.loc[~in_gb, ["latitude", "longitude"]] = np.nan
    if "junction_control" in df and "junction_detail" in df:
        df = fill_junction_control(df)
    return add_ksi(df, "collision_severity", "collision_adjusted_severity_serious")


NOT_AT_JUNCTION = "Not at junction or within 20 metres"


def fill_junction_control(df: pd.DataFrame) -> pd.DataFrame:
    """Blank junction_control is 'not applicable' when the collision was not at a junction (D-011).

    Police leave junction_control as -1 for ~214k collisions not at a junction; the Data Guide's
    code 0 label for this case is used instead of treating them as missing.
    """
    # fillna(False): a missing junction_detail must not count as "not at junction".
    not_at_junction = (df["junction_detail"].astype("string") == NOT_AT_JUNCTION).fillna(False)
    fill = not_at_junction.astype(bool) & df["junction_control"].isna()
    control = df["junction_control"]
    if NOT_AT_JUNCTION not in control.cat.categories:
        control = control.cat.add_categories([NOT_AT_JUNCTION])
    df["junction_control"] = control.mask(fill, NOT_AT_JUNCTION)
    return df


def clean_all(codebook: pd.DataFrame | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Clean every table; return the frames and a data-check report."""
    codebook = build_codebook() if codebook is None else codebook
    frames, report = {}, {}
    for table in TABLES:
        raw = pd.read_csv(RAW_FILES[table], dtype=str, keep_default_na=False, na_values=[""])
        df = clean_table(raw, table, codebook)
        if table == "collision":
            df = add_collision_fields(df, raw["police_force"])
        elif table == "casualty":
            df = add_ksi(df, "casualty_severity", "casualty_adjusted_severity_serious")
        frames[table] = df
        report[table] = check_table(df, table, raw_rows=len(raw))
    return frames, report


def check_table(df: pd.DataFrame, table: str, raw_rows: int) -> dict:
    """Post-ETL checks required by docs/data_rules.md; raises on hard failures."""
    years = set(df["collision_year"].dropna().unique())
    if not years <= set(YEARS):
        raise ValueError(f"{table}: years outside {YEARS}: {sorted(years - set(YEARS))}")
    if len(df) != raw_rows:
        raise ValueError(f"{table}: row count changed {raw_rows} -> {len(df)}")
    key = {
        "collision": ["collision_index"],
        "vehicle": ["collision_index", "vehicle_reference"],
        "casualty": ["collision_index", "casualty_reference"],
    }[table]
    n_dup = int(df.duplicated(subset=key).sum())
    if n_dup:
        raise ValueError(f"{table}: {n_dup} duplicated keys {key}")

    rep: dict = {
        "rows": len(df),
        "rows_by_year": {int(k): int(v) for k, v in df["collision_year"].value_counts().items()},
        "key": key,
        "duplicated_keys": n_dup,
    }
    if "ksi_adj" in df:
        if not df["ksi_adj"].between(0, 1).all():
            raise ValueError(f"{table}: ksi_adj outside [0, 1]")
        rep["ksi_raw_rate"] = round(float(df["ksi_raw"].mean()), 4)
        rep["ksi_adj_rate"] = round(float(df["ksi_adj"].mean()), 4)
    if table == "collision":
        if df["date"].dt.year.ne(df["collision_year"]).any():
            raise ValueError("collision: date year differs from collision_year")
        rep["missing_coordinates"] = int(df["latitude"].isna().sum())
        rep["missing_time"] = int(df["hour"].isna().sum())
        rep["rows_by_country"] = {k: int(v) for k, v in df["country"].value_counts().items()}
        ons_country = (
            df["local_authority_ons_district"]
            .str[0]
            .map({"E": "England", "W": "Wales", "S": "Scotland"})
        )
        known = ons_country.notna()
        rep["country_mismatch_vs_ons_district"] = int(
            (ons_country[known] != df.loc[known, "country"].astype(str)).sum()
        )
    return rep


def main() -> None:
    frames, report = clean_all()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for table, df in frames.items():
        path = PROCESSED_DIR / f"{table}.parquet"
        df.to_parquet(path, index=False)
        report[table]["file"] = str(path.relative_to(PROCESSED_DIR.parent.parent))
        report[table]["size_mb"] = round(path.stat().st_size / 1e6, 1)
    report["_meta"] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "codebook": str(CODEBOOK_PATH.name),
        "ksi_definition": "docs/decisions.md D-007",
    }
    (PROCESSED_DIR / "clean_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
