"""Build the STATS19 code tables (codebook) from the DfT Data Guide and validate raw data against it.

Usage:
    python -m src.etl.codebook   # writes data/reference/codebook.csv and prints a validation report
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, REFERENCE_DIR

DATA_GUIDE = (
    REFERENCE_DIR / "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx"
)
GUIDE_SHEET = "2024_code_list"
CODEBOOK_PATH = REFERENCE_DIR / "codebook.csv"
TABLES = ("collision", "vehicle", "casualty")

RAW_FILES = {t: RAW_DIR / f"dft-road-casualty-statistics-{t}-last-5-years.csv" for t in TABLES}

# Data Guide field name -> column name actually used in the CSV extracts.
FIELD_ALIASES: dict[tuple[str, str], str] = {
    ("collision", "enhanced_collision_severity"): "enhanced_severity_collision",
    ("collision", "collision_adjusted_serious"): "collision_adjusted_severity_serious",
    ("collision", "collision_adjusted_slight"): "collision_adjusted_severity_slight",
    ("collision", "lsoa_of_collision_location"): "lsoa_of_accident_location",
    ("casualty", "casualty_adjusted_serious"): "casualty_adjusted_severity_serious",
    ("casualty", "casualty_adjusted_slight"): "casualty_adjusted_severity_slight",
}

# Data Guide rows that conflict with each other (same code, different label). Value = label to keep.
# casualty_type 23: guide lists both "Electric motorcycle rider or passenger" (2011 spec) and
# "Personal Powered Transporter" (2024 spec). Checked 2026-11-15: 1,584 of 1,585 casualties with
# casualty_type 23 sit in a vehicle with vehicle_type 23 (Electric motorcycle) and none have
# escooter_flag = 1, so code 23 means electric motorcycle. See docs/decisions.md D-008.
CONFLICT_RESOLUTION: dict[tuple[str, str, str], str] = {
    ("casualty", "casualty_type", "23"): "Electric motorcycle rider or passenger",
}

# Codes used in the data but missing from the 2024 Data Guide (added by the Sept 2026 revision).
# casualty_type 33: checked 2026-11-15, all 5,719 such casualties sit in a vehicle_type 33
# (Personal powered transporter) and 5,631 have escooter_flag = 1. See docs/decisions.md D-008.
SUPPLEMENTARY_CODES: list[dict[str, str]] = [
    {
        "table": "casualty",
        "field": "casualty_type",
        "code": "33",
        "label": "Personal powered transporter rider (incl. e-scooter)",
        "note": "not in 2024 guide; inferred from vehicle_type 33, see D-008",
    },
]

# Labels caught by MISSING_LABEL_PATTERN that are real categories, not missing data.
NOT_MISSING: set[tuple[str, str, str]] = {
    ("vehicle", "vehicle_type", "97"),  # Motorcycle - unknown cc (still a motorcycle)
    ("vehicle", "vehicle_type", "98"),  # Goods vehicle - unknown weight (still a goods vehicle)
    ("casualty", "casualty_type", "97"),
    ("casualty", "casualty_type", "98"),
    ("collision", "light_conditions", "7"),  # Darkness - lighting unknown (still dark)
}

# Area-code fields whose ONS codes change with boundary reforms; the guide lags behind, so
# unknown codes there are reported but not treated as errors. Codes are kept as-is (E=England, W=Wales).
LENIENT_FIELDS: set[tuple[str, str]] = {
    ("collision", "local_authority_ons_district"),
    ("collision", "local_authority_highway"),
}

# Labels that mean "no real information". Matched case-insensitively; review output before trusting.
MISSING_LABEL_PATTERN = re.compile(
    r"missing|out of range|unknown|not known|undefined|unallocated|not recorded|not requested",
    re.IGNORECASE,
)

# A code is either an integer (e.g. "-1", "12") or an ONS-style area code (e.g. "E06000001").
CODE_PATTERN = re.compile(r"^(-?\d+|[EWSN]\d{8})$")


def _normalise_code(value: object) -> str | None:
    """Return the code as a clean string, or None if the cell is not a single code."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text if CODE_PATTERN.match(text) else None


def build_codebook(guide_path: Path = DATA_GUIDE) -> pd.DataFrame:
    """Read the Data Guide and return one row per (table, field, code) with its label and missing flag."""
    guide = pd.read_excel(guide_path, sheet_name=GUIDE_SHEET, dtype=object)
    guide = guide.rename(columns={"field name": "guide_field", "code/format": "code"})
    guide = guide[guide["table"].isin(TABLES)].copy()

    guide["code"] = guide["code"].map(_normalise_code)
    guide = guide.dropna(subset=["code"])
    guide["label"] = guide["label"].astype(str).str.strip()
    guide["field"] = [
        FIELD_ALIASES.get((t, f), f) for t, f in zip(guide["table"], guide["guide_field"])
    ]
    extra = pd.DataFrame(SUPPLEMENTARY_CODES)
    guide = pd.concat([guide, extra], ignore_index=True)
    guide["is_missing"] = guide["label"].str.contains(MISSING_LABEL_PATTERN) | (
        guide["code"] == "-1"
    )
    keep_real = [
        (t, f, c) in NOT_MISSING for t, f, c in zip(guide["table"], guide["field"], guide["code"])
    ]
    guide.loc[keep_real, "is_missing"] = False

    codebook = guide[["table", "field", "code", "label", "is_missing", "note"]]
    # Exact repeats in the guide (same code and label) are harmless.
    codebook = codebook.drop_duplicates(subset=["table", "field", "code", "label"])
    for (table, field, code), keep_label in CONFLICT_RESOLUTION.items():
        drop = (
            (codebook["table"] == table)
            & (codebook["field"] == field)
            & (codebook["code"] == code)
            & (codebook["label"] != keep_label)
        )
        codebook = codebook[~drop]
    duplicated = codebook.duplicated(subset=["table", "field", "code"], keep=False)
    if duplicated.any():
        raise ValueError(f"Duplicate codes in Data Guide:\n{codebook[duplicated]}")
    return codebook.reset_index(drop=True)


def coded_fields(codebook: pd.DataFrame) -> dict[str, list[str]]:
    """Fields that are categorical, i.e. have at least two non-missing codes in the codebook."""
    real = codebook[~codebook["is_missing"]]
    counts = real.groupby(["table", "field"]).size()
    out: dict[str, list[str]] = {t: [] for t in TABLES}
    for (table, field), n in counts.items():
        if n >= 2:
            out[table].append(field)
    return out


def find_unknown_codes(df: pd.DataFrame, table: str, codebook: pd.DataFrame) -> pd.DataFrame:
    """Return (field, code, n_rows) for values present in ``df`` but absent from the codebook."""
    rows = []
    for field in coded_fields(codebook)[table]:
        if field not in df.columns:
            continue
        valid = set(
            codebook.loc[(codebook["table"] == table) & (codebook["field"] == field), "code"]
        )
        counts = df[field].dropna().astype(str).value_counts()
        for code, n in counts[~counts.index.isin(valid)].items():
            rows.append(
                {
                    "table": table,
                    "field": field,
                    "code": code,
                    "n_rows": int(n),
                    "lenient": (table, field) in LENIENT_FIELDS,
                }
            )
    out = pd.DataFrame(rows, columns=["table", "field", "code", "n_rows", "lenient"])
    return out.astype({"n_rows": "int64", "lenient": "bool"})


def assert_known_codes(df: pd.DataFrame, table: str, codebook: pd.DataFrame) -> None:
    """Raise ValueError if ``df`` holds codes absent from the codebook (lenient fields excepted)."""
    unknown = find_unknown_codes(df, table, codebook)
    strict = unknown[~unknown["lenient"]]
    if len(strict):
        raise ValueError(f"Unknown codes in {table}:\n{strict.to_string(index=False)}")


def missing_rates(df: pd.DataFrame, table: str, codebook: pd.DataFrame) -> pd.DataFrame:
    """Share of rows that are empty or carry a 'missing' code, for every column with codebook entries."""
    rows = []
    for field in sorted(set(codebook.loc[codebook["table"] == table, "field"]) & set(df.columns)):
        miss_codes = set(
            codebook.loc[
                (codebook["table"] == table)
                & (codebook["field"] == field)
                & codebook["is_missing"],
                "code",
            ]
        )
        values = df[field].astype("string")
        is_missing = values.isna() | values.isin(miss_codes)
        rows.append(
            {"table": table, "field": field, "missing_share": round(float(is_missing.mean()), 4)}
        )
    return pd.DataFrame(rows)


def main() -> None:
    codebook = build_codebook()
    codebook.to_csv(CODEBOOK_PATH, index=False)
    print(
        f"codebook: {len(codebook)} codes, {codebook[['table', 'field']].drop_duplicates().shape[0]} fields -> {CODEBOOK_PATH}"
    )

    print("\nLabels flagged as missing (review):")
    flagged = codebook[codebook["is_missing"]].drop_duplicates(subset=["code", "label"])
    for _, r in flagged.sort_values(["code", "label"]).iterrows():
        print(f"  {r['code']:>4}  {r['label']}")

    unknown_all, missing_all = [], []
    for table, path in RAW_FILES.items():
        df = pd.read_csv(path, dtype=str)
        guide_fields = set(codebook.loc[codebook["table"] == table, "field"])
        not_in_guide = [c for c in df.columns if c not in guide_fields]
        print(f"\n[{table}] columns with no code table in guide: {not_in_guide}")
        unknown_all.append(find_unknown_codes(df, table, codebook))
        missing_all.append(missing_rates(df, table, codebook))

    unknown = pd.concat(unknown_all, ignore_index=True)
    print("\nCodes present in data but NOT in codebook (lenient = area codes, reported only):")
    print(unknown.to_string(index=False) if len(unknown) else "  none")
    n_strict = int((~unknown["lenient"]).sum())
    print(f"STRICT unknown codes: {n_strict}")
    missing = pd.concat(missing_all, ignore_index=True)
    print("\nMissing share per field (> 1%):")
    print(missing[missing["missing_share"] > 0.01].to_string(index=False))


if __name__ == "__main__":
    main()
