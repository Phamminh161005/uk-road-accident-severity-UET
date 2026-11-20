"""Cross-check yearly STATS19 totals against the official DfT summary table RAS0101.

Usage:
    python -m src.etl.ras_check   # prints the comparison and writes data/processed/ras_check.csv
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from src.config import PROCESSED_DIR, REFERENCE_DIR, YEARS

RAS_ZIP = REFERENCE_DIR / "ras-all-tables-excel.zip"
RAS_FILE = "ras0101.xlsx"


def _read_sheet(sheet: str, header_row: int) -> pd.DataFrame:
    with zipfile.ZipFile(RAS_ZIP) as z:
        data = io.BytesIO(z.read(RAS_FILE))
    df = pd.read_excel(data, sheet_name=sheet, header=header_row)
    # "FSC (adjusted)\n[note 2]" -> "FSC (adjusted)"; "Pedal \ncycles" -> "Pedal cycles"
    df.columns = [" ".join(str(c).split()).split(" [")[0] for c in df.columns]
    df = df[pd.to_numeric(df["Year"], errors="coerce").isin(YEARS)].copy()
    df["Year"] = df["Year"].astype(int)
    return df.set_index("Year")


def load_ras_collisions() -> pd.DataFrame:
    """RAS0101 'Collisions' sheet: Fatal, FSC (unadjusted), FSC (adjusted), All collisions."""
    df = _read_sheet("Collisions", header_row=5)
    return df.rename(
        columns={
            "Fatal": "ras_fatal",
            "FSC (unadjusted)": "ras_ksi_raw",
            "FSC (adjusted)": "ras_ksi_adj",
            "All collisions": "ras_collisions",
        }
    )[["ras_collisions", "ras_fatal", "ras_ksi_raw", "ras_ksi_adj"]]


RAS_VEHICLE_COLUMNS = {
    "Pedal cycles": "pedal_cycle",
    "Motorcycles": "motorcycle",
    "Cars": "car",
    "Buses or coaches": "bus_coach",
    "Light goods vehicles": "light_goods",
    "Heavy goods vehicles": "heavy_goods",
    "Other vehicles": "other",
    "Unknown vehicles": "unknown",
}


def load_ras_vehicles() -> pd.DataFrame:
    """RAS0101 'Vehicles' sheet: vehicles involved by group and year."""
    df = _read_sheet("Vehicles", header_row=3)
    return df.rename(columns=RAS_VEHICLE_COLUMNS)[list(RAS_VEHICLE_COLUMNS.values())]


def compare_vehicles(vehicle: pd.DataFrame) -> pd.DataFrame:
    """Our vehicles per group and year minus RAS0101 (0 everywhere = same grouping)."""
    from src.etl.build import vehicle_group  # local import: build depends on cleaned data only

    groups = pd.Series(vehicle_group(vehicle["vehicle_type"]), index=vehicle.index)
    ours = pd.crosstab(vehicle["collision_year"].astype(int), groups)
    ras = load_ras_vehicles()
    return ours.reindex(columns=ras.columns, fill_value=0) - ras


def compare(collision: pd.DataFrame) -> pd.DataFrame:
    """Yearly counts from our cleaned collision table vs RAS0101, with differences."""
    ours = collision.groupby("collision_year").agg(
        collisions=("collision_index", "size"),
        fatal=("collision_severity", lambda s: int((s.astype(str) == "Fatal").sum())),
        ksi_raw=("ksi_raw", "sum"),
        ksi_adj=("ksi_adj", "sum"),
    )
    ours.index = ours.index.astype(int)
    out = ours.join(load_ras_collisions(), how="outer")
    for col in ("collisions", "fatal", "ksi_raw", "ksi_adj"):
        out[f"diff_{col}"] = out[col] - out[f"ras_{col}"]
    return out


def main() -> None:
    collision = pd.read_parquet(
        PROCESSED_DIR / "collision.parquet",
        columns=["collision_index", "collision_year", "collision_severity", "ksi_raw", "ksi_adj"],
    )
    result = compare(collision)
    result.to_csv(PROCESSED_DIR / "ras_check.csv")
    pd.set_option("display.width", 200)
    print(result.round(1).to_string())

    vehicle = pd.read_parquet(
        PROCESSED_DIR / "vehicle.parquet", columns=["collision_year", "vehicle_type"]
    )
    diff = compare_vehicles(vehicle)
    diff.to_csv(PROCESSED_DIR / "ras_check_vehicles.csv")
    print("\nVehicles per group, ours minus RAS0101:")
    print(diff.to_string())


if __name__ == "__main__":
    main()
