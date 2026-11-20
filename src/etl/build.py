"""Build analysis tables from the cleaned STATS19 tables (step 1.4; unit of analysis: D-005).

Usage:
    python -m src.etl.build

Outputs (data/processed/):
    collision_analysis.parquet  one row per collision + features aggregated from vehicles/casualties
    casualty_analysis.parquet   one row per casualty + their vehicle group + collision context
    build_report.json           row counts and join checks
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from src.config import PROCESSED_DIR

# Vehicle groups as in DfT table RAS0101 (sheet "Vehicles"). Verified 2026-11-20: yearly counts per
# group match RAS0101 exactly for 2021-2025 (src/etl/ras_check.py). Minibuses count as cars.
VEHICLE_GROUPS: dict[str, list[str]] = {
    "pedal_cycle": ["Pedal cycle"],
    "motorcycle": [
        "Motorcycle 50cc and under",
        "Motorcycle 125cc and under",
        "Motorcycle over 125cc and up to 500cc",
        "Motorcycle over 500cc",
        "Motorcycle - unknown cc",
        "Electric motorcycle",
    ],
    "car": ["Car", "Taxi/Private hire car", "Minibus (8 - 16 passenger seats)"],
    "bus_coach": ["Bus or coach (17 or more pass seats)"],
    "light_goods": ["Van / Goods 3.5 tonnes mgw or under"],
    "heavy_goods": ["Goods over 3.5t. and under 7.5t", "Goods 7.5 tonnes mgw and over"],
    "other": [
        "Personal powered transporter",
        "Mobility scooter",
        "Ridden horse",
        "Agricultural vehicle",
        "Tram",
        "Other vehicle",
        "Goods vehicle - unknown weight",
    ],
}
VEHICLE_GROUP_ORDER = list(VEHICLE_GROUPS) + ["unknown"]

# Age thresholds used in DfT reporting: young drivers 17-24, older drivers 70+, children 0-15.
YOUNG_DRIVER = (17, 24)
OLDER_AGE = 70
CHILD_MAX_AGE = 15

# Collision columns copied onto the casualty-level table as context.
COLLISION_CONTEXT = [
    "collision_index",
    "collision_year",
    "date",
    "month",
    "hour",
    "day_of_week",
    "country",
    "police_force",
    "local_authority_ons_district",
    "speed_limit",
    "road_type",
    "urban_or_rural_area",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "junction_detail",
    "collision_injury_based",
]


def vehicle_group(vehicle_type: pd.Series) -> pd.Series:
    """Map vehicle_type labels to RAS0101 groups; NaN -> 'unknown'. Raises on unmapped labels."""
    lookup = {label: group for group, labels in VEHICLE_GROUPS.items() for label in labels}
    labels = vehicle_type.astype("string")
    unmapped = set(labels.dropna().unique()) - set(lookup)
    if unmapped:
        raise ValueError(f"vehicle_type labels without a group: {sorted(unmapped)}")
    groups = labels.map(lookup).fillna("unknown")
    return pd.Categorical(groups, categories=VEHICLE_GROUP_ORDER)


def aggregate_vehicles(vehicle: pd.DataFrame) -> pd.DataFrame:
    """One row per collision: vehicle count, presence of each vehicle group, driver ages."""
    v = vehicle[["collision_index", "vehicle_type", "age_of_driver"]].copy()
    v["group"] = vehicle_group(v["vehicle_type"])
    age = v["age_of_driver"]
    motor = ~v["group"].isin(["pedal_cycle", "unknown"]) & ~v["vehicle_type"].astype("string").isin(
        ["Ridden horse", "Mobility scooter"]
    )
    v["young_driver"] = motor & age.between(*YOUNG_DRIVER)
    v["older_driver"] = motor & (age >= OLDER_AGE)

    flags = pd.crosstab(v["collision_index"], v["group"]).reindex(
        columns=VEHICLE_GROUP_ORDER, fill_value=0
    )
    flags = (flags > 0).add_prefix("has_")
    g = v.groupby("collision_index")
    agg = pd.DataFrame(
        {
            "n_vehicles_recorded": g.size(),
            "has_escooter": g["vehicle_type"].agg(
                lambda s: bool((s.astype("string") == "Personal powered transporter").any())
            ),
            "min_driver_age": g["age_of_driver"].min(),
            "max_driver_age": g["age_of_driver"].max(),
            "has_young_driver": g["young_driver"].any(),
            "has_older_driver": g["older_driver"].any(),
        }
    )
    return agg.join(flags)


def aggregate_casualties(casualty: pd.DataFrame) -> pd.DataFrame:
    """One row per collision: casualty count and who was injured (pedestrian, cyclist, child...)."""
    c = casualty[["collision_index", "casualty_class", "casualty_type", "age_of_casualty"]].copy()
    ctype = c["casualty_type"].astype("string")
    c["is_pedestrian"] = c["casualty_class"].astype("string") == "Pedestrian"
    c["is_cyclist"] = ctype == "Cyclist"
    c["is_motorcyclist"] = ctype.str.contains("otorcycle", na=False)
    c["is_child"] = c["age_of_casualty"] <= CHILD_MAX_AGE
    c["is_older"] = c["age_of_casualty"] >= OLDER_AGE
    g = c.groupby("collision_index")
    return pd.DataFrame(
        {
            "n_casualties_recorded": g.size(),
            "has_pedestrian_casualty": g["is_pedestrian"].any(),
            "has_cyclist_casualty": g["is_cyclist"].any(),
            "has_motorcyclist_casualty": g["is_motorcyclist"].any(),
            "has_child_casualty": g["is_child"].any(),
            "has_older_casualty": g["is_older"].any(),
            "min_casualty_age": g["age_of_casualty"].min(),
            "max_casualty_age": g["age_of_casualty"].max(),
        }
    )


def build_collision_table(
    collision: pd.DataFrame, vehicle: pd.DataFrame, casualty: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Join collision-level aggregates onto the collision table, one row per collision."""
    if not collision["collision_index"].is_unique:
        raise ValueError("collision_index is not unique in the collision table")
    veh = aggregate_vehicles(vehicle).reset_index()
    cas = aggregate_casualties(casualty).reset_index()
    out = collision.merge(veh, on="collision_index", how="left", validate="one_to_one")
    out = out.merge(cas, on="collision_index", how="left", validate="one_to_one")
    if len(out) != len(collision):
        raise ValueError(f"row count changed in join: {len(collision)} -> {len(out)}")

    known = set(collision["collision_index"])
    report = {
        "rows_before": len(collision),
        "rows_after": len(out),
        "collisions_without_vehicles": int(out["n_vehicles_recorded"].isna().sum()),
        "collisions_without_casualties": int(out["n_casualties_recorded"].isna().sum()),
        "orphan_vehicles": int((~vehicle["collision_index"].isin(known)).sum()),
        "orphan_casualties": int((~casualty["collision_index"].isin(known)).sum()),
        "vehicle_count_mismatch": int(
            (out["n_vehicles_recorded"] != out["number_of_vehicles"]).sum()
        ),
        "casualty_count_mismatch": int(
            (out["n_casualties_recorded"] != out["number_of_casualties"]).sum()
        ),
    }
    return out, report


def build_casualty_table(
    collision: pd.DataFrame, vehicle: pd.DataFrame, casualty: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Casualty rows + group of the vehicle they were in/hit by + collision context."""
    veh = vehicle[["collision_index", "vehicle_reference", "vehicle_type"]].copy()
    veh["casualty_vehicle_group"] = vehicle_group(veh["vehicle_type"])
    veh = veh.drop(columns="vehicle_type")
    out = casualty.merge(
        veh, on=["collision_index", "vehicle_reference"], how="left", validate="many_to_one"
    )
    out = out.merge(
        collision[COLLISION_CONTEXT].drop(columns="collision_year"),
        on="collision_index",
        how="left",
        validate="many_to_one",
    )
    if len(out) != len(casualty):
        raise ValueError(f"row count changed in join: {len(casualty)} -> {len(out)}")
    report = {
        "rows_before": len(casualty),
        "rows_after": len(out),
        "casualties_without_vehicle_match": int(out["casualty_vehicle_group"].isna().sum()),
        "casualties_without_collision_match": int(out["country"].isna().sum()),
    }
    return out, report


def main() -> None:
    collision = pd.read_parquet(PROCESSED_DIR / "collision.parquet")
    vehicle = pd.read_parquet(PROCESSED_DIR / "vehicle.parquet")
    casualty = pd.read_parquet(PROCESSED_DIR / "casualty.parquet")

    coll_out, coll_rep = build_collision_table(collision, vehicle, casualty)
    cas_out, cas_rep = build_casualty_table(collision, vehicle, casualty)
    coll_out.to_parquet(PROCESSED_DIR / "collision_analysis.parquet", index=False)
    cas_out.to_parquet(PROCESSED_DIR / "casualty_analysis.parquet", index=False)

    report = {
        "collision_analysis": coll_rep,
        "casualty_analysis": cas_rep,
        "_meta": {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "unit_of_analysis": "docs/decisions.md D-005",
        },
    }
    (PROCESSED_DIR / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
