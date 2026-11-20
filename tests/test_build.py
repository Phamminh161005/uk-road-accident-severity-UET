"""Tests for src.etl.build: aggregation and joins must never duplicate or drop rows."""

import pandas as pd
import pytest

from src.etl import build


@pytest.fixture
def tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Collision X: car (45) + motorcycle (19); casualties: car driver slight, rider + pedestrian.
    # Collision Y: one car (30), one casualty.
    collision = pd.DataFrame(
        {
            "collision_index": ["X", "Y"],
            "number_of_vehicles": [2, 1],
            "number_of_casualties": [3, 1],
            "ksi_adj": [1.0, 0.0],
            **{c: [None, None] for c in build.COLLISION_CONTEXT if c != "collision_index"},
        }
    )
    collision["country"] = ["England", "Wales"]
    vehicle = pd.DataFrame(
        {
            "collision_index": ["X", "X", "Y"],
            "vehicle_reference": [1, 2, 1],
            "vehicle_type": ["Car", "Motorcycle 125cc and under", "Car"],
            "age_of_driver": [45.0, 19.0, 30.0],
        }
    )
    casualty = pd.DataFrame(
        {
            "collision_index": ["X", "X", "X", "Y"],
            "vehicle_reference": [1, 2, 1, 1],
            "casualty_reference": [1, 2, 3, 1],
            "casualty_class": [
                "Driver or rider",
                "Driver or rider",
                "Pedestrian",
                "Driver or rider",
            ],
            "casualty_type": [
                "Car occupant",
                "Motorcycle 125cc and under rider or passenger",
                "Pedestrian",
                "Car occupant",
            ],
            "age_of_casualty": [45.0, 19.0, 72.0, 30.0],
        }
    )
    return collision, vehicle, casualty


def test_vehicle_group_maps_and_rejects_unknown_labels():
    out = build.vehicle_group(pd.Series(["Minibus (8 - 16 passenger seats)", None]))
    assert list(out) == ["car", "unknown"]
    with pytest.raises(ValueError, match="without a group"):
        build.vehicle_group(pd.Series(["Hovercraft"]))


def test_collision_table_one_row_per_collision(tables):
    out, rep = build.build_collision_table(*tables)
    assert len(out) == 2 and rep["rows_before"] == rep["rows_after"] == 2
    assert rep["vehicle_count_mismatch"] == 0 and rep["casualty_count_mismatch"] == 0
    x = out.set_index("collision_index").loc["X"]
    assert x["has_car"] and x["has_motorcycle"] and not x["has_heavy_goods"]
    assert x["min_driver_age"] == 19 and x["has_young_driver"]
    assert x["has_pedestrian_casualty"] and x["has_older_casualty"]
    assert x["has_motorcyclist_casualty"] and not x["has_cyclist_casualty"]
    y = out.set_index("collision_index").loc["Y"]
    assert not y["has_motorcycle"] and not y["has_young_driver"]


def test_collision_table_rejects_duplicate_collisions(tables):
    collision, vehicle, casualty = tables
    with pytest.raises(ValueError, match="not unique"):
        build.build_collision_table(pd.concat([collision, collision.iloc[[0]]]), vehicle, casualty)


def test_casualty_table_keeps_rows_and_adds_context(tables):
    out, rep = build.build_casualty_table(*tables)
    assert len(out) == 4 and rep["casualties_without_vehicle_match"] == 0
    assert out["casualty_vehicle_group"].astype(str).tolist() == ["car", "motorcycle", "car", "car"]
    assert out["country"].tolist() == ["England", "England", "England", "Wales"]


def test_casualty_table_rejects_duplicate_vehicle_keys(tables):
    collision, vehicle, casualty = tables
    with pytest.raises(pd.errors.MergeError):
        build.build_casualty_table(collision, pd.concat([vehicle, vehicle.iloc[[0]]]), casualty)
