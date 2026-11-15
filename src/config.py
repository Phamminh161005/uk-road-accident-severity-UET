"""Central project configuration: paths, years, seeds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AGG_DIR = PROCESSED_DIR / "agg"
REFERENCE_DIR = DATA_DIR / "reference"
REPORTS_DIR = ROOT / "reports"

# Study period (see docs/decisions.md D-001)
YEARS = list(range(2021, 2026))
MIDAS_YEARS = list(range(2021, 2025))

# Wales default 20 mph policy start date (RQ4)
POLICY_DATE = "2023-09-17"

RANDOM_SEED = 42
