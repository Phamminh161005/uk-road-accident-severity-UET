"""Cached loaders for pre-aggregated tables in data/processed/agg/ (built by src.etl.aggregate)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

AGG_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "agg"


@st.cache_data
def load_overview() -> pd.DataFrame:
    """Collision counts per year x month x country."""
    return pd.read_parquet(AGG_DIR / "overview_monthly.parquet")
