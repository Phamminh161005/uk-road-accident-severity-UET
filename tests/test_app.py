"""Smoke tests: every Streamlit page renders without exceptions (default and extreme filters)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
AGG_FILE = APP_DIR.parent / "data" / "processed" / "agg" / "overview_monthly.parquet"

pytestmark = pytest.mark.skipif(not AGG_FILE.exists(), reason="run python -m src.etl.aggregate")


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(str(APP_DIR / "Home.py"), default_timeout=30)
    at.run()
    if page != "home":
        at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, at.exception
    return at


def test_home_renders():
    at = run_page("home")
    assert at.title[0].value.startswith("Mức độ nghiêm trọng")


def test_overview_default_filters():
    at = run_page("overview")
    assert at.metric[0].value == "513.801"
    assert len(at.get("plotly_chart")) == 1


def test_overview_empty_filter_shows_warning():
    at = run_page("overview")
    at.sidebar.multiselect[1].set_value([]).run()
    assert not at.exception and len(at.warning) == 1


def test_overview_single_country_and_switch_metric():
    at = run_page("overview")
    at.sidebar.multiselect[1].set_value(["Wales"]).run()
    at.radio[0].set_value("Số vụ").run()
    assert not at.exception
