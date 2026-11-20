"""Shared colours and text helpers so every page looks the same.

Colours mirror src/analysis/plots.py (Okabe-Ito, colour-blind safe; validated for light mode).
"""

from __future__ import annotations

KSI_COLOR = "#D55E00"
COUNTRY_COLORS = {"England": "#0072B2", "Wales": "#D55E00", "Scotland": "#009E73"}
COUNTRY_ORDER = ["England", "Wales", "Scotland"]
GRID_COLOR = "rgba(128,128,128,0.2)"


def fmt_int(x: float) -> str:
    """Vietnamese thousands separator: 513801 -> '513.801'."""
    return f"{x:,.0f}".replace(",", ".")


def fmt_pct(x: float, digits: int = 1) -> str:
    """0.2528 -> '25,3%'."""
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def data_note(n: int, years: str, filters: str, unit: str = "vụ") -> str:
    """Footnote required on every chart: years, filters, unit, severity, N."""
    return (
        f"Dữ liệu: STATS19 {years} · Lọc: {filters} · Đơn vị: {unit} · "
        f"KSI đã điều chỉnh (DfT) · N = {fmt_int(n)}"
    )
