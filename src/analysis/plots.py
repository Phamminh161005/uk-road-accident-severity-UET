"""Shared matplotlib helpers so every static figure looks the same and carries a data note."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

# Okabe-Ito palette (colour-blind safe). KSI is always vermillion.
KSI_COLOR = "#D55E00"
RAW_COLOR = "#999999"
SERIES_COLORS = {"England": "#0072B2", "Wales": "#D55E00", "Scotland": "#009E73"}
REF_COLOR = "#555555"
# Fallback order for other series; markers add a second cue because some pairs are close under CVD.
LINE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#000000"]
LINE_MARKERS = ["o", "s", "^", "D", "v", "P"]


def data_note(
    n: int,
    unit: str = "vụ",
    years: str = "2021–2025",
    filters: str = "không",
    severity: str = "KSI đã điều chỉnh (DfT)",
) -> str:
    """Standard footnote required by the project rules (years, filter, unit, severity, N)."""
    return f"Dữ liệu: STATS19 {years} · Lọc: {filters} · Đơn vị: {unit} · {severity} · N = {n:,}"


def _finish(fig: plt.Figure, note: str, path: Path | None) -> plt.Figure:
    fig.text(0.01, 0.005, note, fontsize=7.5, color=REF_COLOR, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def rate_bars(
    table: pd.DataFrame,
    title: str,
    note: str,
    overall: float | None = None,
    path: Path | None = None,
    label_width: int = 34,
) -> plt.Figure:
    """Horizontal bars of KSI rate with 95% CI; small groups hatched; overall rate as a line."""
    t = table.copy()
    labels = [str(i if not isinstance(i, tuple) else " · ".join(map(str, i))) for i in t.index]
    # Wrap long labels onto two lines instead of truncating them.
    labels = [textwrap.fill(lbl, label_width, max_lines=2, placeholder="…") for lbl in labels]
    wrapped = any("\n" in lbl for lbl in labels)
    y = range(len(t))
    row_h = 0.55 if wrapped else 0.42
    fig, ax = plt.subplots(figsize=(8, row_h * len(t) + 1.6))
    err = [(t["rate"] - t["ci_low"]) * 100, (t["ci_high"] - t["rate"]) * 100]
    bars = ax.barh(y, t["rate"] * 100, xerr=err, color=KSI_COLOR, alpha=0.85, capsize=2)
    for bar, small in zip(bars, t["small_n"]):
        if small:
            bar.set_hatch("///")
            bar.set_alpha(0.35)
    xmax = max(t["ci_high"].max() * 100, 10)
    pad = xmax * 0.015
    # Label starts after the CI whisker; white box keeps the reference line from crossing the text.
    for yi, (rate, hi, n) in enumerate(zip(t["rate"], t["ci_high"], t["n"])):
        ax.text(
            hi * 100 + pad,
            yi,
            f"{rate:.1%}  (n={n:,})",
            va="center",
            fontsize=8,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9},
        )
    # Extra headroom above the first bar holds the reference-line label.
    ax.set_ylim(len(t) - 0.5, -1.3)
    if overall is not None:
        ax.axvline(overall * 100, color=REF_COLOR, ls="--", lw=1, zorder=1)
        ax.text(
            overall * 100 + pad,
            -0.95,
            f"chung {overall:.1%}",
            fontsize=8,
            color=REF_COLOR,
            va="center",
        )
    ax.set_yticks(list(y), labels, fontsize=9)
    ax.set_xlabel("Tỷ lệ KSI (%), thanh lỗi = khoảng tin cậy 95%")
    ax.set_xlim(0, xmax * 1.4)
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, note, path)


def heatmap(
    pivot: pd.DataFrame,
    title: str,
    note: str,
    fmt: str = "{:.0%}",
    cbar_label: str = "Tỷ lệ KSI",
    path: Path | None = None,
    cmap: str = "Oranges",
) -> plt.Figure:
    """Annotated heatmap (e.g. hour x weekday KSI rate)."""
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * pivot.shape[1] + 2), 0.45 * pivot.shape[0] + 1.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), [str(r) for r in pivot.index], fontsize=8)
    ax.set_xlabel(pivot.columns.name or "")
    ax.set_ylabel(pivot.index.name or "")
    if pivot.size <= 300:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.notna(v):
                    # White text on dark cells, black on light cells (by cell luminance).
                    r, g, b, _ = im.cmap(im.norm(v))
                    dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45
                    ax.text(
                        j,
                        i,
                        fmt.format(v),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if dark else "black",
                    )
    cbar = fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    if "%" in fmt:
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title, fontsize=11, loc="left")
    return _finish(fig, note, path)


def lines(
    frame: pd.DataFrame,
    title: str,
    note: str,
    ylabel: str,
    colors: dict[str, str] | None = None,
    vline: tuple[pd.Timestamp, str] | None = None,
    pct: bool = True,
    path: Path | None = None,
) -> plt.Figure:
    """Line chart, one line per column of ``frame`` (index = x)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, col in enumerate(frame.columns):
        color = (colors or {}).get(str(col), LINE_COLORS[k % len(LINE_COLORS)])
        ax.plot(
            frame.index,
            frame[col] * (100 if pct else 1),
            marker=LINE_MARKERS[k % len(LINE_MARKERS)],
            ms=3.5,
            label=str(col),
            color=color,
        )
    if pd.api.types.is_integer_dtype(frame.index):
        # Whole years/months only: no 2021.5 ticks.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if vline is not None:
        ax.axvline(vline[0], color=REF_COLOR, ls="--", lw=1)
        ax.text(vline[0], ax.get_ylim()[1], f" {vline[1]}", fontsize=8, va="top", color=REF_COLOR)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, note, path)
