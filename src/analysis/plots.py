"""Shared matplotlib helpers so every static figure looks the same and carries a data note."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito palette (colour-blind safe). KSI is always vermillion.
KSI_COLOR = "#D55E00"
RAW_COLOR = "#999999"
SERIES_COLORS = {"England": "#0072B2", "Wales": "#D55E00", "Scotland": "#009E73"}
REF_COLOR = "#555555"


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
    label_width: int = 38,
) -> plt.Figure:
    """Horizontal bars of KSI rate with 95% CI; small groups hatched; overall rate as a line."""
    t = table.copy()
    labels = [str(i if not isinstance(i, tuple) else " · ".join(map(str, i))) for i in t.index]
    labels = [lbl if len(lbl) <= label_width else lbl[: label_width - 1] + "…" for lbl in labels]
    y = range(len(t))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.4))
    err = [(t["rate"] - t["ci_low"]) * 100, (t["ci_high"] - t["rate"]) * 100]
    bars = ax.barh(y, t["rate"] * 100, xerr=err, color=KSI_COLOR, alpha=0.85, capsize=2)
    for bar, small in zip(bars, t["small_n"]):
        if small:
            bar.set_hatch("///")
            bar.set_alpha(0.35)
    for yi, (rate, n) in enumerate(zip(t["rate"], t["n"])):
        ax.text(rate * 100 + 0.6, yi, f"{rate:.1%}  (n={n:,})", va="center", fontsize=8)
    if overall is not None:
        ax.axvline(overall * 100, color=REF_COLOR, ls="--", lw=1)
        ax.text(overall * 100, -0.8, f"chung {overall:.1%}", fontsize=8, color=REF_COLOR)
    ax.set_yticks(list(y), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Tỷ lệ KSI (%), thanh lỗi = khoảng tin cậy 95%")
    ax.set_xlim(0, max(t["ci_high"].max() * 100 * 1.25, 10))
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
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
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
    for col in frame.columns:
        color = (colors or {}).get(str(col))
        ax.plot(
            frame.index,
            frame[col] * (100 if pct else 1),
            marker="o",
            ms=3,
            label=str(col),
            color=color,
        )
    if vline is not None:
        ax.axvline(vline[0], color=REF_COLOR, ls="--", lw=1)
        ax.text(vline[0], ax.get_ylim()[1], f" {vline[1]}", fontsize=8, va="top", color=REF_COLOR)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, note, path)
