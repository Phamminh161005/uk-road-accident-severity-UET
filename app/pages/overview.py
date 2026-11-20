"""Overview page: headline counts, KSI rate and monthly trend (collision level)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components.data import load_overview
from components.style import (
    COUNTRY_COLORS,
    COUNTRY_ORDER,
    GRID_COLOR,
    data_note,
    fmt_int,
    fmt_pct,
)

POLICY_DATE = pd.Timestamp("2023-09-17")
METRICS = {"Tỷ lệ KSI": "rate", "Số vụ": "collisions"}

df = load_overview()
all_years = sorted(df["collision_year"].unique().tolist())
countries = [c for c in COUNTRY_ORDER if c in set(df["country"])]

st.title("Tổng quan")

with st.sidebar:
    st.header("Bộ lọc")
    years = st.multiselect("Năm", all_years, default=all_years)
    sel_countries = st.multiselect("Quốc gia", countries, default=countries)

sub = df[df["collision_year"].isin(years) & df["country"].isin(sel_countries)]
if sub.empty:
    st.warning("Không có dữ liệu cho bộ lọc đã chọn. Hãy chọn ít nhất một năm và một quốc gia.")
    st.stop()

years = sorted(years)
contiguous = years == list(range(years[0], years[-1] + 1))
if len(years) == 1:
    years_label = str(years[0])
elif contiguous:
    years_label = f"{years[0]}–{years[-1]}"
else:
    years_label = ", ".join(map(str, years))
filters_label = "không" if len(sel_countries) == len(countries) else ", ".join(sel_countries)

n = int(sub["collisions"].sum())
rate = sub["ksi_adj"].sum() / n

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số vụ tai nạn", fmt_int(n))
c2.metric("Số người thương vong", fmt_int(sub["casualties"].sum()))
c3.metric("Số vụ có người tử vong", fmt_int(sub["fatal"].sum()))
c4.metric("Tỷ lệ KSI", fmt_pct(rate))

# Monthly trend per country
monthly = (
    sub.groupby(["country", "collision_year", "month"], as_index=False)[["collisions", "ksi_adj"]]
    .sum()
    .assign(
        rate=lambda d: d["ksi_adj"] / d["collisions"],
        period=lambda d: pd.to_datetime(
            {"year": d["collision_year"], "month": d["month"], "day": 1}
        ),
    )
    .sort_values("period")
)

metric_label = st.radio("Chỉ số", list(METRICS), horizontal=True)
metric = METRICS[metric_label]

fig = go.Figure()
for country in [c for c in COUNTRY_ORDER if c in sel_countries]:
    d = monthly[monthly["country"] == country]
    fig.add_trace(
        go.Scatter(
            x=d["period"],
            y=d[metric],
            name=country,
            mode="lines",
            line={"color": COUNTRY_COLORS[country], "width": 2},
            customdata=list(zip(d["collisions"], d["rate"] * 100)),
            hovertemplate=(
                f"<b>{country}</b> · %{{x|%m/%Y}}<br>"
                "Số vụ: %{customdata[0]:,}<br>"
                "Tỷ lệ KSI: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )
    )
if "Wales" in sel_countries and monthly["period"].min() <= POLICY_DATE <= monthly["period"].max():
    fig.add_vline(x=POLICY_DATE, line={"color": "gray", "width": 1, "dash": "dot"})
    fig.add_annotation(
        x=POLICY_DATE,
        y=1,
        yref="paper",
        text="Wales áp dụng 20 mph (17/9/2023)",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "gray"},
    )
fig.update_layout(
    title=f"{metric_label} theo tháng",
    hovermode="x unified" if metric == "collisions" else "closest",
    height=420,
    margin={"l": 10, "r": 10, "t": 60, "b": 10},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    xaxis={"title": "Tháng", "gridcolor": GRID_COLOR},
    yaxis={
        "title": "Tỷ lệ KSI" if metric == "rate" else "Số vụ",
        "tickformat": ".0%" if metric == "rate" else ",",
        "gridcolor": GRID_COLOR,
        "rangemode": "tozero",
    },
)
st.plotly_chart(fig, width="stretch")
st.caption(data_note(n, years_label, filters_label))

# Key observation, computed from the current filter
by_country = sub.groupby("country")[["collisions", "ksi_adj"]].sum()
by_country["rate"] = by_country["ksi_adj"] / by_country["collisions"]
if len(by_country) > 1:
    hi = by_country["rate"].idxmax()
    lo = by_country["rate"].idxmin()
    st.markdown(
        f"Trong giai đoạn đã chọn, tỷ lệ KSI chung là {fmt_pct(rate)}. {hi} có tỷ lệ cao nhất "
        f"({fmt_pct(by_country.loc[hi, 'rate'])}), {lo} thấp nhất "
        f"({fmt_pct(by_country.loc[lo, 'rate'])}). Số liệu theo tháng của Wales và Scotland "
        "dựa trên ít vụ hơn nhiều so với England nên dao động mạnh hơn."
    )
else:
    st.markdown(f"Trong giai đoạn đã chọn, tỷ lệ KSI là {fmt_pct(rate)}.")

with st.expander("Bảng số liệu theo năm"):
    yearly = (
        sub.groupby(["collision_year", "country"], as_index=False)[
            ["collisions", "casualties", "fatal", "ksi_adj"]
        ]
        .sum()
        .assign(rate=lambda d: d["ksi_adj"] / d["collisions"])
    )
    st.dataframe(
        yearly,
        hide_index=True,
        width="stretch",
        column_config={
            "collision_year": st.column_config.NumberColumn("Năm", format="%d"),
            "country": "Quốc gia",
            "collisions": st.column_config.NumberColumn("Số vụ", format="localized"),
            "casualties": st.column_config.NumberColumn("Thương vong", format="localized"),
            "fatal": st.column_config.NumberColumn("Vụ tử vong", format="localized"),
            "ksi_adj": st.column_config.NumberColumn("Số vụ KSI (ước tính)", format="%.0f"),
            "rate": st.column_config.NumberColumn("Tỷ lệ KSI", format="percent"),
        },
    )
