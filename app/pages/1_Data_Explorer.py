""" the interactive data explorer (EDA) page.

It lets the analyst freely explore the full dataset: choose any date range and 
pick any features to plot together. It views the patterns that drive prices.
This is where we look for daily/weekly seasonality, the relationship between
residual load and price, negative-price events, and volatility — the things
 must understand before trusting any model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# === CONFIGURATION ===

PROCESSED = Path("data/processed")
DATASET = PROCESSED / "model_dataset.parquet"

st.set_page_config(page_title="Data Explorer", page_icon="🔎", layout="wide")
st.title(" Data explorer (EDA)")
st.caption("Explore the full dataset freely — pick dates, pick features, find patterns")

if not DATASET.exists():
    st.warning(
        "No dataset found. Build it first:\n\n"
        "`python -m src.build_features`\n\nThen refresh."
    )
    st.stop()


@st.cache_data(ttl=600)
def load() -> pd.DataFrame:
    """It loads the full feature dataset once and cache it."""
    return pd.read_parquet(DATASET).sort_values("datetime").reset_index(drop=True)


df = load()

# Friendly labels for the columns most useful to explore.
NICE = {
    "intraday_price_eur_mwh": "Intraday price (€/MWh)",
    "da_price_eur_mwh": "Day-ahead price (€/MWh)",
    "id_da_spread_lag1": "Intraday − Day-ahead spread (€/MWh)",
    "wind_total_mw": "Wind total (MW)",
    "solar_mw": "Solar (MW)",
    "residual_load_mw": "Residual load (MW)",
    "total_load_mw": "Total load (MW)",
    "renewable_share": "Renewable share (0–1)",
}
available = {k: v for k, v in NICE.items() if k in df.columns}

# === SIDEBAR CONTROLS ===

st.sidebar.header("Controls")

dmin = df.datetime.min().date()
dmax = df.datetime.max().date()

# Quick presets so the full 3-year history is one click away instead of
# paging the calendar widget back month by month.
preset = st.sidebar.radio(
    "Quick range",
    ["Last 30 days", "Last 6 months", "Last year", "Full history", "Custom"],
    index=0,
)
PRESET_DAYS = {"Last 30 days": 30, "Last 6 months": 182, "Last year": 365}

if preset == "Custom":
    date_range = st.sidebar.date_input(
        "Date range", value=(dmax - pd.Timedelta(days=30), dmax),
        min_value=dmin, max_value=dmax,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = dmin, dmax
elif preset == "Full history":
    start, end = dmin, dmax
else:
    start = max(dmin, dmax - pd.Timedelta(days=PRESET_DAYS[preset]))
    end = dmax

mask = (df.datetime.dt.date >= start) & (df.datetime.dt.date <= end)
view = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.write(f"**{len(view):,} hours** selected")
st.sidebar.write(f"{start} → {end}")

# === TABS ===

tab1, tab2, tab3, tab4 = st.tabs(
    [" Time series", " Relationships", " Patterns", " Distribution"]
)

# --- TAB 1: free time-series plot ------------------------------------------
with tab1:
    st.subheader("Plot any features over time")
    picks = st.multiselect(
        "Choose one or more features",
        options=list(available.keys()),
        default=["intraday_price_eur_mwh", "da_price_eur_mwh"],
        format_func=lambda k: available[k],
    )
    if picks:
        fig = go.Figure()
        for col in picks:
            fig.add_trace(go.Scatter(x=view.datetime, y=view[col], name=available[col], mode="lines"))
        fig.update_layout(
            template="plotly_white", hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)), height=460, hovermode="x",
            xaxis_title="Time (Europe/Berlin)", yaxis_title="Value",
            legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_traces(hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Tip: features on very different scales (price vs MW) will look "
            "flat together. Plot price-like things together, MW-like things together."
        )

# --- TAB 2: scatter relationships ------------------------------------------
with tab2:
    st.subheader("How does price relate to a driver?")
    st.caption("The merit order predicts: higher residual load → higher price.")
    c1, c2 = st.columns(2)
    xcol = c1.selectbox(
        "X axis", [k for k in available if k != "intraday_price_eur_mwh"],
        index=0, format_func=lambda k: available[k],
    )
    ycol = c2.selectbox(
        "Y axis", list(available.keys()),
        index=list(available).index("intraday_price_eur_mwh"),
        format_func=lambda k: available[k],
    )
    fig = px.scatter(
        view, x=xcol, y=ycol, opacity=0.45,
        labels={xcol: available[xcol], ycol: available[ycol]},
        trendline="ols", trendline_color_override="#d62728",
    )
    fig.update_layout(template="plotly_white", hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)), height=460)
    fig.update_traces(hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)))
    st.plotly_chart(fig, use_container_width=True)
    if xcol in view and ycol in view:
        corr = view[[xcol, ycol]].corr().iloc[0, 1]
        st.metric("Correlation", f"{corr:+.2f}",
                  help="−1 to +1. Near ±1 = strong linear link; near 0 = weak.")

# --- TAB 3: seasonality patterns -------------------------------------------
with tab3:
    st.subheader("When are prices high or low?")
    price = "intraday_price_eur_mwh"
    v = view.copy()
    v["hour"] = v.datetime.dt.hour
    v["weekday"] = v.datetime.dt.day_name()

    col1, col2 = st.columns(2)
    with col1:
        by_hour = v.groupby("hour")[price].mean().reset_index()
        fig = px.bar(by_hour, x="hour", y=price, labels={price: "Avg price €/MWh", "hour": "Hour of day"})
        fig.update_layout(template="plotly_white", hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)), height=360, title="Average price by hour")
        fig.update_traces(hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Look for the morning ramp (~8h) and evening peak (~19h).")
    with col2:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_dow = v.groupby("weekday")[price].mean().reindex(order).reset_index()
        fig = px.bar(by_dow, x="weekday", y=price, labels={price: "Avg price €/MWh", "weekday": ""})
        fig.update_layout(template="plotly_white", hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)), height=360, title="Average price by weekday")
        fig.update_traces(hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Weekends are usually cheaper — less industrial demand.")

# --- TAB 4: distribution + extremes ----------------------------------------
with tab4:
    st.subheader("Price distribution and extreme events")
    price = "intraday_price_eur_mwh"
    fig = px.histogram(view, x=price, nbins=60, labels={price: "Intraday price €/MWh"})
    fig.update_layout(template="plotly_white", hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)), height=380, title="How often does each price occur?")
    fig.update_traces(hoverlabel=dict(bgcolor="white", bordercolor="black", font=dict(color="black", size=13)))
    st.plotly_chart(fig, use_container_width=True)

    neg = int((view[price] < 0).sum())
    spike = int((view[price] > 150).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Negative-price hours", f"{neg}", help="Renewables exceeded demand — surplus.")
    c2.metric("Spike hours (>€150)", f"{spike}", help="Scarcity — low renewables, high demand.")
    c3.metric("Median price", f"{view[price].median():.1f} €/MWh")
    st.caption(
        "Electricity prices are not a neat bell curve. They have a long right "
        "tail (rare expensive spikes) and a hard floor that goes negative. This "
        "is exactly why a single point forecast is not enough and it needs "
        "probabilistic forecasting."
    )
