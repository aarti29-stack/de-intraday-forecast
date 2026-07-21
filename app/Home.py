"""Run the Streamlit market monitor — page one of the forecasting app.

showS the last days of German intraday and day-ahead prices, the generation
mix, and key market metrics. It automatically uses real SMARD data when it
exists, and fall back to demo data so the app always works on day one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# === CONFIGURATION ===

PROCESSED = Path("data/processed")
REAL_PRICES = PROCESSED / "smard_intraday_hourly.parquet"
REAL_DA = PROCESSED / "smard_day_ahead_hourly.parquet"
REAL_FEATURES = PROCESSED / "smard_features_hourly.parquet"
DEMO_PRICES = PROCESSED / "demo_intraday_hourly.parquet"
DEMO_DA = PROCESSED / "demo_day_ahead_hourly.parquet"
DEMO_FEATURES = PROCESSED / "demo_features_hourly.parquet"

st.set_page_config(
    page_title="DE Intraday Price Monitor",
    page_icon="⚡",
    layout="wide",
)


# === DATA LOADING ===

@st.cache_data(ttl=600)
def load_data() -> tuple[pd.DataFrame, bool]:
    """ loads real SMARD data if present, otherwise demo data.

     merges intraday price, day-ahead price and generation on datetime.
    The boolean tells the page whether to show the demo-mode banner.
    """
    use_real = REAL_PRICES.exists() and REAL_DA.exists() and REAL_FEATURES.exists()
    p_path, da_path, f_path = (
        (REAL_PRICES, REAL_DA, REAL_FEATURES) if use_real
        else (DEMO_PRICES, DEMO_DA, DEMO_FEATURES)
    )
    if not p_path.exists():
        return pd.DataFrame(), False

    prices = pd.read_parquet(p_path)
    da = pd.read_parquet(da_path)
    feats = pd.read_parquet(f_path)
    df = prices.merge(da, on="datetime", how="left").merge(feats, on="datetime", how="left")
    return df.sort_values("datetime").reset_index(drop=True), use_real


df, is_real = load_data()

# === PAGE ===

st.title(" German Intraday Electricity Price Monitor")
st.caption("probabilistic intraday price forecasting, TH Köln")

if df.empty:
    st.error(
        "No data found. Run one of these first:\n\n"
        "`python -m src.make_demo_data` (30 seconds, synthetic)\n\n"
        "`python -m src.fetch_smard` (few minutes, real SMARD data)"
    )
    st.stop()

if not is_real:
    st.warning(
        "**Demo mode** — showing synthetic data. "
        "Run `python -m src.fetch_smard` to load real SMARD market data, "
        "then refresh this page."
    )
else:
    st.success("Live SMARD data (Bundesnetzagentur) — region DE-LU, hourly resolution.")

# I show the most recent 7 days by default; a slider lets the user widen this.
days = st.sidebar.slider("Days to display", 3, 60, 7)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n\n"
    "SMARD.de — intraday & day-ahead prices, generation, load\n\n"
    "Filter 4170 = intraday price (forecast target)"
)
cutoff = df.datetime.max() - pd.Timedelta(days=days)
view = df[df.datetime >= cutoff]

# --- Metric cards -----------------------------------------------------------
latest = view.iloc[-1]
last24 = view.tail(24)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest intraday price", f"{latest.intraday_price_eur_mwh:.1f} €/MWh")
c2.metric(
    "24h average",
    f"{last24.intraday_price_eur_mwh.mean():.1f} €/MWh",
    delta=f"{latest.intraday_price_eur_mwh - last24.intraday_price_eur_mwh.mean():+.1f}",
)
c3.metric(f"{days}d min", f"{view.intraday_price_eur_mwh.min():.1f} €/MWh")
c4.metric(f"{days}d max", f"{view.intraday_price_eur_mwh.max():.1f} €/MWh")

# --- Price chart ------------------------------------------------------------
st.subheader("Day-ahead vs intraday price")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=view.datetime, y=view.da_price_eur_mwh,
    name="Day-ahead", line=dict(color="#1f77b4", width=1.5),
))
fig.add_trace(go.Scatter(
    x=view.datetime, y=view.intraday_price_eur_mwh,
    name="Intraday", line=dict(color="#ff7f0e", width=1.5),
))
fig.update_layout(
    xaxis_title="Time (Europe/Berlin)", yaxis_title="Price (EUR/MWh)",
    hovermode="x unified", template="plotly_dark", hoverlabel=dict(bgcolor="rgba(25,25,25,0.95)", font=dict(color="white")), height=420,
    legend=dict(orientation="h", y=1.08, bgcolor="rgba(25,25,25,0.6)", font=dict(color="white")),
)
st.plotly_chart(fig, use_container_width=True, theme=None)

st.caption(
    "The gap between the two lines is the intraday-vs-day-ahead spread — "
    "the signal this forecasting model learns to predict. Large gaps usually "
    "mean renewable generation came in different than the day-ahead auction expected."
)

# --- Generation chart -------------------------------------------------------
st.subheader("Generation and load")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=view.datetime, y=view.wind_onshore_mw + view.wind_offshore_mw,
    name="Wind total", stackgroup="gen", line=dict(width=0), fillcolor="rgba(31,119,180,0.55)",
))
fig2.add_trace(go.Scatter(
    x=view.datetime, y=view.solar_mw,
    name="Solar", stackgroup="gen", line=dict(width=0), fillcolor="rgba(255,200,50,0.65)",
))
fig2.add_trace(go.Scatter(
    x=view.datetime, y=view.total_load_mw,
    name="Total load", line=dict(color="#d62728", width=1.5, dash="dot"),
))
fig2.update_layout(
    xaxis_title="Time (Europe/Berlin)", yaxis_title="MW",
    hovermode="x unified", template="plotly_dark", hoverlabel=dict(bgcolor="rgba(25,25,25,0.95)", font=dict(color="white")), height=380,
    legend=dict(orientation="h", y=1.08, bgcolor="rgba(25,25,25,0.6)", font=dict(color="white")),
)
st.plotly_chart(fig2, use_container_width=True, theme=None)

st.caption(
    "When wind + solar (the filled area) approaches total load (the dotted line), "
    "residual load is low — and the merit order says prices fall, sometimes below zero."
)
