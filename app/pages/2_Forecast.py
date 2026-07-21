"""It shows the probabilistic price forecast — the page that proves the model works.

It plot the P50 central forecast, the shaded P10-P90 uncertainty band, and the
actual prices on top so a viewer can instantly see how well the model tracks
reality and how honest its uncertainty is. It also shows the headline metrics:
MAE and interval coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# === CONFIGURATION ===

REPORTS = Path("reports")
PREDS_PATH = REPORTS / "lgbm_valid_predictions.parquet"
METRICS_PATH = REPORTS / "lgbm_metrics.json"

st.set_page_config(page_title="Forecast", page_icon="📈", layout="wide")
st.title(" Probabilistic price forecast")
st.caption("LightGBM quantile regression — P10 / P50 / P90 on the validation set")

if not PREDS_PATH.exists():
    st.warning(
        "No forecast found yet. Run the model first:\n\n"
        "`python -m src.build_features`\n\n"
        "`python -m src.split_data`\n\n"
        "`python -m src.train_lightgbm`\n\n"
        "Then refresh this page."
    )
    st.stop()

preds = pd.read_parquet(PREDS_PATH).sort_values("datetime")
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

# --- Metric cards ----------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("P50 forecast error (MAE)", f"{metrics.get('mae_p50', float('nan')):.2f} €/MWh")
cov = metrics.get("coverage_80", float("nan")) * 100
c2.metric("80% interval coverage", f"{cov:.1f}%", help="Target is 80%. Higher = intervals too wide; lower = overconfident.")
c3.metric("Validation hours", f"{metrics.get('n_valid', 0):,}")

# --- Date window slider -----------------------------------------------------
days = st.sidebar.slider("Days to display", 2, 14, 7)
cutoff = preds.datetime.max() - pd.Timedelta(days=days)
view = preds[preds.datetime >= cutoff]

# --- Forecast chart with uncertainty band ----------------------------------
st.subheader("Forecast vs actual with 80% uncertainty band")
fig = go.Figure()

# Shaded P10-P90 band. I draw the upper bound, then the lower bound filled up to it.
fig.add_trace(go.Scatter(
    x=view.datetime, y=view["q_0.9"],
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=view.datetime, y=view["q_0.1"],
    fill="tonexty", fillcolor="rgba(31,119,180,0.18)",
    line=dict(width=0), name="P10–P90 (80% interval)", hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=view.datetime, y=view["q_0.5"],
    name="P50 forecast", line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    x=view.datetime, y=view["y_true"],
    name="Actual price", line=dict(color="#d62728", width=1.5, dash="dot"),
))
fig.update_layout(
    xaxis_title="Time (Europe/Berlin)", yaxis_title="Price (EUR/MWh)",
    hovermode="x unified", template="plotly_white", hoverlabel=dict(bgcolor="#0e1117", bordercolor="#666", font=dict(color="white", size=13)), height=460,
    legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The blue line is the model's central (P50) forecast. The shaded band is "
    "the 80% prediction interval: in a well-calibrated model, the red actual "
    "line stays inside the band about 80% of the time. When the band is wide, "
    "the model is telling you the hour is genuinely hard to predict."
)
