# German Electricity Price Forecasting (DE Intraday)

Probabilistic forecasting of German intraday electricity prices using LightGBM
quantile regression, with a Streamlit app for exploring the market data and
judging how honest the model's uncertainty really is.

**Independent research project — planned M.Sc. thesis topic, Renewable Energy
Management, TH Köln**

---

## Key finding so far

The first LightGBM quantile model produced prediction intervals that were
**overconfident**: the 80% interval only contained the actual price in
**~65% of hours** on the validation set.

Why this matters: a battery or trading strategy that relies on an "80%
confident" price range that is really only right 63% of the time will make
systematically bad decisions. The next step is fixing this with
**Conformalized Quantile Regression (CQR)** — a calibration method that
adjusts the intervals so the stated confidence matches reality.

## What the app shows

| Page | What it does |
|------|--------------|
|  Home — Market Monitor | Last days of intraday vs. day-ahead prices, generation mix, load |
|  Data Explorer | Free EDA: pick any date range and features, seasonality, price distribution, negative-price and spike hours |
|  Forecast | P50 forecast with shaded P10–P90 band vs. actual prices, plus MAE and interval coverage metrics |

## Quick start (no API key needed)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get data — pick ONE:
python -m src.make_demo_data    # 30 seconds, synthetic demo data
python -m src.fetch_smard       # few minutes, real SMARD market data

# 4. Build features and train the model
python -m src.build_features
python -m src.split_data
python -m src.train_lightgbm

# 5. Launch the app
streamlit run app/Home.py
```

The browser opens at http://localhost:8501.

## Data

All market data comes from [SMARD.de](https://www.smard.de) (Bundesnetzagentur)
— a fully public API, no registration required. The downloader fetches 3 years
of hourly intraday prices, day-ahead prices, wind, solar, load and residual
load for the DE-LU bidding zone.

ENTSO-E Transparency Platform data (requires a free API key, see
`.env.example`) is planned for phase 2.

## Modelling approach

- **Target:** hourly German intraday price (€/MWh)
- **Model:** LightGBM quantile regression (P10 / P50 / P90)
- **Features:** leakage-safe only — lagged prices, calendar features, and
  day-ahead information already known at prediction time
- **Split:** strict time-based 60/20/20 train / validation / test
  (no shuffling — the model never sees the future)
- **Evaluation:** MAE for the central forecast, empirical interval coverage
  for the uncertainty bands

## Project structure

```
src/
├── config.py            reads secrets from .env
├── utils_io.py          parquet save/load helpers + logging setup
├── fetch_smard.py       SMARD downloader (prices, generation, load)
├── make_demo_data.py    synthetic data so the app works on day one
├── build_features.py    leakage-safe feature engineering
├── split_data.py        time-based train/validation/test split
└── train_lightgbm.py    quantile model training + metrics
app/
├── Home.py              market monitor (entrypoint)
└── pages/
    ├── 1_Data_Explorer.py
    └── 2_Forecast.py
data/
├── raw/                 untouched API downloads (audit trail)
└── processed/           cleaned files the app and models read
reports/                 model predictions and metrics
```

## Roadmap

- [x] Phase 1a — market monitor with real SMARD data
- [x] Phase 1b — leakage-safe features + LightGBM quantile forecast page
- [x] Diagnosis — found overconfident intervals (80% nominal vs ~63% actual coverage)
- [ ] Phase 1c — **CQR calibration** so intervals become trustworthy
- [ ] Phase 2 — ENTSO-E features, weather data, LASSO baseline, CRPS evaluation
- [ ] Phase 3 — decision layer: battery dispatch optimization (MILP) using the calibrated forecasts
