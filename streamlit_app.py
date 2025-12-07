
# app.py
# Interactive Streamlit dashboard tailored to data.csv:
# Year → Brand → Model portfolio with interactive selections, comparators, drill‑downs,
# and a Model X override (absolute and inflation‑adjusted prices), plus purge of pre‑2016 Model X rows.

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------- Page Config -------------------------
st.set_page_config(
    page_title="Car Price Portfolio (Interactive + Model X US Prices)",
    page_icon= st.image(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Price Predictions</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: #1a1a1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .poster {
            width: 100%;
            max-width: 900px;
            background: linear-gradient(to bottom, #2c3e50 0%, #1a252f 100%);
            border-radius: 8px;
            padding: 50px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.8);
            position: relative;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        h1 {
            font-size: 3.5em;
            font-weight: 900;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);
            margin-bottom: 15px;
        }

        .divider {
            width: 200px;
            height: 4px;
            background: linear-gradient(90deg, transparent, #e74c3c, transparent);
            margin: 0 auto;
        }

        .truck-container {
            margin: 50px 0;
            padding: 40px;
            background: linear-gradient(to bottom, #87ceeb 0%, #e0e0e0 70%, #666 100%);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }

        .truck-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 60%;
            background: linear-gradient(to bottom, #87ceeb, #b0d4e3);
            z-index: 0;
        }

        .truck {
            position: relative;
            z-index: 1;
            width: 100%;
            height: 350px;
        }

        svg {
            width: 100%;
            height: 100%;
            filter: drop-shadow(0 20px 30px rgba(0, 0, 0, 0.6));
        }

        @media (max-width: 600px) {
            h1 {
                font-size: 2em;
            }
            
            .poster {
                padding: 30px 20px;
            }

            .truck {
                height: 250px;
            }
        }
    </style>
</head>
<body>
    <div class="poster">
        <div class="header">
            <h1>Vehicle Price<br>Predictions</h1>
            <div class="divider"></div>
        </div>

        <div class="truck-container">
            <div class="truck">
                <svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <!-- Gradients for realistic shading -->
                        <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" style="stop-color:#c0392b;stop-opacity:1" />
                            <stop offset="50%" style="stop-color:#e74c3c;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#922b21;stop-opacity:1" />
                        </linearGradient>
                        
                        <linearGradient id="bedGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" style="stop-color:#a93226;stop-opacity:1" />
                            <stop offset="50%" style="stop-color:#c0392b;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#7b241c;stop-opacity:1" />
                        </linearGradient>

                        <linearGradient id="roofGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" style="stop-color:#e74c3c;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#c0392b;stop-opacity:1" />
                        </linearGradient>

                        <radialGradient id="wheelGrad" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" style="stop-color:#1a1a1a;stop-opacity:1" />
                            <stop offset="70%" style="stop-color:#0a0a0a;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#000000;stop-opacity:1" />
                        </radialGradient>

                        <linearGradient id="windowGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" style="stop-color:#4a90a4;stop-opacity:0.9" />
                            <stop offset="100%" style="stop-color:#1a3a4a;stop-opacity:1" />
                        </linearGradient>

                        <radialGradient id="rimGrad" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" style="stop-color:#c0c0c0;stop-opacity:1" />
                            <stop offset="60%" style="stop-color:#808080;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#404040;stop-opacity:1" />
                        </radialGradient>
                    </defs>

                    <!-- Ground shadow -->
                    <ellipse cx="400" cy="340" rx="280" ry="25" fill="#000000" opacity="0.3"/>

                    <!-- Truck Bed -->
                    <path d="M 420 180 L 420 160 L 640 160 L 640 280 L 630 280 L 630 190 L 430 190 L 430 280 L 420 280 Z" 
                          fill="url(#bedGrad)" stroke="#7b241c" stroke-width="2"/>
                    
                    <!-- Bed vertical lines -->
                    <line x1="480" y1="160" x2="480" y2="280" stroke="#7b241c" stroke-width="2"/>
                    <line x1="530" y1="160" x2="530" y2="280" stroke="#7b241c" stroke-width="2"/>
                    <line x1="580" y1="160" x2="580" y2="280" stroke="#7b241c" stroke-width="2"/>
                    
                    <!-- Bed gate handle -->
                    <rect x="620" y="220" width="15" height="8" fill="#2c2c2c" rx="2"/>
                    
                    <!-- Main Body -->
                    <rect x="180" y="200" width="260" height="80" fill="url(#bodyGrad)" stroke="#7b241c" stroke-width="2" rx="3"/>
                    
                    <!-- Cabin/Roof -->
                    <path d="M 200 200 L 240 140 L 410 140 L 440 180 L 440 200 Z" 
                          fill="url(#roofGrad)" stroke="#7b241c" stroke-width="2"/>
                    
                    <!-- Front windshield -->
                    <path d="M 205 198 L 245 145 L 360 145 L 360 198 Z" 
                          fill="url(#windowGrad)" stroke="#1a3a4a" stroke-width="2"/>
                    
                    <!-- Side window -->
                    <path d="M 370 145 L 430 145 L 435 180 L 435 198 L 370 198 Z" 
                          fill="url(#windowGrad)" stroke="#1a3a4a" stroke-width="2"/>
                    
                    <!-- Window highlights -->
                    <path d="M 250 150 L 350 150 L 350 160 L 255 160 Z" 
                          fill="#ffffff" opacity="0.3"/>
                    <path d="M 375 150 L 425 150 L 428 170 L 375 170 Z" 
                          fill="#ffffff" opacity="0.3"/>

                    <!-- Front bumper -->
                    <rect x="145" y="210" width="40" height="60" fill="#2c2c2c" rx="4"/>
                    <rect x="150" y="215" width="30" height="50" fill="#1a1a1a" rx="2"/>
                    
                    <!-- Grille -->
                    <rect x="155" y="220" width="20" height="40" fill="#0a0a0a"/>
                    <line x1="155" y1="230" x2="175" y2="230" stroke="#2c2c2c" stroke-width="1"/>
                    <line x1="155" y1="240" x2="175" y2="240" stroke="#2c2c2c" stroke-width="1"/>
                    <line x1="155" y1="250" x2="175" y2="250" stroke="#2c2c2c" stroke-width="1"/>
                    
                    <!-- Headlights -->
                    <circle cx="160" cy="205" r="10" fill="#ffd700" opacity="0.9" filter="blur(1px)"/>
                    <circle cx="160" cy="205" r="8" fill="#ffed4e"/>
                    
                    <circle cx="160" cy="270" r="10" fill="#ff4500" opacity="0.7"/>
                    <circle cx="160" cy="270" r="7" fill="#ff6347"/>
                    
                    <!-- Side mirror -->
                    <rect x="190" y="170" width="15" height="4" fill="#1a1a1a"/>
                    <path d="M 205 168 L 215 165 L 215 177 L 205 174 Z" fill="#1a3a4a" opacity="0.6"/>
                    
                    <!-- Door handle -->
                    <rect x="330" y="235" width="25" height="6" fill="#1a1a1a" rx="3"/>
                    
                    <!-- Door line -->
                    <line x1="290" y1="200" x2="290" y2="280" stroke="#7b241c" stroke-width="2"/>

                    <!-- Toyota badge area -->
                    <ellipse cx="165" cy="240" rx="12" ry="10" fill="#c0c0c0" opacity="0.9"/>
                    <text x="165" y="244" font-family="Arial, sans-serif" font-size="10" font-weight="bold" 
                          fill="#c0392b" text-anchor="middle">T</text>

                    <!-- Rear wheel -->
                    <circle cx="560" cy="280" r="45" fill="url(#wheelGrad)"/>
                    <circle cx="560" cy="280" r="36" fill="url(#rimGrad)"/>
                    <circle cx="560" cy="280" r="15" fill="#2c2c2c"/>
                    
                    <!-- Rear wheel spokes -->
                    <line x1="560" y1="244" x2="560" y2="316" stroke="#505050" stroke-width="3"/>
                    <line x1="524" y1="280" x2="596" y2="280" stroke="#505050" stroke-width="3"/>
                    <line x1="535" y1="255" x2="585" y2="305" stroke="#505050" stroke-width="2"/>
                    <line x1="535" y1="305" x2="585" y2="255" stroke="#505050" stroke-width="2"/>

                    <!-- Front wheel -->
                    <circle cx="260" cy="280" r="45" fill="url(#wheelGrad)"/>
                    <circle cx="260" cy="280" r="36" fill="url(#rimGrad)"/>
                    <circle cx="260" cy="280" r="15" fill="#2c2c2c"/>
                    
                    <!-- Front wheel spokes -->
                    <line x1="260" y1="244" x2="260" y2="316" stroke="#505050" stroke-width="3"/>
                    <line x1="224" y1="280" x2="296" y2="280" stroke="#505050" stroke-width="3"/>
                    <line x1="235" y1="255" x2="285" y2="305" stroke="#505050" stroke-width="2"/>
                    <line x1="235" y1="305" x2="285" y2="255" stroke="#505050" stroke-width="2"/>

                    <!-- Wheel well shadows -->
                    <path d="M 215 280 Q 260 260 305 280" fill="none" stroke="#000000" stroke-width="2" opacity="0.3"/>
                    <path d="M 515 280 Q 560 260 605 280" fill="none" stroke="#000000" stroke-width="2" opacity="0.3"/>

                    <!-- Body highlights -->
                    <rect x="185" y="205" width="250" height="8" fill="#ffffff" opacity="0.2" rx="2"/>
                    <path d="M 205 202 L 245 145 L 255 145 L 210 202 Z" fill="#ffffff" opacity="0.3"/>

                    <!-- Side body line detail -->
                    <line x1="180" y1="240" x2="440" y2="240" stroke="#922b21" stroke-width="1.5"/>
                </svg>
            </div>
        </div>
    </div>
</body>
</html>),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Column Mapping (fixed to your CSV) -------------------------
COL_ID = "Car ID"
COL_BRAND = "Brand"
COL_YEAR = "Year"
COL_ENGINE = "Engine Size"
COL_FUEL = "Fuel Type"
COL_TRANS = "Transmission"
COL_MILEAGE = "Mileage"
COL_COND = "Condition"
COL_PRICE = "Price"
COL_MODEL = "Model"

EXPECTED_COLUMNS = [
    COL_ID, COL_BRAND, COL_YEAR, COL_ENGINE, COL_FUEL,
    COL_TRANS, COL_MILEAGE, COL_COND, COL_PRICE, COL_MODEL
]

# ------------------------- Session Defaults -------------------------
def init_state():
    defaults = {
        "favorites": set(),
        "w_price": 1.0,
        "w_mileage": 0.8,
        "w_engine": 0.4,
        "engine_dir": "Minimize",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ------------------------- Helpers -------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Place 'data.csv' next to app.py.")
    df = pd.read_csv(path)
    miss = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in data.csv: {miss}")

    # Clean numerics
    for c in [COL_YEAR, COL_ENGINE, COL_MILEAGE, COL_PRICE]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Standardize text
    for c in [COL_BRAND, COL_FUEL, COL_TRANS, COL_COND, COL_MODEL]:
        df[c] = df[c].astype(str).str.strip()

    # --- Purge synthetic/incorrect Tesla Model X rows prior to 2016 ---
    pre_count = len(df)
    df = df[~((df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model X") & (df[COL_YEAR] < 2016))]
    removed = pre_count - len(df)

    return df, removed

def minmax_by_group(values: pd.Series, group_index: pd.Series) -> pd.Series:
    d = pd.DataFrame({"v": values, "g": group_index})
    def _mm(x):
        v = x["v"]
        vmax, vmin = v.max(), v.min()
        if pd.isna(vmax) or pd.isna(vmin) or vmax == vmin:
            return pd.Series(np.full(len(v), 0.5), index=x.index)
        return (v - vmin) / (vmax - vmin)
    return d.groupby("g", dropna=False).apply(_mm).reset_index(level=0, drop=True)

def build_value_score(df: pd.DataFrame) -> pd.Series:
    w_price   = st.session_state["w_price"]
    w_mileage = st.session_state["w_mileage"]
    w_engine  = st.session_state["w_engine"]
    engine_dir = st.session_state["engine_dir"]

    year = df[COL_YEAR]
    n_price   = minmax_by_group(df[COL_PRICE], year)
    n_mileage = minmax_by_group(df[COL_MILEAGE], year)
    n_engine  = minmax_by_group(df[COL_ENGINE], year)

    s_price   = (1 - n_price)        # Minimize
    s_mileage = (1 - n_mileage)      # Minimize
    s_engine  = (1 - n_engine) if engine_dir == "Minimize" else n_engine

    raw = w_price * s_price + w_mileage * s_mileage + w_engine * s_engine
    rmin, rmax = raw.min(), raw.max()
    if pd.isna(rmin) or pd.isna(rmax) or rmax == rmin:
        return pd.Series(np.full(len(raw), 50.0), index=df.index)
    return 100.0 * (raw - rmin) / (rmax - rmin)

def best_worst_per_year(df: pd.DataFrame, score_col: str):
    valid = df.dropna(subset=[score_col])
    gb = valid.groupby(COL_YEAR, dropna=False)
    best_idx = gb[score_col].idxmax()
    worst_idx = gb[score_col].idxmin()
    best = valid.loc[best_idx].sort_values(by=COL_YEAR, ascending=True)
    worst = valid.loc[worst_idx].sort_values(by=COL_YEAR, ascending=True)
    return best, worst

def currency(x) -> str:
    try: return f"${x:,.0f}"
    except: return "-"

# ------------------------- Load base data & purge report -------------------------
DATA_PATH = "data.csv"
df, removed_rows = load_data(DATA_PATH)

# ------------------------- Optional Model X price override -------------------------
# Expected file: modelx_prices.csv with columns:
# Year, AbsolutePrice_USD, InflationAdjusted_USD  (Years: 2016..2023)
MODELX_FILE = "modelx_prices.csv"
modelx_msg = None
modelx_df = None

if os.path.exists(MODELX_FILE):
    modelx_df = pd.read_csv(MODELX_FILE)
    # Clean
    modelx_df["Year"] = pd.to_numeric(modelx_df["Year"], errors="coerce")
    modelx_df["AbsolutePrice_USD"] = pd.to_numeric(modelx_df["AbsolutePrice_USD"], errors="coerce")
    modelx_df["InflationAdjusted_USD"] = pd.to_numeric(modelx_df["InflationAdjusted_USD"], errors="coerce")
    # Only years 2016..2023
    modelx_df = modelx_df[(modelx_df["Year"] >= 2016) & (modelx_df["Year"] <= 2023)]
else:
    modelx_msg = ("The original dataset for the Tesla Model X did not accurately model any absolute prices from 2000 - 2024; to account for this, external data was added.")

# Merge override series (left merge on Year for Tesla Model X only)
if modelx_df is not None and not modelx_df.empty:
    # Separate Model X and non-Model X
    non_x = df[~((df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model X"))].copy()
    x_rows = df[(df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model X")].copy()

    # Merge series onto Model X rows by Year
    x_rows = x_rows.merge(modelx_df, how="left", left_on=COL_YEAR, right_on="Year")
    # If AbsolutePrice_USD present, use that to override Price (to keep the dataset consistent)
    x_rows.loc[~x_rows["AbsolutePrice_USD"].isna(), COL_PRICE] = x_rows["AbsolutePrice_USD"]

    # Recombine
    df = pd.concat([non_x, x_rows.drop(columns=["Year"])], ignore_index=True)

# ------------------------- Sidebar Controls -------------------------
st.sidebar.header("Filters")
years = sorted([int(y) for y in df[COL_YEAR].dropna().unique()])
brands = sorted(df[COL_BRAND].dropna().unique().tolist())
models = sorted(df[COL_MODEL].dropna().unique().tolist())

year_sel = st.sidebar.multiselect("Year", options=years, default=years)
brand_sel = st.sidebar.multiselect("Brand", options=brands, default=brands)
model_sel = st.sidebar.multiselect("Model/Trim", options=models, default=models)

min_price = float(np.nanmin(df[COL_PRICE])); max_price = float(np.nanmax(df[COL_PRICE]))
price_sel = st.sidebar.slider("Price range", min_value=min_price, max_value=max_price,
                              value=(min_price, max_price), step=(max_price - min_price)/100)

st.sidebar.divider()
st.sidebar.header("Value Score")
st.sidebar.number_input("Weight: Price (Minimize)", 0.0, 5.0, step=0.1, value=st.session_state["w_price"], key="w_price")
st.sidebar.number_input("Weight: Mileage (Minimize)", 0.0, 5.0, step=0.1, value=st.session_state["w_mileage"], key="w_mileage")
st.sidebar.number_input("Weight: Engine Size", 0.0, 5.0, step=0.1, value=st.session_state["w_engine"], key="w_engine")
st.sidebar.radio("Engine Size direction", ["Minimize", "Maximize"],
                 index=0 if st.session_state["engine_dir"]=="Minimize" else 1, key="engine_dir")

# ------------------------- Apply Filters -------------------------
view = df.copy()
if year_sel:   view = view[view[COL_YEAR].isin(year_sel)]
if brand_sel:  view = view[view[COL_BRAND].isin(brand_sel)]
if model_sel:  view = view[view[COL_MODEL].isin(model_sel)]
view = view[(view[COL_PRICE] >= price_sel[0]) & (view[COL_PRICE] <= price_sel[1])]

# Compute ValueScore for current view
view["ValueScore"] = build_value_score(view)

# ------------------------- Title & KPIs -------------------------
st.title("Vehicle Price Portfolio (Interactive)")
st.caption("Fixed dataset: Year · Brand · Model/Trim · Price · Mileage · Engine Size · Fuel · Transmission · Condition")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Vehicles", f"{len(view):,}")
k2.metric("Median Price", currency(np.nanmedian(view[COL_PRICE])))
k3.metric("Avg Price", currency(np.nanmean(view[COL_PRICE])))
k4.metric("Median Mileage", f"{np.nanmedian(view[COL_MILEAGE]):,.0f}")
k5.metric("Top ValueScore", f"{np.nanmax(view['ValueScore']):.1f}")

# Purge notice
with st.expander("Data Housekeeping"):
    st.markdown(f"- Purged **{removed_rows:,}** synthetic Tesla Model X rows (years < 2016).")
    if modelx_msg:
        st.warning(modelx_msg)
    else:
        yrs = ", ".join([str(int(y)) for y in sorted(modelx_df['Year'].unique())])
        st.success(f"Loaded Model X US price series for years: {yrs}. "
                   f"Absolute prices override base listing price for Model X.")

# ------------------------- Model X price panel -------------------------
st.subheader("🟢 Tesla Model X — US Market Prices (Absolute & Inflation‑Adjusted)")
x_panel = df[(df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model X")].copy()

# Build a compact series for plotting
plot_df = x_panel[[COL_YEAR, COL_PRICE]].rename(columns={COL_PRICE: "AbsolutePrice_USD"}).copy()
if "InflationAdjusted_USD" in x_panel.columns:
    # Add inflation-adjusted series if present from the sidecar file
    plot_df = plot_df.merge(
        x_panel[[COL_YEAR, "InflationAdjusted_USD"]].drop_duplicates(COL_YEAR),
        on=COL_YEAR, how="left"
    )
else:
    plot_df["InflationAdjusted_USD"] = np.nan

plot_df = plot_df.dropna(subset=[COL_YEAR]).drop_duplicates(COL_YEAR).sort_values(COL_YEAR)

if plot_df.empty:
    st.info("No Model X data available in the current filter.")
else:
    # Melt for multi-line chart
    m = plot_df.melt(id_vars=[COL_YEAR], value_vars=["AbsolutePrice_USD", "InflationAdjusted_USD"],
                     var_name="Series", value_name="PriceUSD")

    line = alt.Chart(m).mark_line(point=True, interpolate="monotone").encode(
        x=alt.X(f"{COL_YEAR}:O", title="Year"),
        y=alt.Y("PriceUSD:Q", title="Price (USD)", axis=alt.Axis(format="~s")),
        color=alt.Color("Series:N", title="Series", scale=alt.Scale(scheme="teals")),
        tooltip=[COL_YEAR, "Series", alt.Tooltip("PriceUSD:Q", format=",.0f", title="Price")]
    ).properties(height=280)

    st.altair_chart(line, use_container_width=True)

    cta, ctb = st.columns(2)
    with cta:
        st.markdown("**Model X price table**")
        st.dataframe(plot_df.rename(columns={
            "AbsolutePrice_USD": "Absolute (USD)",
            "InflationAdjusted_USD": "Inflation‑adj (USD)"
        }), use_container_width=True)
    with ctb:
        st.download_button("⬇️ Download Model X price series (CSV)",
                           plot_df.to_csv(index=False).encode("utf-8"),
                           file_name="modelx_price_series.csv")

# ------------------------- Existing interactive charts -------------------------
# (Same as your interactive build — bar, box, trend)
st.subheader("*Interactive Charts (changes are made through the filter sidebar)*")
# Aggregates for charts
chart_df = view.copy()
chart_df[COL_BRAND] = chart_df[COL_BRAND].astype(str)
chart_df[COL_MODEL] = chart_df[COL_MODEL].astype(str)

agg_brand_year = chart_df.groupby([COL_YEAR, COL_BRAND])[COL_PRICE].mean().reset_index().rename(columns={COL_PRICE: "AvgPrice"})
agg_model_year = chart_df.groupby([COL_YEAR, COL_MODEL])[COL_PRICE].mean().reset_index().rename(columns={COL_PRICE: "AvgPrice"})

brand_select = alt.selection_point(fields=[COL_BRAND], bind='legend')

bar = alt.Chart(agg_brand_year).mark_bar().encode(
    x=alt.X(f"{COL_BRAND}:N", title="Brand", sort="-y"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{COL_BRAND}:N", legend=alt.Legend(title="Brand")),
    column=alt.Column(f"{COL_YEAR}:N", title="Year"),
    tooltip=[COL_YEAR, COL_BRAND, alt.Tooltip("AvgPrice:Q", format=",.0f", title="Avg Price")],
).add_params(brand_select).transform_filter(brand_select).properties(height=280)

box = alt.Chart(chart_df).mark_boxplot(size=16).encode(
    x=alt.X(f"{COL_MODEL}:N", title="Model/Trim"),
    y=alt.Y(f"{COL_PRICE}:Q", title="Price"),
    color=alt.Color(f"{COL_BRAND}:N", legend=None),
    column=alt.Column(f"{COL_YEAR}:N", title="Year"),
    tooltip=[COL_YEAR, COL_BRAND, COL_MODEL, alt.Tooltip(COL_PRICE, format=",.0f", title="Price")]
).transform_filter(brand_select).properties(height=280)

line_brand = alt.Chart(agg_brand_year).mark_line(point=True).encode(
    x=alt.X(f"{COL_YEAR}:O", title="Year"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{COL_BRAND}:N", title="Brand"),
    tooltip=[COL_YEAR, COL_BRAND, alt.Tooltip("AvgPrice:Q", format=",.0f", title="Avg Price")],
).add_params(brand_select).transform_filter(brand_select).properties(height=280)

st.altair_chart(bar, use_container_width=True)
st.altair_chart(box, use_container_width=True)
st.altair_chart(line_brand, use_container_width=True)

# ------------------------- Overview (best/worst), Compare, Drill, Table -------------------------
tab_overview, tab_compare, tab_drill, tab_table = st.tabs(["Overview", "Compare", "Drill‑down", "Table"])

with tab_overview:
    st.markdown("### 🏅 Per‑Year Picks (Most & Least Worth Buying)")
    best, worst = best_worst_per_year(view, "ValueScore")
    cols = [COL_YEAR, COL_BRAND, COL_MODEL, COL_PRICE, COL_MILEAGE, COL_ENGINE, "ValueScore"]
    cols = [c for c in cols if c in best.columns]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Best (per Year)**")
        st.dataframe(best[cols], use_container_width=True)
        st.download_button("⬇️ Download Best (CSV)", best[cols].to_csv(index=False).encode("utf-8"),
                           file_name="best_worth_buying_per_year.csv")
    with c2:
        st.markdown("**Worst (per Year)**")
        st.dataframe(worst[cols], use_container_width=True)
        st.download_button("⬇️ Download Worst (CSV)", worst[cols].to_csv(index=False).encode("utf-8"),
                           file_name="least_worth_buying_per_year.csv")

with tab_compare:
    st.markdown("### 🔁 Brand / Model Comparator")
    brands_list = sorted(view[COL_BRAND].unique())
    brand_a = st.selectbox("Brand A", options=brands_list, index=0)
    models_a = sorted(view[view[COL_BRAND] == brand_a][COL_MODEL].unique())
    model_a = st.selectbox("Model A", options=models_a, index=0)

    brand_b = st.selectbox("Brand B", options=brands_list, index=min(1, len(brands_list)-1))
    models_b = sorted(view[view[COL_BRAND] == brand_b][COL_MODEL].unique())
    model_b = st.selectbox("Model B", options=models_b, index=0)

    comp = view[
        ((view[COL_BRAND] == brand_a) & (view[COL_MODEL] == model_a)) |
        ((view[COL_BRAND] == brand_b) & (view[COL_MODEL] == model_b))
    ].copy()

    if comp.empty:
        st.info("No matching rows for the selected pairs—adjust filters.")
    else:
        dist = alt.Chart(comp).transform_fold(
            [COL_PRICE, COL_MILEAGE, "ValueScore"], as_=["Metric", "Value"]
        ).mark_boxplot(size=40).encode(
            x=alt.X("Metric:N", title="Metric"),
            y=alt.Y("Value:Q"),
            color=alt.Color(f"{COL_BRAND}:N"),
            column=alt.Column(f"{COL_YEAR}:N", title="Year"),
            tooltip=[COL_YEAR, COL_BRAND, COL_MODEL, "Metric", alt.Tooltip("Value:Q", format=",.2f")]
        ).properties(height=250)
      
