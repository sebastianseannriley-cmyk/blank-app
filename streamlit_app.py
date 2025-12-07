
# app.py
# Minimal-to-full Streamlit app that:
# 1) loads electricnotpetrol.csv (or clean_electricnotpetrol.csv if present),
# 2) purges Tesla Model X < 2016,
# 3) forces Tesla fuel type to Electric and makes ValueScore EV-aware,
# 4) shows quick diagnostics + charts + table.

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------- Page Config -------------------------
st.set_page_config(
    page_title="Car Price Portfolio (EV-aware)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Column Mapping -------------------------
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

EXPECTED_COLUMNS = [COL_ID, COL_BRAND, COL_YEAR, COL_ENGINE, COL_FUEL,
                    COL_TRANS, COL_MILEAGE, COL_COND, COL_PRICE, COL_MODEL]

# ------------------------- Cache helpers -------------------------
# If your Streamlit version < 1.18 (no cache_data), uncomment the next line and replace cache_data with cache.
# @st.cache
@st.cache_data(show_spinner=False)
def _read_csv(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_and_clean(path_primary, path_clean=None):
    # Prefer pre-cleaned file if present; else load primary and clean on the fly
    src = path_clean if (path_clean and os.path.exists(path_clean)) else path_primary
    if not os.path.exists(src):
        raise FileNotFoundError(f"Could not find '{src}'. Put the CSV next to app.py.")

    df = _read_csv(src)

    # Validate columns
    miss = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # Cast numerics
    for c in [COL_YEAR, COL_ENGINE, COL_MILEAGE, COL_PRICE]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Trim text
    for c in [COL_BRAND, COL_FUEL, COL_TRANS, COL_COND, COL_MODEL]:
        df[c] = df[c].astype(str).str.strip()

    # ---- Required clean-ups ----
    pre_rows = len(df)
    # Purge Tesla Model X rows before 2016
    mask_purge = (df[COL_BRAND].str.casefold() == "tesla") & (df[COL_MODEL].str.strip() == "Model X") & (df[COL_YEAR] < 2016)
    df = df[~mask_purge].copy()
    removed_rows = pre_rows - len(df)

    # Force Tesla to Electric fuel type
    df.loc[df[COL_BRAND].str.casefold() == "tesla", COL_FUEL] = "Electric"

    # EV rows: engine size not meaningful (set NaN)
    df.loc[df[COL_FUEL] == "Electric", COL_ENGINE] = np.nan

    return df, removed_rows, os.path.basename(src)

# ------------------------- Scoring helpers -------------------------
def minmax_by_group(series, group):
    d = pd.DataFrame({"v": series, "g": group})
    def _mm(x):
        v = x["v"]
        vmax, vmin = v.max(), v.min()
        if pd.isna(vmax) or pd.isna(vmin) or vmax == vmin:
            return pd.Series(np.full(len(v), 0.5), index=x.index)  # neutral
        return (v - vmin) / (vmax - vmin)
    return d.groupby("g", dropna=False).apply(_mm).reset_index(level=0, drop=True)

def build_value_score(df, engine_dir, w_price, w_mileage, w_engine, engine_for_ev):
    year = df[COL_YEAR]
    n_price   = minmax_by_group(df[COL_PRICE], year)
    n_mileage = minmax_by_group(df[COL_MILEAGE], year)
    n_engine  = minmax_by_group(df[COL_ENGINE], year)

    s_price   = (1 - n_price)                 # Minimize
    s_mileage = (1 - n_mileage)               # Minimize
    s_engine  = (1 - n_engine) if engine_dir == "Minimize" else n_engine

    # EV-aware: zero engine contribution on EV rows when engine_for_ev=False
    ev_mask = df[COL_FUEL].eq("Electric")
    if not engine_for_ev:
        s_engine = s_engine.copy()
        s_engine[ev_mask] = 0.0

    raw = w_price * s_price + w_mileage * s_mileage + w_engine * s_engine
    rmin, rmax = raw.min(), raw.max()
    if pd.isna(rmin) or pd.isna(rmax) or rmax == rmin:
        return pd.Series(np.full(len(raw), 50.0), index=df.index)
    return 100.0 * (raw - rmin) / (rmax - rmin)

def currency(x):
    try: return f"${x:,.0f}"
    except: return "-"

# ------------------------- Load data (with robust error handling) -------------------------
PRIMARY = "electricnotpetrol.csv"
PRE_CLEAN = "clean_electricnotpetrol.csv"  # optional
try:
    df, removed_rows, src_used = load_and_clean(PRIMARY, PRE_CLEAN)
except Exception as e:
    st.error(f"Failed to load/clean data: {e}")
    st.stop()

# ------------------------- Sidebar -------------------------
st.sidebar.header("Filters")
years  = sorted([int(y) for y in df[COL_YEAR].dropna().unique()])
brands = sorted(df[COL_BRAND].dropna().unique().tolist())
fuels  = sorted(df[COL_FUEL].dropna().unique().tolist())

year_sel  = st.sidebar.multiselect("Year",      options=years,  default=years)
brand_sel = st.sidebar.multiselect("Brand",     options=brands, default=brands)
fuel_sel  = st.sidebar.multiselect("Fuel Type", options=fuels,  default=fuels)

min_price = float(np.nanmin(df[COL_PRICE])); max_price = float(np.nanmax(df[COL_PRICE]))
price_sel = st.sidebar.slider("Price range", min_value=min_price, max_value=max_price,
                              value=(min_price, max_price), step=(max_price - min_price)/100)

st.sidebar.divider()
st.sidebar.header("ValueScore (EV-aware)")
w_price      = st.sidebar.number_input("Weight: Price (Minimize)",   0.0, 5.0, step=0.1, value=1.0)
w_mileage    = st.sidebar.number_input("Weight: Mileage (Minimize)", 0.0, 5.0, step=0.1, value=0.8)
w_engine     = st.sidebar.number_input("Weight: Engine Size",        0.0, 5.0, step=0.1, value=0.4)
engine_dir   = st.sidebar.radio("Engine Size direction", ["Minimize", "Maximize"], index=0)
engine_for_ev = st.sidebar.checkbox("Apply engine component to EVs (Tesla)", value=False)

# ------------------------- Diagnostics -------------------------
st.success(f"Source: **{src_used}** · Purged Tesla Model X (<2016) rows: **{removed_rows}**")
tesla_rows = df[df[COL_BRAND].str.casefold() == "tesla"]
st.caption(f"Telsa rows after clean: **{len(tesla_rows)}** · Fuel types present: {sorted(tesla_rows[COL_FUEL].unique())}")

# ------------------------- Apply filters -------------------------
view = df.copy()
if year_sel:  view = view[view[COL_YEAR].isin(year_sel)]
if brand_sel: view = view[view[COL_BRAND].isin(brand_sel)]
if fuel_sel:  view = view[view[COL_FUEL].isin(fuel_sel)]
view = view[(view[COL_PRICE] >= price_sel[0]) & (view[COL_PRICE] <= price_sel[1])]

# ValueScore
view["ValueScore"] = build_value_score(view, engine_dir, w_price, w_mileage, w_engine, engine_for_ev)

# ------------------------- KPIs -------------------------
st.title("🚗 Car Price Portfolio (EV-aware)")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Vehicles", f"{len(view):,}")
k2.metric("Median Price", currency(np.nanmedian(view[COL_PRICE])))
k3.metric("Avg Price", currency(np.nanmean(view[COL_PRICE])))
k4.metric("Median Mileage", f"{np.nanmedian(view[COL_MILEAGE]):,.0f}")
k5.metric("Top ValueScore", f"{np.nanmax(view['ValueScore']):.1f}")

# ------------------------- Charts -------------------------
st.subheader("🗂️ Interactive Charts")

agg_brand_year = view.groupby([COL_YEAR, COL_BRAND])[COL_PRICE].mean().reset_index()
agg_brand_year = agg_brand_year.rename(columns={COL_PRICE: "AvgPrice"})
brand_select = alt.selection_point(fields=[COL_BRAND], bind='legend')

bar = alt.Chart(agg_brand_year).mark_bar().encode(
    x=alt.X(f"{COL_BRAND}:N", title="Brand", sort="-y"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{COL_BRAND}:N", legend=alt.Legend(title="Brand")),
