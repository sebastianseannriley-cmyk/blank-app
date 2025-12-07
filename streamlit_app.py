
# app.py
# Streamlit app tailored to your dataset (data.csv) for a car price portfolio:
# Organized by Year → Brand → Model/Trim, with per-year ValueScore and best/worst picks.
# Author: Riley, Sebastian-Seann B CDT 2029 (built by M365 Copilot)

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import os
import re

# ------------------------- Page Config -------------------------
st.set_page_config(
    page_title="Car Price Portfolio (Year → Brand → Model)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Constants (column mapping) -------------------------
# Tailored to your dataset's headers. Do NOT change unless your CSV changes.
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

# ------------------------- Helpers -------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Place your 'data.csv' next to app.py."
        )
    df = pd.read_csv(path)
    # Basic sanity: ensure expected columns exist (case-sensitive)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data.csv: {missing}")

    # Clean numerics (defensive)
    for c in [COL_YEAR, COL_ENGINE, COL_MILEAGE, COL_PRICE]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Standardize text columns to string
    for c in [COL_BRAND, COL_FUEL, COL_TRANS, COL_COND, COL_MODEL]:
        df[c] = df[c].astype(str).str.strip()

    return df

def minmax_by_group(values: pd.Series, group_index: pd.Series) -> pd.Series:
    """Min-max normalize within groups (e.g., per year)."""
    df = pd.DataFrame({"v": values, "g": group_index})
    def _mm(x):
        v = x["v"]
        vmax, vmin = v.max(), v.min()
        if pd.isna(vmax) or pd.isna(vmin) or vmax == vmin:
            return pd.Series(np.full(len(v), 0.5), index=x.index)
        return (v - vmin) / (vmax - vmin)
    return df.groupby("g", dropna=False).apply(_mm).reset_index(level=0, drop=True)

def build_value_score(df: pd.DataFrame) -> pd.Series:
    """
    ValueScore (0–100), normalized **within each Year**, combining selected features:
    - Price (Minimize)
    - Mileage (Minimize)
    - Engine Size (configurable Minimize/Maximize via sidebar)
    You can tune weights in the sidebar.
    """
    # Sidebar-configured weights & direction
    w_price = st.session_state.get("w_price", 1.0)
    w_mileage = st.session_state.get("w_mileage", 0.8)
    w_engine = st.session_state.get("w_engine", 0.4)
    engine_dir = st.session_state.get("engine_dir", "Minimize")  # or "Maximize"

    year = df[COL_YEAR]
    # Normalize numeric features per year (defensive: NaNs become 0.5 after normalization)
    n_price = minmax_by_group(df[COL_PRICE], year)
    n_mileage = minmax_by_group(df[COL_MILEAGE], year)
    n_engine = minmax_by_group(df[COL_ENGINE], year)

    # Lower is better for minimize; invert normalized values.
    s_price = (1 - n_price)
    s_mileage = (1 - n_mileage)
    s_engine = (1 - n_engine) if engine_dir == "Minimize" else n_engine

    raw = w_price * s_price + w_mileage * s_mileage + w_engine * s_engine
    # Scale to 0–100 across the whole selection for readability
    rmin, rmax = raw.min(), raw.max()
    if pd.isna(rmin) or pd.isna(rmax) or rmax == rmin:
        return pd.Series(np.full(len(raw), 50.0), index=df.index)
    return 100.0 * (raw - rmin) / (rmax - rmin)

def best_worst_per_year(df: pd.DataFrame, score_col: str):
    """Return DataFrames for best/worst vehicles per year based on ValueScore."""
    valid = df.dropna(subset=[score_col])
    gb = valid.groupby(COL_YEAR, dropna=False)
    best_idx = gb[score_col].idxmax()
    worst_idx = gb[score_col].idxmin()
    best = valid.loc[best_idx].sort_values(by=COL_YEAR, ascending=True)
    worst = valid.loc[worst_idx].sort_values(by=COL_YEAR, ascending=True)
    return best, worst

def currency(x) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "-"

# ------------------------- Load data (no upload option) -------------------------
DATA_PATH = "data.csv"
df = load_data(DATA_PATH)

st.title("🚗 Car Price Portfolio (Year → Brand → Model)")
st.caption("Data columns fixed to your CSV: Year, Brand, Model/Trim, Price, Mileage, Engine Size, Fuel, Transmission, Condition. (File: data.csv)")

# ------------------------- Sidebar: Filters & Grouping -------------------------
st.sidebar.title("🧭 Filters & Grouping")

# Year filter
years_avail = sorted([int(y) for y in df[COL_YEAR].dropna().unique()])
years_selected = st.sidebar.multiselect("Filter by Year", options=years_avail, default=years_avail)

# Brand filter
brands_avail = sorted(df[COL_BRAND].dropna().unique().tolist())
brands_selected = st.sidebar.multiselect("Filter by Brand", options=brands_avail, default=brands_avail)

# Model filter (optional)
models_avail = sorted(df[COL_MODEL].dropna().unique().tolist())
models_selected = st.sidebar.multiselect("Filter by Model/Trim", options=models_avail, default=models_avail)

# Secondary grouping
group_primary = st.sidebar.selectbox(
    "Primary grouping (within Year)",
    options=[COL_BRAND, COL_MODEL, COL_FUEL, COL_TRANS, COL_COND],
    index=0,
    help="Default is Brand; switch to Model/Trim or other categories (Fuel, Transmission, Condition)."
)

group_secondary = st.sidebar.selectbox(
    "Optional secondary grouping",
    options=["(none)", COL_MODEL, COL_FUEL, COL_TRANS, COL_COND],
    index=0
)

# Price range
min_price = float(np.nanmin(df[COL_PRICE]))
max_price = float(np.nanmax(df[COL_PRICE]))
price_range = st.sidebar.slider("Price range", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=(max_price - min_price)/100)

# Apply filters
view = df.copy()
if years_selected:
    view = view[view[COL_YEAR].isin(years_selected)]
if brands_selected:
    view = view[view[COL_BRAND].isin(brands_selected)]
if models_selected:
    view = view[view[COL_MODEL].isin(models_selected)]
view = view[(view[COL_PRICE] >= price_range[0]) & (view[COL_PRICE] <= price_range[1])]

# ------------------------- Sidebar: Value Score config -------------------------
st.sidebar.title("🎯 Value Score")
st.sidebar.write("Per‑year normalization. Tune weights based on your priorities.")
st.sidebar.number_input("Weight: Price (Minimize)", min_value=0.0, max_value=5.0, step=0.1, value=1.0, key="w_price")
st.sidebar.number_input("Weight: Mileage (Minimize)", min_value=0.0, max_value=5.0, step=0.1, value=0.8, key="w_mileage")
st.sidebar.number_input("Weight: Engine Size", min_value=0.0, max_value=5.0, step=0.1, value=0.4, key="w_engine")
st.sidebar.selectbox("Engine Size direction", options=["Minimize", "Maximize"], index=0, key="engine_dir")

# Compute ValueScore for current view
view["ValueScore"] = build_value_score(view)

# ------------------------- Summary KPIs -------------------------
st.subheader("📊 Overview (current selection)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Vehicles", f"{len(view):,}")
try:
    k2.metric("Median Price", currency(np.nanmedian(view[COL_PRICE])))
    k3.metric("Average Price", currency(np.nanmean(view[COL_PRICE])))
    k4.metric("Price Std Dev", currency(np.nanstd(view[COL_PRICE])))
except Exception:
    st.warning("Some price stats were unavailable due to non-numeric values.")

# ------------------------- Charts -------------------------
st.subheader("🗂️ Average Price by Year → Group")
chart = view.copy()
chart[group_primary] = chart[group_primary].astype(str)
if group_secondary != "(none)":
    chart[group_secondary] = chart[group_secondary].astype(str)

gb_cols = [COL_YEAR, group_primary] + ([group_secondary] if group_secondary != "(none)" else [])
agg = chart.groupby(gb_cols, dropna=False)[COL_PRICE].mean().reset_index().rename(columns={COL_PRICE: "AvgPrice"})

color_col = group_secondary if group_secondary != "(none)" else group_primary
bar = alt.Chart(agg).mark_bar().encode(
    x=alt.X(f"{group_primary}:N", title=group_primary),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{color_col}:N", legend=alt.Legend(title=color_col)),
    column=alt.Column(f"{COL_YEAR}:N", title="Year"),
    tooltip=[COL_YEAR, group_primary] + ([group_secondary] if group_secondary != "(none)" else []) + ["AvgPrice"]
).properties(height=300)
st.altair_chart(bar, use_container_width=True)

st.subheader("📦 Price Distribution (Box Plot)")
box = alt.Chart(chart).mark_boxplot(size=18).encode(
    x=alt.X(f"{group_primary}:N", title=group_primary),
    y=alt.Y(f"{COL_PRICE}:Q", title="Price"),
    color=alt.Color(f"{group_primary}:N", legend=None),
    column=alt.Column(f"{COL_YEAR}:N", title="Year"),
    tooltip=[COL_YEAR, group_primary, COL_PRICE]
).properties(height=300)
st.altair_chart(box, use_container_width=True)

st.subheader("📉 Average Price Trend by Brand")
trend = chart.groupby([COL_YEAR, COL_BRAND], dropna=False)[COL_PRICE].mean().reset_index().rename(columns={COL_PRICE: "AvgPrice"})
line = alt.Chart(trend).mark_line(point=True).encode(
    x=alt.X(f"{COL_YEAR}:O", title="Year"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{COL_BRAND}:N", title="Brand"),
    tooltip=[COL_YEAR, COL_BRAND, "AvgPrice"]
).properties(height=300)
st.altair_chart(line, use_container_width=True)

# ------------------------- Best & Worst per Year -------------------------
st.subheader("🏅 Most & Least Worth Buying per Year (ValueScore)")
best, worst = best_worst_per_year(view, score_col="ValueScore")

cols_min = [COL_YEAR, COL_BRAND, COL_MODEL, COL_PRICE, COL_MILEAGE, COL_ENGINE, "ValueScore"]
cols_min = [c for c in cols_min if c in best.columns]

c1, c2 = st.columns(2)
with c1:
    st.markdown("### ✅ Most Worth Buying")
    st.dataframe(best[cols_min].sort_values(by=COL_YEAR), use_container_width=True)
    st.download_button(
        "⬇️ Download Best (CSV)",
        data=best[cols_min].to_csv(index=False).encode("utf-8"),
        file_name="best_worth_buying_per_year.csv",
        mime="text/csv",
    )
with c2:
    st.markdown("### ❌ Least Worth Buying")
    st.dataframe(worst[cols_min].sort_values(by=COL_YEAR), use_container_width=True)
    st.download_button(
        "⬇️ Download Worst (CSV)",
        data=worst[cols_min].to_csv(index=False).encode("utf-8"),
        file_name="least_worth_buying_per_year.csv",
        mime="text/csv",
    )

# ------------------------- Explore Table -------------------------
st.subheader("🔍 Explore Listings")
with st.expander("Filters", expanded=False):
    brand_filter = st.text_input("Brand contains", value="")
    model_filter = st.text_input("Model/Trim contains", value="")

filtered = view.copy()
if brand_filter.strip():
    filtered = filtered[filtered[COL_BRAND].str.contains(brand_filter, case=False, na=False)]
if model_filter.strip():
    filtered = filtered[filtered[COL_MODEL].str.contains(model_filter, case=False, na=False)]

show_cols = [COL_ID, COL_YEAR, COL_BRAND, COL_MODEL, COL_PRICE, COL_MILEAGE, COL_ENGINE, COL_FUEL, COL_TRANS, COL_COND, "ValueScore"]
show_cols = [c for c in show_cols if c in filtered.columns]
st.dataframe(filtered[show_cols].sort_values(by=[COL_YEAR, COL_BRAND, COL_MODEL]), use_container_width=True)

st.download_button(
    "⬇️ Download current view (CSV)",
    data=filtered[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="car_portfolio_view.csv",
    mime="text/csv",
)

# ------------------------- Data Notes -------------------------
with st.expander("ℹ️ Data & Scoring Settings", expanded=False):
    st.markdown(f"""
**Dataset columns** (fixed):  
- `{', '.join(EXPECTED_COLUMNS)}`  
Scoring criteria:  
- **Price**: lower is better (Minimize).  
- **Mileage**: lower is better (Minimize).  
- **Engine Size**: sidebar lets you choose Minimize or Maximize.  
ValueScore is min‑max normalized **per Year**, then scaled to 0–100.
""")
