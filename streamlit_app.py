
# app.py
# Uses electricnotpetrol.csv and enforces Tesla EV-only,
# purges Tesla Model X rows before 2016, and applies EV-aware ValueScore.

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

# ------------------------- Helpers -------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(path_primary: str, path_clean: str | None = None):
    # Prefer pre-cleaned file if present
    src = path_clean if path_clean and os.path.exists(path_clean) else path_primary
    if not os.path.exists(src):
        raise FileNotFoundError(f"Could not find {src}. Place your CSV next to app.py.")

    df = pd.read_csv(src)
    miss = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # Cast numerics
    for c in [COL_YEAR, COL_ENGINE, COL_MILEAGE, COL_PRICE]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Trim text
    for c in [COL_BRAND, COL_FUEL, COL_TRANS, COL_COND, COL_MODEL]:
        df[c] = df[c].astype(str).str.strip()

    # ---- Required clean-ups (applied even if file is pre-cleaned) ----
    # 1) Purge Tesla Model X rows before 2016
    pre_rows = len(df)
    mask_purge = (df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model X") & (df[COL_YEAR] < 2016)
    df = df[~mask_purge].copy()
    removed_rows = pre_rows - len(df)

    # 2) Force Tesla to Electric fuel type
    df.loc[df[COL_BRAND] == "Tesla", COL_FUEL] = "Electric"

    # 3) EV rows: engine size not meaningful
    df.loc[df[COL_FUEL] == "Electric", COL_ENGINE] = np.nan

    return df, removed_rows, src

def minmax_by_group(series: pd.Series, group: pd.Series) -> pd.Series:
    d = pd.DataFrame({"v": series, "g": group})
    def _mm(x):
        v = x["v"]
        vmax, vmin = v.max(), v.min()
        if pd.isna(vmax) or pd.isna(vmin) or vmax == vmin:
            # Avoid divide-by-zero; neutral normalization
            return pd.Series(np.full(len(v), 0.5), index=x.index)
        return (v - vmin) / (vmax - vmin)
    return d.groupby("g", dropna=False).apply(_mm).reset_index(level=0, drop=True)

def build_value_score(df: pd.DataFrame, engine_dir: str, w_price: float, w_mileage: float, w_engine: float, engine_for_ev: bool):
    year = df[COL_YEAR]
    n_price   = minmax_by_group(df[COL_PRICE], year)
    n_mileage = minmax_by_group(df[COL_MILEAGE], year)

    # Engine normalization only where engine exists
    n_engine = minmax_by_group(df[COL_ENGINE], year)
    # Scoring directions
    s_price   = (1 - n_price)        # Minimize
    s_mileage = (1 - n_mileage)      # Minimize
    s_engine  = (1 - n_engine) if engine_dir == "Minimize" else n_engine

    # EV-aware: zero-out engine contribution on EV rows (Tesla)
    ev_mask = df[COL_FUEL].eq("Electric")
    if not engine_for_ev:
        s_engine = s_engine.copy()
        s_engine[ev_mask] = 0.0

    raw = w_price * s_price + w_mileage * s_mileage + w_engine * s_engine
    rmin, rmax = raw.min(), raw.max()
    if pd.isna(rmin) or pd.isna(rmax) or rmax == rmin:
        return pd.Series(np.full(len(raw), 50.0), index=df.index)
    return 100.0 * (raw - rmin) / (rmax - rmin)

def currency(x) -> str:
    try: return f\"${x:,.0f}\"
    except: return \"-\"

# ------------------------- Load data -------------------------
PRIMARY = \"electricnotpetrol.csv\"
PRE_CLEAN = \"clean_electricnotpetrol.csv\"  # auto-generated in prep step; optional
df, removed_rows, src_used = load_and_clean(PRIMARY, PRE_CLEAN)

# ------------------------- Sidebar -------------------------
st.sidebar.header(\"Filters\")
years  = sorted([int(y) for y in df[COL_YEAR].dropna().unique()])
brands = sorted(df[COL_BRAND].dropna().unique().tolist())
models = sorted(df[COL_MODEL].dropna().unique().tolist())
fuels  = sorted(df[COL_FUEL].dropna().unique().tolist())

year_sel  = st.sidebar.multiselect(\"Year\", options=years,  default=years)
brand_sel = st.sidebar.multiselect(\"Brand\", options=brands, default=brands)
fuel_sel  = st.sidebar.multiselect(\"Fuel Type\", options=fuels, default=fuels)

min_price = float(np.nanmin(df[COL_PRICE])); max_price = float(np.nanmax(df[COL_PRICE]))
price_sel = st.sidebar.slider(\"Price range\", min_value=min_price, max_value=max_price,
                              value=(min_price, max_price), step=(max_price - min_price)/100)

st.sidebar.divider()
st.sidebar.header(\"ValueScore (EV-aware)\")
w_price   = st.sidebar.number_input(\"Weight: Price (Minimize)\", 0.0, 5.0, step=0.1, value=1.0)
w_mileage = st.sidebar.number_input(\"Weight: Mileage (Minimize)\", 0.0, 5.0, step=0.1, value=0.8)
w_engine  = st.sidebar.number_input(\"Weight: Engine Size\", 0.0, 5.0, step=0.1, value=0.4)
engine_dir = st.sidebar.radio(\"Engine Size direction\", [\"Minimize\", \"Maximize\"], index=0)
engine_for_ev = st.sidebar.checkbox(\"Apply engine component to EVs (Tesla)\", value=False,
                                    help=\"When off, engine size contributes 0 for EV rows.\")

# ------------------------- Apply filters -------------------------
view = df.copy()
if year_sel:   view = view[view[COL_YEAR].isin(year_sel)]
if brand_sel:  view = view[view[COL_BRAND].isin(brand_sel)]
if fuel_sel:   view = view[view[COL_FUEL].isin(fuel_sel)]
view = view[(view[COL_PRICE] >= price_sel[0]) & (view[COL_PRICE] <= price_sel[1])]

# ValueScore
view[\"ValueScore\"] = build_value_score(view, engine_dir, w_price, w_mileage, w_engine, engine_for_ev)

# ------------------------- Title & KPIs -------------------------
st.title(\"🚗 Car Price Portfolio (EV-aware)\")
st.caption(f\"Source file: **{os.path.basename(src_used)}** · Tesla set to Electric · Model X < 2016 purged\")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(\"Vehicles\", f\"{len(view):,}\")
k2.metric(\"Median Price\", currency(np.nanmedian(view[COL_PRICE])))
k3.metric(\"Avg Price\", currency(np.nanmean(view[COL_PRICE])))
k4.metric(\"Median Mileage\", f\"{np.nanmedian(view[COL_MILEAGE]):,.0f}\")
k5.metric(\"Top ValueScore\", f\"{np.nanmax(view['ValueScore']):.1f}\")

# ------------------------- Charts -------------------------
st.subheader(\"🗂️ Interactive Charts\")

# Brand-year average price
agg_brand_year = view.groupby([COL_YEAR, COL_BRAND])[COL_PRICE].mean().reset_index().rename(columns={COL_PRICE: \"AvgPrice\"})
brand_select = alt.selection_point(fields=[COL_BRAND], bind='legend')

bar = alt.Chart(agg_brand_year).mark_bar().encode(
    x=alt.X(f\"{COL_BRAND}:N\", title=\"Brand\", sort=\"-y\"),
    y=alt.Y(\"AvgPrice:Q\", title=\"Average Price\"),
    color=alt.Color(f\"{COL_BRAND}:N\", legend=alt.Legend(title=\"Brand\")),
    column=alt.Column(f\"{COL_YEAR}:N\", title=\"Year\"),
    tooltip=[COL_YEAR, COL_BRAND, alt.Tooltip(\"AvgPrice:Q\", format=\",.0f\", title=\"Avg Price\")],
).add_params(brand_select).transform_filter(brand_select).properties(height=280)

st.altair_chart(bar, use_container_width=True)

# Price distribution by Fuel Type per Year
box = alt.Chart(view).mark_boxplot(size=16).encode(
    x=alt.X(f\"{COL_FUEL}:N\", title=\"Fuel Type\"), 
    y=alt.Y(f\"{COL_PRICE}:Q\", title=\"Price\"), 
    color=alt.Color(f\"{COL_FUEL}:N\", legend=None),
    column=alt.Column(f\"{COL_YEAR}:N\", title=\"Year\"),
    tooltip=[COL_YEAR, COL_BRAND, COL_MODEL, COL_FUEL, alt.Tooltip(COL_PRICE, format=\",.0f\", title=\"Price\")],
).properties(height=280)

st.altair_chart(box, use_container_width=True)

# ------------------------- Table -------------------------
st.subheader(\"🔢 Table (Tesla shown as EV)\")
show_cols = [COL_ID, COL_YEAR, COL_BRAND, COL_MODEL, COL_FUEL, COL_PRICE, COL_MILEAGE, COL_ENGINE, COL_TRANS, COL_COND, \"ValueScore\"]
show_cols = [c for c in show_cols if c in view.columns]
st.dataframe(view[show_cols].sort_values(by=[COL_YEAR, COL_BRAND, COL_MODEL]), use_container_width=True)

st.download_button(\"⬇️ Download current view (CSV)\",
                   view[show_cols].to_csv(index=False).encode(\"utf-8\"),
                   file_name=\"view_ev_aware.csv\")

# ------------------------- Footer -------------------------
with st.expander(\"ℹ️ Notes\", expanded=False):
    st.markdown(\"\"\"\n- Tesla rows are forced to **Electric** fuel type; ICE/Hybrid labels in the raw file are corrected.\n- EV rows ignore **Engine Size** in ValueScore.\n- All **Tesla Model X** rows with **Year < 2016** are removed as synthetic/incorrect.\n\"\"\")
