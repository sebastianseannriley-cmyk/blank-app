
# app.py
# Interactive Streamlit dashboard tailored to data.csv:
# Year → Brand → Model portfolio with interactive selections, comparators, and drill-downs.

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------- Page Config -------------------------
st.set_page_config(
    page_title="Car Price Portfolio (Interactive)",
    page_icon="🚗",
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
        "selected_brand": None,
        "selected_model": None,
        "selected_year": None,
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
    for c in [COL_BRAND, COL_FUEL, COL_TRANS, COL_COND, COL_MODEL]:
        df[c] = df[c].astype(str).str.strip()
    return df

def minmax_by_group(values: pd.Series, group_index: pd.Series) -> pd.Series:
    df = pd.DataFrame({"v": values, "g": group_index})
    def _mm(x):
        v = x["v"]
        vmax, vmin = v.max(), v.min()
        if pd.isna(vmax) or pd.isna(vmin) or vmax == vmin:
            return pd.Series(np.full(len(v), 0.5), index=x.index)
        return (v - vmin) / (vmax - vmin)
    return df.groupby("g", dropna=False).apply(_mm).reset_index(level=0, drop=True)

def build_value_score(df: pd.DataFrame) -> pd.Series:
    w_price = st.session_state["w_price"]
    w_mileage = st.session_state["w_mileage"]
    w_engine = st.session_state["w_engine"]
    engine_dir = st.session_state["engine_dir"]

    year = df[COL_YEAR]
    n_price = minmax_by_group(df[COL_PRICE], year)
    n_mileage = minmax_by_group(df[COL_MILEAGE], year)
    n_engine = minmax_by_group(df[COL_ENGINE], year)

    s_price = (1 - n_price)        # Minimize
    s_mileage = (1 - n_mileage)    # Minimize
    s_engine = (1 - n_engine) if engine_dir == "Minimize" else n_engine

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

# ------------------------- Load data -------------------------
DATA_PATH = "data.csv"
df = load_data(DATA_PATH)

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
st.sidebar.radio("Engine Size direction", ["Minimize", "Maximize"], index=0 if st.session_state["engine_dir"]=="Minimize" else 1, key="engine_dir")

st.sidebar.divider()
st.sidebar.header("Quick Actions")
reset_btn = st.sidebar.button("Reset selections & favorites")

if reset_btn:
    st.session_state["favorites"] = set()
    for key in ["selected_brand", "selected_model", "selected_year"]:
        st.session_state[key] = None
    st.experimental_rerun()

# ------------------------- Apply Filters -------------------------
view = df.copy()
if year_sel:   view = view[view[COL_YEAR].isin(year_sel)]
if brand_sel:  view = view[view[COL_BRAND].isin(brand_sel)]
if model_sel:  view = view[view[COL_MODEL].isin(model_sel)]
view = view[(view[COL_PRICE] >= price_sel[0]) & (view[COL_PRICE] <= price_sel[1])]

# Compute ValueScore for current view
view["ValueScore"] = build_value_score(view)

# ------------------------- Title & KPIs -------------------------
st.title("🚗 Car Price Portfolio (Interactive)")
st.caption("Fixed dataset: Year · Brand · Model/Trim · Price · Mileage · Engine Size · Fuel · Transmission · Condition")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Vehicles", f"{len(view):,}")
k2.metric("Median Price", currency(np.nanmedian(view[COL_PRICE])))
k3.metric("Avg Price", currency(np.nanmean(view[COL_PRICE])))
k4.metric("Median Mileage", f"{np.nanmedian(view[COL_MILEAGE]):,.0f}")
k5.metric("Top ValueScore", f"{np.nanmax(view['ValueScore']):.1f}")

# ------------------------- Altair selection (click to filter) -------------------------
# Shared selection: clicking brand bars filters other charts
brand_select = alt.selection_point(fields=[COL_BRAND], bind='legend')
year_select = alt.selection_point(fields=[COL_YEAR])

# Data for charts
chart_df = view.copy()
chart_df[COL_BRAND] = chart_df[COL_BRAND].astype(str)
chart_df[COL_MODEL] = chart_df[COL_MODEL].astype(str)

# Aggregates
agg_brand_year = chart_df.groupby([COL_YEAR, COL_BRAND])[COL_PRICE].mean().reset_index().rename(columns={COL_PRICE: "AvgPrice"})
agg_model_year = chart_df.groupby([COL_YEAR, COL_MODEL])[COL_PRICE].mean().reset_index().rename(columns={COL_PRICE: "AvgPrice"})

# Bar: Avg price by Brand faceted by Year (interactive)
bar = alt.Chart(agg_brand_year).mark_bar().encode(
    x=alt.X(f"{COL_BRAND}:N", title="Brand", sort="-y"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{COL_BRAND}:N", legend=alt.Legend(title="Brand")),
    column=alt.Column(f"{COL_YEAR}:N", title="Year"),
    tooltip=[COL_YEAR, COL_BRAND, alt.Tooltip("AvgPrice:Q", format=",.0f", title="Avg Price")],
).add_params(brand_select).transform_filter(brand_select).properties(height=280)

# Box plot: Price distribution by selected Brand/Model per Year
box = alt.Chart(chart_df).mark_boxplot(size=16).encode(
    x=alt.X(f"{COL_MODEL}:N", title="Model/Trim"),
    y=alt.Y(f"{COL_PRICE}:Q", title="Price"),
    color=alt.Color(f"{COL_BRAND}:N", legend=None),
    column=alt.Column(f"{COL_YEAR}:N", title="Year"),
    tooltip=[COL_YEAR, COL_BRAND, COL_MODEL, alt.Tooltip(COL_PRICE, format=",.0f", title="Price")]
).transform_filter(brand_select).properties(height=280)

# Trend: Brand avg price over Years (click year to highlight)
line = alt.Chart(agg_brand_year).mark_line(point=True).encode(
    x=alt.X(f"{COL_YEAR}:O", title="Year"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{COL_BRAND}:N", title="Brand"),
    tooltip=[COL_YEAR, COL_BRAND, alt.Tooltip("AvgPrice:Q", format=",.0f", title="Avg Price")],
).add_params(brand_select).transform_filter(brand_select).properties(height=280)

st.subheader("🗂️ Interactive Charts")
st.altair_chart(bar, use_container_width=True)
st.altair_chart(box, use_container_width=True)
st.altair_chart(line, use_container_width=True)

# ------------------------- Tabs for deeper interactivity -------------------------
tab_overview, tab_compare, tab_drill, tab_table = st.tabs(["Overview", "Compare", "Drill‑down", "Table"])

with tab_overview:
    st.markdown("### 🏅 Per‑Year Picks (Most & Least Worth Buying)")
    best, worst = best_worst_per_year(view, "ValueScore")
    cols = [COL_YEAR, COL_BRAND, COL_MODEL, COL_PRICE, COL_MILEAGE, COL_ENGINE, "ValueScore"]
    cols = [c for c in cols if c in best.columns]

    # Jump‑to‑Year selector to spotlight cards
    jump_year = st.select_slider("Jump to year", options=sorted(best[COL_YEAR].unique().tolist()),
                                 value=sorted(best[COL_YEAR].unique().tolist())[0])
    b_row = best[best[COL_YEAR] == jump_year].iloc[0] if len(best[best[COL_YEAR] == jump_year]) else None
    w_row = worst[worst[COL_YEAR] == jump_year].iloc[0] if len(worst[worst[COL_YEAR] == jump_year]) else None

    c1, c2 = st.columns(2)
    with c1:
        st.info("✅ **Most Worth Buying**")
        if b_row is not None:
            st.metric("Brand · Model", f"{b_row[COL_BRAND]} · {b_row[COL_MODEL]}")
            st.metric("Price", currency(b_row[COL_PRICE]), delta=f"ValueScore {b_row['ValueScore']:.1f}")
            st.metric("Mileage · Engine", f"{int(b_row[COL_MILEAGE]):,} · {b_row[COL_ENGINE]:.1f}L")
            if st.button("⭐ Add to favorites", key=f"fav_best_{jump_year}"):
                st.session_state["favorites"].add(int(b_row[COL_ID]))
    with c2:
        st.warning("❌ **Least Worth Buying**")
        if w_row is not None:
            st.metric("Brand · Model", f"{w_row[COL_BRAND]} · {w_row[COL_MODEL]}")
            st.metric("Price", currency(w_row[COL_PRICE]), delta=f"ValueScore {w_row['ValueScore']:.1f}")
            st.metric("Mileage · Engine", f"{int(w_row[COL_MILEAGE]):,} · {w_row[COL_ENGINE]:.1f}L")
            if st.button("⭐ Add to favorites", key=f"fav_worst_{jump_year}"):
                st.session_state["favorites"].add(int(w_row[COL_ID]))

    st.divider()
    st.markdown("#### Tables & Export")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Best (per Year)**")
        st.dataframe(best[cols], use_container_width=True)
        st.download_button("⬇️ Download Best (CSV)", best[cols].to_csv(index=False).encode("utf-8"),
                           file_name="best_worth_buying_per_year.csv")
    with cB:
        st.markdown("**Worst (per Year)**")
        st.dataframe(worst[cols], use_container_width=True)
        st.download_button("⬇️ Download Worst (CSV)", worst[cols].to_csv(index=False).encode("utf-8"),
                           file_name="least_worth_buying_per_year.csv")

with tab_compare:
    st.markdown("### 🔁 Brand / Model Comparator")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        brand_a = st.selectbox("Brand A", options=brands, index=0)
        model_a = st.selectbox("Model A", options=sorted(view[view[COL_BRAND]==brand_a][COL_MODEL].unique()), index=0)
    with c2:
        brand_b = st.selectbox("Brand B", options=brands, index=min(1, len(brands)-1))
        model_b = st.selectbox("Model B", options=sorted(view[view[COL_BRAND]==brand_b][COL_MODEL].unique()), index=0)
    with c3:
        st.caption("Charts respond to brand/model selections.")

    comp = view[
        ((view[COL_BRAND] == brand_a) & (view[COL_MODEL] == model_a)) |
        ((view[COL_BRAND] == brand_b) & (view[COL_MODEL] == model_b))
    ].copy()

    if comp.empty:
        st.info("No matching rows for the selected pairs—adjust filters.")
    else:
        # Violin / box hybrid (Altair workaround: layered distributions)
        dist = alt.Chart(comp).transform_fold(
            [COL_PRICE, COL_MILEAGE, "ValueScore"], as_=["Metric", "Value"]
        ).mark_boxplot(size=40).encode(
            x=alt.X("Metric:N", title="Metric"),
            y=alt.Y("Value:Q"),
            color=alt.Color(f"{COL_BRAND}:N"),
            column=alt.Column(f"{COL_YEAR}:N", title="Year"),
            tooltip=[COL_YEAR, COL_BRAND, COL_MODEL, "Metric", alt.Tooltip("Value:Q", format=",.2f")]
        ).properties(height=250)

        st.altair_chart(dist, use_container_width=True)

        # KPI compare
        kpa, kpb, kpc, kpd = st.columns(4)
        kpa.metric(f"{brand_a} {model_a}: Avg Price", currency(comp[(comp[COL_BRAND]==brand_a) & (comp[COL_MODEL]==model_a)][COL_PRICE].mean()))
        kpb.metric(f"{brand_b} {model_b}: Avg Price", currency(comp[(comp[COL_BRAND]==brand_b) & (comp[COL_MODEL]==model_b)][COL_PRICE].mean()))
        kpc.metric(f"{brand_a} {model_a}: Avg ValueScore", f"{comp[(comp[COL_BRAND]==brand_a)&(comp[COL_MODEL]==model_a)]['ValueScore'].mean():.1f}")
        kpd.metric(f"{brand_b} {model_b}: Avg ValueScore", f"{comp[(comp[COL_BRAND]==brand_b)&(comp[COL_MODEL]==model_b)]['ValueScore'].mean():.1f}")

with tab_drill:
    st.markdown("### 🔎 Drill‑down (click selects → table updates)")
    # Allow user to pick a Year & Brand to drill
    drill_year = st.selectbox("Year", options=years, index=years.index(min(years)))
    drill_brand = st.selectbox("Brand", options=brands, index=0)
    drill = view[(view[COL_YEAR] == drill_year) & (view[COL_BRAND] == drill_brand)].copy()

    st.write(f"**{drill_brand} — {drill_year}** ({len(drill)} vehicles)")
    scatter = alt.Chart(drill).mark_circle(size=90, opacity=0.75).encode(
        x=alt.X(f"{COL_MILEAGE}:Q", title="Mileage"),
        y=alt.Y(f"{COL_PRICE}:Q", title="Price"),
        color=alt.Color("ValueScore:Q", scale=alt.Scale(scheme="greenblue"), title="ValueScore"),
        tooltip=[COL_MODEL, alt.Tooltip(COL_PRICE, format=",.0f", title="Price"),
                 alt.Tooltip(COL_MILEAGE, format=",.0f", title="Mileage"),
                 alt.Tooltip("ValueScore:Q", format=",.1f", title="ValueScore")]
    ).interactive().properties(height=320)
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("**Vehicles**")
    cols_show = [COL_ID, COL_YEAR, COL_BRAND, COL_MODEL, COL_PRICE, COL_MILEAGE, COL_ENGINE, COL_FUEL, COL_TRANS, COL_COND, "ValueScore"]
    cols_show = [c for c in cols_show if c in drill.columns]
    st.dataframe(drill[cols_show].sort_values(by=["ValueScore"], ascending=False), use_container_width=True)

    # Add favorites via selection
    cfa, cfb = st.columns(2)
    add_id = cfa.number_input("Add Car ID to favorites", min_value=0, step=1)
    if cfa.button("⭐ Add"):
        st.session_state["favorites"].add(int(add_id))
        st.toast(f"Added Car ID {int(add_id)} to favorites.", icon="⭐")
    if cfb.button("🗑️ Clear favorites"):
        st.session_state["favorites"] = set()
        st.toast("Favorites cleared.", icon="🗑️")

with tab_table:
    st.markdown("### 🔢 Full Table & Export")
    brand_filter = st.text_input("Brand contains", value="")
    model_filter = st.text_input("Model/Trim contains", value="")
    tview = view.copy()
    if brand_filter.strip():
        tview = tview[tview[COL_BRAND].str.contains(brand_filter, case=False, na=False)]
    if model_filter.strip():
        tview = tview[tview[COL_MODEL].str.contains(model_filter, case=False, na=False)]

    # Highlight favorites
    tview["⭐ Favorite"] = tview[COL_ID].astype(int).isin(st.session_state["favorites"])
    show_cols = [COL_ID, "⭐ Favorite", COL_YEAR, COL_BRAND, COL_MODEL, COL_PRICE, COL_MILEAGE, COL_ENGINE, COL_FUEL, COL_TRANS, COL_COND, "ValueScore"]
    show_cols = [c for c in show_cols if c in tview.columns]
    st.dataframe(tview[show_cols].sort_values(by=[COL_YEAR, COL_BRAND, COL_MODEL]), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download current view (CSV)",
                       tview[show_cols].to_csv(index=False).encode("utf-8"),
                       file_name="car_portfolio_view.csv")
    c2.download_button("⬇️ Download favorites only (CSV)",
                       tview[tview["⭐ Favorite"]][show_cols].to_csv(index=False).encode("utf-8"),
                       file_name="favorites.csv")

# ------------------------- Footer -------------------------
with st.expander("ℹ️ Tips & About", expanded=False):
    st.markdown(f"""
**Dataset columns**: `{', '.join(EXPECTED_COLUMNS)}`  
**ValueScore** is min‑max normalized within each **Year**:
- **Price** (Minimize), **Mileage** (Minimize), **Engine Size** (direction configurable).  
Tune weights in the sidebar; charts respond to clicks/legend selections.
""")
