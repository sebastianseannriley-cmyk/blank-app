
# app.py
# Interactive Streamlit dashboard tailored to data.csv:
# Year → Brand → Model portfolio with filters, comparator,
# Model X/Model Y handling (Model Y removed from table views; X chart shows X+Y),
# Model X override (absolute & inflation-adjusted), and purge of pre‑2016 Model X rows.

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------- Page Config -------------------------
st.set_page_config(
    page_title="Car Price Portfolio (Interactive + Model X/Y Fix)",
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
    removed_x_pre2016 = pre_count - len(df)

    # 🔧 CHANGE: Remove all Tesla Model Y rows from the dataset (they won't appear in tables/charts)
    pre_count2 = len(df)
    df = df[~((df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model Y"))].copy()
    removed_all_model_y = pre_count2 - len(df)

    return df, removed_x_pre2016, removed_all_model_y

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
    try:
        return f"${x:,.0f}"
    except Exception:
        return "-"

# ------------------------- Load base data & purge report -------------------------
DATA_PATH = "data.csv"
df, removed_x_pre2016, removed_all_model_y = load_data(DATA_PATH)

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
    modelx_msg = (
        "The original dataset for the Tesla Model X/Y did not accurately model absolute or inflation-adjusted prices; "
        "to account for this, external data was added for Model X."
    )

# Merge override series (left merge on Year for Tesla Model X only)
if modelx_df is not None and not modelx_df.empty:
    # NOTE: We already removed Model Y rows from df. We still override Model X prices where provided.
    non_x = df[~((df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model X"))].copy()
    x_rows = df[(df[COL_BRAND] == "Tesla") & (df[COL_MODEL] == "Model X")].copy()

    # Merge series onto Model X rows by Year
    x_rows = x_rows.merge(modelx_df, how="left", left_on=COL_YEAR, right_on="Year")
    x_rows.loc[~x_rows["AbsolutePrice_USD"].isna(), COL_PRICE] = x_rows["AbsolutePrice_USD"]

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
price_sel = st.sidebar.slider(
    "Price range",
    min_value=min_price, max_value=max_price,
    value=(min_price, max_price),
    step=max(1.0, (max_price - min_price) / 100.0)
)

st.sidebar.divider()
st.sidebar.header("Value Score")
st.sidebar.number_input("Weight: Price (Minimize)", 0.0, 5.0, step=0.1, value=st.session_state["w_price"], key="w_price")
st.sidebar.number_input("Weight: Mileage (Minimize)", 0.0, 5.0, step=0.1, value=st.session_state["w_mileage"], key="w_mileage")
st.sidebar.number_input("Weight: Engine Size", 0.0, 5.0, step=0.1, value=st.session_state["w_engine"], key="w_engine")
st.sidebar.radio(
    "Engine Size direction",
    ["Minimize", "Maximize"],
    index=0 if st.session_state["engine_dir"] == "Minimize" else 1,
    key="engine_dir"
)

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
st.caption("Dataset: Year · Brand · Model/Trim · Price · Mileage · Engine Size · Fuel · Transmission · Condition")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Vehicles", f"{len(view):,}")
k2.metric("Median Price", currency(np.nanmedian(view[COL_PRICE])))
k3.metric("Avg Price", currency(np.nanmean(view[COL_PRICE])))
k4.metric("Median Mileage", f"{np.nanmedian(view[COL_MILEAGE]):,.0f}")
k5.metric("Top ValueScore", f"{np.nanmax(view['ValueScore']):.1f}")

# Purge notice
with st.expander("Data Housekeeping"):
    st.markdown(f"- Purged **{removed_x_pre2016:,}** synthetic Tesla Model X rows (years < 2016).")
    st.markdown(f"- Removed **{removed_all_model_y:,}** Tesla Model Y rows from dataset (rolled into Model X chart).")
    if modelx_msg:
        st.warning(modelx_msg)
    elif modelx_df is not None:
        yrs = ", ".join([str(int(y)) for y in sorted(modelx_df["Year"].unique())])
        st.success(
            f"Loaded Model X US price series for years: {yrs}. "
            f"Absolute prices override base listing price for Model X."
        )

# ------------------------- Model X(+Y) price panel -------------------------
st.subheader("🟢 Tesla Model X chart (Model X + Model Y combined)")
# 🔧 CHANGE: Build chart from original raw data (including Model Y), not from df where Y was removed.
# We need X+Y combined series per year. If modelx_prices.csv is present, use AbsolutePrice_USD to override X rows only.

# Safely reload the raw CSV to access Model Y
try:
    raw_df = pd.read_csv(DATA_PATH)
    # Clean typing
    raw_df[COL_YEAR] = pd.to_numeric(raw_df[COL_YEAR], errors="coerce")
    raw_df[COL_PRICE] = pd.to_numeric(raw_df[COL_PRICE], errors="coerce")
    raw_df[COL_BRAND] = raw_df[COL_BRAND].astype(str).str.strip()
    raw_df[COL_MODEL] = raw_df[COL_MODEL].astype(str).str.strip()
except Exception:
    raw_df = None
    st.warning("Could not reload raw data for X+Y chart; showing Model X only from current view.")

if raw_df is not None:
    # Remove Model X < 2016, keep X 2016+ and all Y for combination
    raw_xy = raw_df[
        (raw_df[COL_BRAND] == "Tesla") &
        (raw_df[COL_MODEL].isin(["Model X", "Model Y"]))
    ].copy()
    raw_xy = raw_xy[~((raw_xy[COL_MODEL] == "Model X") & (raw_xy[COL_YEAR] < 2016))]

    # If modelx_prices.csv exists, override Model X prices
    if modelx_df is not None and not modelx_df.empty:
        mx = raw_xy[raw_xy[COL_MODEL] == "Model X"].copy()
        mx = mx.merge(modelx_df, how="left", left_on=COL_YEAR, right_on="Year")
        mx.loc[~mx["AbsolutePrice_USD"].isna(), COL_PRICE] = mx["AbsolutePrice_USD"]
        mx = mx.drop(columns=["Year"])
        my = raw_xy[raw_xy[COL_MODEL] == "Model Y"].copy()
        raw_xy = pd.concat([mx, my], ignore_index=True)

    # 🔧 CHANGE: Combine X + Y by year using mean price (you can switch to median if preferred)
    comb = (
        raw_xy.groupby(COL_YEAR, dropna=True)[COL_PRICE]
        .mean()
        .reset_index()
        .rename(columns={COL_PRICE: "Combined_XY_USD"})
    )

    # Optional inflation line: if modelx_df has InflationAdjusted_USD, we show it as Model X adjusted
    if modelx_df is not None and "InflationAdjusted_USD" in modelx_df.columns:
        infl = modelx_df[["Year", "InflationAdjusted_USD"]].rename(columns={"Year": COL_YEAR})
        plot_df = comb.merge(infl, on=COL_YEAR, how="left")
    else:
        plot_df = comb.copy()
        plot_df["InflationAdjusted_USD"] = np.nan

    plot_df = plot_df.dropna(subset=[COL_YEAR]).sort_values(COL_YEAR)

    if plot_df.empty:
        st.info("No Model X/Y data available for chart.")
    else:
        m = plot_df.melt(
            id_vars=[COL_YEAR],
            value_vars=["Combined_XY_USD", "InflationAdjusted_USD"],
            var_name="Series",
            value_name="PriceUSD"
        )

        # Pretty names
        m["Series"] = m["Series"].map({
            "Combined_XY_USD": "Combined X + Y (avg)",
            "InflationAdjusted_USD": "Model X (inflation‑adj)"
        }).fillna("Series")

        line_xy = (
            alt.Chart(m)
            .mark_line(point=True, interpolate="monotone")
            .encode(
                x=alt.X(f"{COL_YEAR}:O", title="Year"),
                y=alt.Y("PriceUSD:Q", title="Price (USD)", axis=alt.Axis(format="~s")),
                color=alt.Color("Series:N", title="Series", scale=alt.Scale(scheme="teals")),
                tooltip=[
                    alt.Tooltip(COL_YEAR, title="Year"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("PriceUSD:Q", format=",.0f", title="Price"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(line_xy, use_container_width=True)

        cta, ctb = st.columns(2)
        with cta:
            st.markdown("**Combined (X + Y) price table**")
            st.dataframe(
                plot_df.rename(columns={
                    "Combined_XY_USD": "Combined X+Y (USD)",
                    "InflationAdjusted_USD": "Inflation‑adj (USD)"
                }),
                use_container_width=True
            )
        with ctb:
            st.download_button(
                "⬇️ Download X+Y series (CSV)",
                plot_df.to_csv(index=False).encode("utf-8"),
                file_name="tesla_model_x_plus_y_series.csv"
            )
else:
    # Fallback: show Model X from filtered df only
    x_panel = view[(view[COL_BRAND] == "Tesla") & (view[COL_MODEL] == "Model X")].copy()
    plot_df = x_panel[[COL_YEAR, COL_PRICE]].rename(columns={COL_PRICE: "Combined_XY_USD"}).copy()
    plot_df["InflationAdjusted_USD"] = np.nan
    plot_df = plot_df.dropna(subset=[COL_YEAR]).drop_duplicates(COL_YEAR).sort_values(COL_YEAR)
    if plot_df.empty:
        st.info("No Model X data available in the current filter.")
    else:
        m = plot_df.melt(
            id_vars=[COL_YEAR],
            value_vars=["Combined_XY_USD", "InflationAdjusted_USD"],
            var_name="Series",
            value_name="PriceUSD"
        )
        m["Series"] = m["Series"].map({
            "Combined_XY_USD": "Combined X + Y (avg)",
            "InflationAdjusted_USD": "Model X (inflation‑adj)"
        })
        line = (
            alt.Chart(m)
            .mark_line(point=True, interpolate="monotone")
            .encode(
                x=alt.X(f"{COL_YEAR}:O", title="Year"),
                y=alt.Y("PriceUSD:Q", title="Price (USD)", axis=alt.Axis(format="~s")),
                color=alt.Color("Series:N", title="Series", scale=alt.Scale(scheme="teals")),
                tooltip=[
                    alt.Tooltip(COL_YEAR, title="Year"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("PriceUSD:Q", format=",.0f", title="Price"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(line, use_container_width=True)

# ------------------------- Interactive charts -------------------------
st.subheader("🗂️ Interactive Charts")

chart_df = view.copy()
chart_df[COL_BRAND] = chart_df[COL_BRAND].astype(str)
chart_df[COL_MODEL] = chart_df[COL_MODEL].astype(str)

agg_brand_year = (
    chart_df.groupby([COL_YEAR, COL_BRAND])[COL_PRICE]
    .mean()
    .reset_index()
    .rename(columns={COL_PRICE: "AvgPrice"})
)
agg_model_year = (
    chart_df.groupby([COL_YEAR, COL_MODEL])[COL_PRICE]
    .mean()
    .reset_index()
    .rename(columns={COL_PRICE: "AvgPrice"})
)

brand_select = alt.selection_point(fields=[COL_BRAND], bind="legend")

bar = (
    alt.Chart(agg_brand_year)
    .mark_bar()
    .encode(
        x=alt.X(f"{COL_BRAND}:N", title="Brand", sort="-y"),
        y=alt.Y("AvgPrice:Q", title="Average Price"),
        color=alt.Color(f"{COL_BRAND}:N", legend=alt.Legend(title="Brand")),
        column=alt.Column(f"{COL_YEAR}:N", title="Year"),
        tooltip=[
            alt.Tooltip(COL_YEAR, title="Year"),
            alt.Tooltip(COL_BRAND, title="Brand"),
            alt.Tooltip("AvgPrice:Q", format=",.0f", title="Avg Price"),
        ],
    )
    .add_params(brand_select)
    .transform_filter(brand_select)
    .properties(height=280)
)
st.altair_chart(bar, use_container_width=True)

box = (
    alt.Chart(chart_df)
    .mark_boxplot(size=16)
    .encode(
        x=alt.X(f"{COL_MODEL}:N", title="Model/Trim"),
        y=alt.Y(f"{COL_PRICE}:Q", title="Price"),
        color=alt.Color(f"{COL_BRAND}:N", legend=None),
        column=alt.Column(f"{COL_YEAR}:N", title="Year"),
        tooltip=[
            alt.Tooltip(COL_YEAR, title="Year"),
            alt.Tooltip(COL_BRAND, title="Brand"),
            alt.Tooltip(COL_MODEL, title="Model"),
            alt.Tooltip(COL_PRICE, format=",.0f", title="Price"),
        ],
    )
    .transform_filter(brand_select)
    .properties(height=280)
)
st.altair_chart(box, use_container_width=True)

line_brand = (
    alt.Chart(agg_brand_year)
    .mark_line(point=True)
    .encode(
        x=alt.X(f"{COL_YEAR}:O", title="Year"),
        y=alt.Y("AvgPrice:Q", title="Average Price"),
        color=alt.Color(f"{COL_BRAND}:N", title="Brand"),
        tooltip=[
            alt.Tooltip(COL_YEAR, title="Year"),
            alt.Tooltip(COL_BRAND, title="Brand"),
            alt.Tooltip("AvgPrice:Q", format=",.0f", title="Avg Price"),
        ],
    )
    .add_params(brand_select)
    .transform_filter(brand_select)
    .properties(height=280)
)
st.altair_chart(line_brand, use_container_width=True)

# ------------------------- Overview (Best/Worst via slider) & Compare -------------------------
tab_overview, tab_compare = st.tabs(["Overview", "Compare"])

with tab_overview:
    st.markdown("### 🏅 Per‑Year Picks (Most & Least Worth Buying)")
    best, worst = best_worst_per_year(view, "ValueScore")

    year_min, year_max = 2000, 2023
    selected_year = st.slider(
        "Drag to pick a year",
        min_value=year_min,
        max_value=year_max,
        value=max(year_min, min(year_max, int(np.nanmin(view[COL_YEAR])) if len(view) else year_min)),
        step=1
    )

    best_row = best[best[COL_YEAR] == selected_year]
    worst_row = worst[worst[COL_YEAR] == selected_year]

    c1, c2 = st.columns(2)
    with c1:
        st.info("✅ **Most Worth Buying**")
        if not best_row.empty:
            r = best_row.iloc[0]
            st.metric("Brand · Model", f"{r[COL_BRAND]} · {r[COL_MODEL]}")
            st.metric("Price", currency(r[COL_PRICE]), delta=f"ValueScore {r['ValueScore']:.1f}")
            st.metric("Mileage · Engine", f"{int(r[COL_MILEAGE]):,} · {r[COL_ENGINE] if not pd.isna(r[COL_ENGINE]) else '—'}")
        else:
            st.caption("No data for the selected year.")

    with c2:
        st.warning("❌ **Least Worth Buying**")
        if not worst_row.empty:
            r = worst_row.iloc[0]
            st.metric("Brand · Model", f"{r[COL_BRAND]} · {r[COL_MODEL]}")
            st.metric("Price", currency(r[COL_PRICE]), delta=f"ValueScore {r['ValueScore']:.1f}")
            st.metric("Mileage · Engine", f"{int(r[COL_MILEAGE]):,} · {r[COL_ENGINE] if not pd.isna(r[COL_ENGINE]) else '—'}")
        else:
            st.caption("No data for the selected year.")

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
        cols_for_chart = [COL_YEAR, COL_BRAND, COL_MODEL, COL_PRICE, COL_MILEAGE, "ValueScore"]
        long = comp[cols_for_chart].melt(
            id_vars=[COL_YEAR, COL_BRAND, COL_MODEL],
            value_vars=[COL_PRICE, COL_MILEAGE, "ValueScore"],
            var_name="Metric",
            value_name="Value"
        )
        long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
        long = long.dropna(subset=["Value", COL_YEAR])

        dist = (
            alt.Chart(long)
            .mark_boxplot(size=40)
            .encode(
                x=alt.X("Metric:N", title="Metric"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color(f"{COL_BRAND}:N", title="Brand"),
                facet=alt.Facet(f"{COL_YEAR}:N", title="Year"),
                tooltip=[
                    alt.Tooltip(COL_YEAR, title="Year"),
                    alt.Tooltip(COL_BRAND, title="Brand"),
                    alt.Tooltip(COL_MODEL, title="Model"),
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Value:Q", format=",.2f", title="Value"),
                ],
            )
            .properties(height=250)
        )
        st.altair_chart(dist, use_container_width=True)

        kpa, kpb, kpc, kpd = st.columns(4)
        kpa.metric(
            f"{brand_a} {model_a}: Avg Price",
            currency(comp[(comp[COL_BRAND]==brand_a) & (comp[COL_MODEL]==model_a)][COL_PRICE].mean())
        )
        kpb.metric(
            f"{brand_b} {model_b}: Avg Price",
            currency(comp[(comp[COL_BRAND]==brand_b) & (comp[COL_MODEL]==model_b)][COL_PRICE].mean())
        )
        kpc.metric(
            f"{brand_a} {model_a}: Avg ValueScore",
            f"{comp[(comp[COL_BRAND]==brand_a) & (comp[COL_MODEL]==model_a)]['ValueScore'].mean():.1f}"
        )
        kpd.metric(
            f"{brand_b} {model_b}: Avg ValueScore",
            f"{comp[(comp[COL_BRAND]==brand_b) & (comp[COL_MODEL]==model_b)]['ValueScore'].mean():.1f}"
        )

# ------------------------- Footer -------------------------
with st.expander("ℹ️ Notes", expanded=False):
    st.markdown(
        "- Removed **Tesla Model Y** rows from dataset (rolled into the Model X chart as X+Y combined).\n"
        "- All Tesla Model X rows with Year < 2016 are removed.\n"
        "- If `modelx_prices.csv` is present, it overrides Model X prices for those years.\n"
        "- Best/Worst uses a year slider; Drill‑down and Table views are not included."
    )
