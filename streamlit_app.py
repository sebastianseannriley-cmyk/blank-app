
# app.py
# Streamlit interactive portfolio of car prices by year/brand with value scoring.
# Author: Riley, Sebastian-Seann B CDT 2029 (built by M365 Copilot)

import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------- Page Config -------------------------
st.set_page_config(
    page_title="Car Price Portfolio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Helpers -------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # strip whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    return df

def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """Convert prices/odometer-like columns to numeric (handles $ and commas)."""
    # If already numeric, just return
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    # Otherwise strip currency, commas, non-numeric
    cleaned = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def guess_column(df: pd.DataFrame, candidates, numeric: bool | None = None):
    """Try to auto-guess a column by name substring."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                if numeric is None:
                    return orig
                if numeric and pd.api.types.is_numeric_dtype(df[orig]):
                    return orig
                if numeric and not pd.api.types.is_numeric_dtype(df[orig]):
                    # Check if convertible
                    try:
                        _ = pd.to_numeric(df[orig], errors="coerce")
                        return orig
                    except Exception:
                        pass
    return None

def minmax_by_group(s: pd.Series, group_index: pd.Series) -> pd.Series:
    """Min-max normalize within groups (e.g., per year)."""
    df = pd.DataFrame({"val": s, "grp": group_index})
    def _mm(x):
        v = x["val"]
        vmax, vmin = v.max(), v.min()
        if pd.isna(vmax) or pd.isna(vmin) or vmax == vmin:
            return pd.Series(np.full(len(v), 0.5), index=x.index)
        return (v - vmin) / (vmax - vmin)
    return df.groupby("grp", dropna=False).apply(_mm).reset_index(level=0, drop=True)

def build_value_score(
    data: pd.DataFrame,
    year_col: str,
    feature_config: pd.DataFrame,
) -> pd.Series:
    """
    feature_config: DataFrame with columns:
      ['feature', 'weight', 'direction'] where direction in {'Minimize','Maximize'}
    Returns: pd.Series of ValueScore.
    """
    score = pd.Series(np.zeros(len(data)), index=data.index, dtype=float)

    # Normalize selected features per year and aggregate weighted score
    for _, row in feature_config.iterrows():
        feat = row["feature"]
        w = float(row["weight"])
        direction = row["direction"]
        if feat not in data.columns or w == 0:
            continue

        # Numeric cleanup for robust normalization
        col = data[feat].copy()
        col = _to_numeric_clean(col)

        # Compute per-year minmax normalization
        norm = minmax_by_group(col, data[year_col])

        # Invert if minimizing (lower is better)
        adj = (1 - norm) if direction == "Minimize" else norm
        score = score + w * adj

    # Scale to 0-100 for readability
    score_min, score_max = score.min(), score.max()
    if pd.isna(score_min) or pd.isna(score_max) or score_max == score_min:
        return pd.Series(np.full(len(score), 50.0), index=score.index)
    return 100.0 * (score - score_min) / (score_max - score_min)

def best_worst_per_year(df: pd.DataFrame, year_col: str, score_col: str):
    """Return DataFrames with best and worst vehicles per year."""
    # idxmax / idxmin per year, ignore NaN score rows
    valid = df.dropna(subset=[score_col])
    gb = valid.groupby(year_col, dropna=False)
    best_idx = gb[score_col].idxmax()
    worst_idx = gb[score_col].idxmin()

    best = valid.loc[best_idx].sort_values(by=year_col, ascending=True)
    worst = valid.loc[worst_idx].sort_values(by=year_col, ascending=True)
    return best, worst

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return ""

# ------------------------- Sidebar: Data Input -------------------------
st.sidebar.title("🚗 Data & Settings")

uploaded = st.sidebar.file_uploader(
    "Upload your CSV file of vehicle listings",
    type=["csv"],
    help="Include columns such as Year, Brand/Make, Model, Price, Mileage, MPG, Body, Fuel, etc.",
)

# Optional demo dataset if no file provided
use_demo = False
if uploaded is None:
    use_demo = st.sidebar.checkbox("Use demo dataset (synthetic)", value=True)
else:
    use_demo = False

if use_demo:
    # Synthetic demo dataset (for illustration only)
    rng = np.random.default_rng(42)
    years = rng.choice(np.arange(2015, 2025), size=800, replace=True)
    brands = rng.choice(["Toyota", "Honda", "Ford", "BMW", "Tesla", "Chevrolet", "Hyundai", "Kia"], size=800)
    models_map = {
        "Toyota": ["Camry", "Corolla", "RAV4"],
        "Honda": ["Civic", "Accord", "CR-V"],
        "Ford": ["F-150", "Escape", "Fusion"],
        "BMW": ["3 Series", "X3", "5 Series"],
        "Tesla": ["Model 3", "Model Y", "Model S"],
        "Chevrolet": ["Silverado", "Equinox", "Malibu"],
        "Hyundai": ["Elantra", "Sonata", "Tucson"],
        "Kia": ["Optima", "Sorento", "Sportage"],
    }
    models = [rng.choice(models_map[b]) for b in brands]
    price = (20000 + (years - 2015) * 1200 + rng.normal(0, 4000, size=800)).clip(8000, 90000)
    mileage = (rng.normal(60000, 25000, size=800) - (years - 2015) * 5000).clip(0, 200000)
    mpg = (rng.normal(30, 5, size=800) + (brands == "Tesla") * 60).clip(12, 120)
    body = rng.choice(["Sedan", "SUV", "Truck", "Coupe", "Hatchback"], size=800)
    fuel = np.where(brands == "Tesla", "Electric", rng.choice(["Gasoline", "Hybrid", "Diesel"], size=800))
    df = pd.DataFrame({
        "Year": years,
        "Brand": brands,
        "Model": models,
        "Price": price.round(0),
        "Mileage": mileage.round(0),
        "MPG_or_EQ": mpg.round(1),  # treat as efficiency metric (MPG or electric equivalent)
        "Body": body,
        "Fuel": fuel,
    })
    st.sidebar.info("Using a synthetic demo dataset. In your repo, upload your CSV to use real data.")
else:
    if uploaded is not None:
        df = load_csv(uploaded)
    else:
        st.stop()

# ------------------------- Column Selection -------------------------
st.header("Interactive Car Price Portfolio")

st.markdown("""
Upload a CSV, map the key columns (Year, Brand, Model, Price), and explore.
Define a flexible **Value Score** per vehicle to find the **Most** and **Least Worth Buying** per year.
""")

# Attempt to guess sensible defaults
year_guess = guess_column(df, ["year"], numeric=None) or "Year" if "Year" in df.columns else None
brand_guess = guess_column(df, ["brand", "make", "manufacturer"], numeric=None)
model_guess = guess_column(df, ["model", "trim"], numeric=None)
price_guess = guess_column(df, ["price", "msrp", "listing_price"], numeric=None)

with st.expander("🔧 Map your CSV columns", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    year_col = col1.selectbox("Year column", options=df.columns, index=(df.columns.get_loc(year_guess) if year_guess in df.columns else 0))
    brand_col = col2.selectbox("Brand/Make column", options=df.columns, index=(df.columns.get_loc(brand_guess) if brand_guess in df.columns else 0))
    model_col = col3.selectbox("Model/Trim column", options=df.columns, index=(df.columns.get_loc(model_guess) if model_guess in df.columns else 0))
    price_col = col4.selectbox("Price column", options=df.columns, index=(df.columns.get_loc(price_guess) if price_guess in df.columns else 0))

    # Ensure Year is numeric for grouping
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    # Clean price column to numeric
    df[price_col] = _to_numeric_clean(df[price_col])

# ------------------------- Categorization & Filters -------------------------
# Identify categorical and numeric columns for grouping/scoring
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in [brand_col, model_col]]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

with st.sidebar.expander("🧭 Filter & Grouping", expanded=True):
    years_available = sorted([y for y in df[year_col].dropna().unique()])
    years_selected = st.multiselect("Filter by Year", options=years_available, default=years_available)
    group_by_col = st.selectbox(
        "Primary grouping for charts (within Year)",
        options=[brand_col] + cat_cols,
        index=0,
        help="Default is Brand/Make; choose other categories like Body, Fuel, etc."
    )

    secondary_group_col = st.selectbox(
        "Optional secondary grouping",
        options=["(none)"] + [c for c in cat_cols if c != group_by_col],
        index=0,
        help="Refine the grouping even more (e.g., Brand → Fuel)."
    )

# Apply Year filter
df_view = df[df[year_col].isin(years_selected)] if len(years_selected) else df.copy()

# ------------------------- Value Score Config -------------------------
st.subheader("🎯 Value Score (customizable)")
st.markdown("""
Select features and weights to compute a **ValueScore (0–100)** per vehicle (normalized within each year).
- Set **direction** to *Minimize* for cost-like metrics (Price, Mileage).
- Set **direction** to *Maximize* for benefit-like metrics (MPG, Horsepower, SafetyRating).
- The app will pick **Most Worth Buying** (highest score) and **Least Worth Buying** (lowest score) per year.
""")

# Pick features for scoring
default_features = []
if price_col in df.columns:
    default_features.append(price_col)
if "Mileage" in df.columns:
    default_features.append("Mileage")
if "MPG" in df.columns:
    default_features.append("MPG")
if "MPG_or_EQ" in df.columns:
    default_features.append("MPG_or_EQ")

features_selected = st.multiselect(
    "Select numeric features to include in the Value Score",
    options=num_cols,
    default=list(dict.fromkeys(default_features)),  # de-duplicate while preserving order
)

# Build editable config table
if len(features_selected) == 0:
    st.info("Select at least one numeric feature (e.g., Price, Mileage, MPG) to compute the Value Score.")
else:
    initial_rows = []
    for f in features_selected:
        # Heuristic default: Price & Mileage → Minimize; others → Maximize
        dir_default = "Minimize" if re.search(r"price|msrp|cost|mile|odometer", f, re.I) else "Maximize"
        wt_default = 1.0 if f == price_col else 0.8 if re.search(r"mile|odometer", f, re.I) else 0.6
        initial_rows.append({"feature": f, "weight": wt_default, "direction": dir_default})

    feature_config = st.data_editor(
        pd.DataFrame(initial_rows),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "feature": st.column_config.SelectboxColumn("Feature", options=num_cols),
            "weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=5.0, step=0.1),
            "direction": st.column_config.SelectboxColumn("Direction", options=["Minimize", "Maximize"]),
        },
        hide_index=True
    )

    # Compute Value Score
    df_view["ValueScore"] = build_value_score(df_view, year_col=year_col, feature_config=feature_config)

# ------------------------- Summary KPIs -------------------------
st.subheader("📈 Price Insights")
colA, colB, colC, colD = st.columns(4)
try:
    colA.metric("Vehicles in selection", f"{len(df_view):,}")
    colB.metric("Median Price", f"${np.nanmedian(df_view[price_col]):,.0f}")
    colC.metric("Average Price", f"${np.nanmean(df_view[price_col]):,.0f}")
    colD.metric("Price Std Dev", f"${np.nanstd(df_view[price_col]):,.0f}")
except Exception:
    st.warning("Price metrics unavailable due to missing or non-numeric data in the Price column.")

# ------------------------- Charts -------------------------
st.subheader("🗂️ Prices by Year → Category")

chart_data = df_view.copy()
chart_data[group_by_col] = chart_data[group_by_col].astype(str)

if secondary_group_col != "(none)":
    chart_data[secondary_group_col] = chart_data[secondary_group_col].astype(str)

# Average price by group per year
gb_cols = [year_col, group_by_col] + ([secondary_group_col] if secondary_group_col != "(none)" else [])
agg = chart_data.groupby(gb_cols, dropna=False)[price_col].mean().reset_index().rename(columns={price_col: "AvgPrice"})

# Bar chart
color_col = secondary_group_col if secondary_group_col != "(none)" else group_by_col
bar = alt.Chart(agg).mark_bar().encode(
    x=alt.X(f"{group_by_col}:N", title="Category"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{color_col}:N", legend=alt.Legend(title=color_col)),
    column=alt.Column(f"{year_col}:N", title="Year"),
    tooltip=[year_col, group_by_col] + ([secondary_group_col] if secondary_group_col != "(none)" else []) + ["AvgPrice"]
).properties(height=300)

st.altair_chart(bar, use_container_width=True)

# Box plot of prices by brand per year
st.subheader("📦 Price Distribution (Box Plot)")
box = alt.Chart(chart_data).mark_boxplot(size=18).encode(
    x=alt.X(f"{group_by_col}:N", title="Category"),
    y=alt.Y(f"{price_col}:Q", title="Price"),
    color=alt.Color(f"{group_by_col}:N", legend=None),
    column=alt.Column(f"{year_col}:N", title="Year"),
    tooltip=[year_col, group_by_col, price_col]
).properties(height=300)

st.altair_chart(box, use_container_width=True)

# Trend line per brand (optional)
st.subheader("📉 Average Price Trend by Brand")
trend = chart_data.groupby([year_col, brand_col], dropna=False)[price_col].mean().reset_index().rename(columns={price_col: "AvgPrice"})
line = alt.Chart(trend).mark_line(point=True).encode(
    x=alt.X(f"{year_col}:O", title="Year"),
    y=alt.Y("AvgPrice:Q", title="Average Price"),
    color=alt.Color(f"{brand_col}:N", title="Brand"),
    tooltip=[year_col, brand_col, "AvgPrice"]
).properties(height=300)
st.altair_chart(line, use_container_width=True)

# ------------------------- Best & Worst per Year -------------------------
st.subheader("🏅 Most & Least Worth Buying per Year")

if "ValueScore" in df_view.columns:
    best, worst = best_worst_per_year(df_view, year_col=year_col, score_col="ValueScore")

    # Display side-by-side tables
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Most Worth Buying")
        cols_to_show = [year_col, brand_col, model_col, price_col, "ValueScore"]
        cols_to_show = [c for c in cols_to_show if c in best.columns]
        st.dataframe(best[cols_to_show].sort_values(by=year_col), use_container_width=True)

    with c2:
        st.markdown("### ❌ Least Worth Buying")
        cols_to_show = [year_col, brand_col, model_col, price_col, "ValueScore"]
        cols_to_show = [c for c in cols_to_show if c in worst.columns]
        st.dataframe(worst[cols_to_show].sort_values(by=year_col), use_container_width=True)

    # Download buttons
    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "⬇️ Download Most Worth Buying (CSV)",
        data=best.to_csv(index=False).encode("utf-8"),
        file_name="best_worth_buying_per_year.csv",
        mime="text/csv",
    )
    col_dl2.download_button(
        "⬇️ Download Least Worth Buying (CSV)",
        data=worst.to_csv(index=False).encode("utf-8"),
        file_name="least_worth_buying_per_year.csv",
        mime="text/csv",
    )
else:
    st.info("Select features and weights above to compute the Value Score and see per-year recommendations.")

# ------------------------- Full Table & Filters -------------------------
st.subheader("🔍 Explore Listings")
with st.expander("Filters", expanded=False):
    # Simple text filters for brand/model
    brand_filter = st.text_input("Filter Brand (contains)", value="")
    model_filter = st.text_input("Filter Model (contains)", value="")
    min_price = st.number_input("Min Price", min_value=0, value=0)
    max_price = st.number_input("Max Price", min_value=0, value=int(np.nanmax(df_view[price_col]) if df_view[price_col].notna().any() else 100000))

filtered = df_view.copy()
if brand_filter.strip():
    filtered = filtered[filtered[brand_col].astype(str).str.contains(brand_filter, case=False, na=False)]
if model_filter.strip():
    filtered = filtered[filtered[model_col].astype(str).str.contains(model_filter, case=False, na=False)]
filtered = filtered[(filtered[price_col] >= min_price) & (filtered[price_col] <= max_price)]

show_cols = [year_col, brand_col, model_col, price_col, "ValueScore"] + [c for c in cat_cols if c in filtered.columns]
show_cols = [c for c in show_cols if c in filtered.columns]

st.dataframe(filtered[show_cols].sort_values(by=[year_col, brand_col, model_col]), use_container_width=True)

st.download_button(
    "⬇️ Download current view (CSV)",
    data=filtered[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="car_portfolio_view.csv",
    mime="text/csv",
)

# ------------------------- Tips -------------------------
with st.expander("ℹ️ Tips for Data Formatting", expanded=False):
    st.markdown("""
- Ensure your CSV has columns for **Year**, **Brand/Make**, **Model**, and **Price**.
- Price can include currency symbols (e.g., `$25,000`); the app will clean it.
- Add more numeric features (e.g., **Mileage**, **MPG**, **Horsepower**, **SafetyRating**) for richer **Value Score** calculation.
- You can customize **weights** and **directions** per feature to match your priorities.
""")
