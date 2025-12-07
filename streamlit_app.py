
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from io import StringIO

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Yearly Price Forecast by Category", page_icon="📈", layout="wide")
st.title("📈 Yearly Price Prediction by Category")
st.caption("Upload a CSV (like your car dataset), pick one or more category columns, and forecast price trends per category.")

# -----------------------------
# File upload
# -----------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload your `data.csv` to begin. Minimum columns needed: `Year` (int) and `Price` (numeric). Optional category columns like `Brand`, `Model`, `Fuel Type`.", icon="📄")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# -----------------------------
# Column mapping UI
# -----------------------------
st.subheader("🔍 Column Mapping & Options")

# Auto-detect helpful defaults
all_cols = df.columns.tolist()
default_year = "Year" if "Year" in df.columns else next((c for c in all_cols if "year" in c.lower()), all_cols[0])
default_price = "Price" if "Price" in df.columns else next((c for c in all_cols if "price" in c.lower()), all_cols[-1])

year_col = st.selectbox("Year column", options=all_cols, index=df.columns.get_loc(default_year) if default_year in df.columns else 0)
price_col = st.selectbox("Price column", options=all_cols, index=df.columns.get_loc(default_price) if default_price in df.columns else 0)

# Candidate categorical columns: objects + low-cardinality numerics
cat_candidates = [
    c for c in all_cols
    if (
        df[c].dtype == "object" or
        (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= max(20, int(0.05 * len(df))))
    )
    and c not in {year_col, price_col}
]
if len(cat_candidates) == 0:
    st.warning("No obvious categorical columns found. You can still proceed by selecting any columns manually.")
cat_cols = st.multiselect(
    "Category columns (pick one or more to define the grouping)",
    options=cat_candidates if cat_candidates else all_cols,
    default=[c for c in ["Brand", "Model", "Fuel Type"] if c in df.columns] or (cat_candidates[:1] if cat_candidates else [])
)

agg_method = st.radio("Aggregate method for Year×Category price", options=["mean", "median"], horizontal=True)
forecast_horizon = st.slider("Forecast horizon (years ahead)", min_value=1, max_value=10, value=3)
poly_degree = st.slider("Polynomial degree for trend", min_value=1, max_value=3, value=2)
test_years = st.slider("Evaluation: last N years for testing", min_value=1, max_value=5, value=2)
min_years_per_cat = st.slider("Minimum historical years per category to model", 2, 10, 3)
trim_outliers = st.checkbox("Trim outliers in price (IQR capping)", value=True)

st.divider()

# -----------------------------
# Preprocessing helpers
# -----------------------------
def create_category_key(df: pd.DataFrame, cat_cols: list) -> pd.Series:
    """
    Combine selected categorical columns into a single category key.
    """
    if not cat_cols:
        return pd.Series(["All"] * len(df), index=df.index)
