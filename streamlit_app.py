
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from forecast import (
    prepare_yearly_category_data,
    train_and_forecast_per_category,
    evaluate_per_category,
)

st.set_page_config(page_title="Category Price Forecast", page_icon="📈", layout="wide")

st.title("📈 Price Change Prediction by Category")
st.caption("Upload a CSV, explore trends, and forecast yearly price changes per category.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Map your columns and control forecasting options below.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get started. Example columns: `date`, `category`, `price`.", icon="📄")
    st.stop()

# Load raw data
try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("🔍 Column Mapping")

# Auto-detect likely columns
default_date_col = next((c for c in raw.columns if "date" in c.lower() or "year" in c.lower()), None)
default_cat_col = next((c for c in raw.columns if "cat" in c.lower() or "segment" in c.lower()), None)
default_price_col = next((c for c in raw.columns if "price" in c.lower() or "amount" in c.lower() or "cost" in c.lower()), None)

col1, col2, col3 = st.columns(3)
with col1:
    date_col = st.selectbox("Date/Year Column", options=raw.columns.tolist(), index=raw.columns.get_loc(default_date_col) if default_date_col in raw.columns else 0)
with col2:
    category_col = st.selectbox("Category Column", options=raw.columns.tolist(), index=raw.columns.get_loc(default_cat_col) if default_cat_col in raw.columns else 0)
with col3:
    price_col = st.selectbox("Price Column", options=raw.columns.tolist(), index=raw.columns.get_loc(default_price_col) if default_price_col in raw.columns else 0)

agg_method = st.radio("Aggregate method per Year×Category", options=["mean", "median"], horizontal=True)
forecast_horizon = st.slider("Forecast horizon (years ahead)", min_value=1, max_value=10, value=3)
poly_degree = st.slider("Polynomial degree for trend", min_value=1, max_value=3, value=2)
test_years = st.slider("Evaluation: last N years for testing", min_value=1, max_value=5, value=2)

st.divider()

# Prepare yearly aggregated data
with st.spinner("Preparing data..."):
    df_yearly = prepare_yearly_category_data(
        df=raw,
        date_col=date_col,
        category_col=category_col,
        price_col=price_col,
        agg_method=agg_method,
    )

if df_yearly.empty:
    st.error("No valid yearly/category data found after parsing. Check column mappings and date formats.")
    st.stop()

st.subheader("🗂️ Yearly Aggregation Preview")
st.dataframe(df_yearly.head(20), use_container_width=True)

# Summary stats
st.subheader("📊 Summary by Category")
summary = (
    df_yearly.groupby(category_col)
    .agg(
        years=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        avg_price=(price_col, "mean"),
        median_price=(price_col, "median"),
    )
    .sort_values("avg_price", ascending=False)
)
st.dataframe(summary, use_container_width=True)

# Category selector
categories = sorted(df_yearly[category_col].dropna().unique().tolist())
selected_cats = st.multiselect("Select categories to visualize", options=categories, default=categories[: min(5, len(categories))])

# Plot trends
st.subheader("📈 Yearly Price Trends")
if selected_cats:
    fig = px.line(
        df_yearly[df_yearly[category_col].isin(selected_cats)],
        x="year", y=price_col, color=category_col,
        markers=True,
        title="Yearly aggregated price per category",
    )
    fig.update_layout(legend_title_text="Category", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# YoY change
st.subheader("📉 Year-over-Year (YoY) % Change")
df_yearly_sorted = df_yearly.sort_values(["{category}".format(category=category_col), "year"])
df_yearly_sorted["yoy_%"] = (
    df_yearly_sorted.groupby(category_col)[price_col].pct_change() * 100.0
)
fig_yoy = px.line(
    df_yearly_sorted[df_yearly_sorted[category_col].isin(selected_cats)],
    x="year", y="yoy_%", color=category_col, markers=True,
    title="YoY % Change in price",
)
fig_yoy.add_hline(y=0, line_dash="dot", line_color="gray")
st.plotly_chart(fig_yoy, use_container_width=True)

# Evaluate models
st.subheader("🧪 Model Evaluation (time-aware split)")
with st.spinner("Evaluating per-category models..."):
    eval_df = evaluate_per_category(
        df_yearly=df_yearly,
        category_col=category_col,
        price_col=price_col,
        test_years=test_years,
        poly_degree=poly_degree,
    )
st.dataframe(eval_df, use_container_width=True)
st.caption("Metrics are computed per category using the last N years as test set. MAE=Mean Absolute Error; MAPE=Mean Absolute Percentage Error.")

# Train & Forecast
st.subheader("🔮 Forecasts")
with st.spinner("Training models and forecasting..."):
    forecasts = train_and_forecast_per_category(
        df_yearly=df_yearly,
        category_col=category_col,
        price_col=price_col,
        horizon=forecast_horizon,
        poly_degree=poly_degree,
    )

if forecasts.empty:
    st.warning("No forecasts generated. Verify that categories have sufficient historical years.")
else:
    st.dataframe(forecasts, use_container_width=True)

    # Visualize forecast for selected categories
    st.subheader("🖼️ Forecast Charts")
    for cat in selected_cats:
        hist = df_yearly[df_yearly[category_col] == cat]
        fcat = forecasts[forecasts[category_col] == cat]
        if hist.empty and fcat.empty:
            continue

        fig_cat = go.Figure()
        # Historical
        fig_cat.add_trace(go.Scatter(
            x=hist["year"], y=hist[price_col], name="Historical",
            mode="lines+markers", line=dict(color="#1f77b4")
        ))
        # Forecast
        fig_cat.add_trace(go.Scatter(
            x=fcat["year"], y=fcat["pred"], name="Forecast",
            mode="lines+markers", line=dict(color="#ff7f0e")
        ))
        # Confidence band
        fig_cat.add_trace(go.Scatter(
            x=pd.concat([fcat["year"], fcat["year"][::-1]]),
            y=pd.concat([fcat["upper"], fcat["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,127,14,0.2)",
            line=dict(color="rgba(255,127,14,0)"),
            hoverinfo="skip",
            name="Confidence band"
        ))
        fig_cat.update_layout(
            title=f"Forecast — {cat}",
            xaxis_title="Year",
            yaxis_title="Aggregated Price",
            hovermode="x unified",
            showlegend=True,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Download button
    csv_buf = StringIO()
    forecasts.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download forecasts as CSV",
        data=csv_buf.getvalue(),
        file_name="category_price_forecasts.csv",
        mime="text/csv",
    )

st.success("Done! Adjust settings in the sidebar to refine results.")

