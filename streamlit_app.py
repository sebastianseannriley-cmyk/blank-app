
# ------------------------- Model X & Model Y price panel (separate) -------------------------
st.subheader("🟢 Tesla Model X & Model Y — US Market Prices (separate series)")

# We removed Model Y from `df` earlier, so reload the raw file just for this panel.
try:
    raw_df = pd.read_csv(DATA_PATH)
    # Clean typing
    raw_df[COL_YEAR] = pd.to_numeric(raw_df[COL_YEAR], errors="coerce")
    raw_df[COL_PRICE] = pd.to_numeric(raw_df[COL_PRICE], errors="coerce")
    raw_df[COL_BRAND] = raw_df[COL_BRAND].astype(str).str.strip()
    raw_df[COL_MODEL] = raw_df[COL_MODEL].astype(str).str.strip()
except Exception:
    raw_df = None
    st.warning("Could not reload raw data; Model X/Y panel is unavailable.")

if raw_df is not None:
    # Filter Tesla X and Y; purge X < 2016
    raw_x = raw_df[(raw_df[COL_BRAND] == "Tesla") & (raw_df[COL_MODEL] == "Model X")].copy()
    raw_x = raw_x[raw_x[COL_YEAR] >= 2016]
    raw_y = raw_df[(raw_df[COL_BRAND] == "Tesla") & (raw_df[COL_MODEL] == "Model Y")].copy()

    # If modelx_prices.csv exists, override Model X prices by year
    if modelx_df is not None and not modelx_df.empty:
        x_override = raw_x.merge(modelx_df, how="left", left_on=COL_YEAR, right_on="Year")
        x_override.loc[~x_override["AbsolutePrice_USD"].isna(), COL_PRICE] = x_override["AbsolutePrice_USD"]
        # drop the extra Year column introduced by merge
        raw_x = x_override.drop(columns=["Year"])

    # Aggregate to per-year averages (switch .mean() to .median() if you prefer)
    mx_by_year = (
        raw_x.groupby(COL_YEAR, dropna=True)[COL_PRICE]
        .mean()
        .reset_index()
        .rename(columns={COL_PRICE: "Model X (USD)"})
    )
    my_by_year = (
        raw_y.groupby(COL_YEAR, dropna=True)[COL_PRICE]
        .mean()
        .reset_index()
        .rename(columns={COL_PRICE: "Model Y (USD)"})
    )

    # Build table with separate columns for X and Y
    xy_wide = mx_by_year.merge(my_by_year, on=COL_YEAR, how="outer")

    # Add Model X inflation-adjusted (if available in modelx_prices.csv)
    if modelx_df is not None and "InflationAdjusted_USD" in modelx_df.columns:
        infl = modelx_df[["Year", "InflationAdjusted_USD"]].rename(columns={"Year": COL_YEAR})
        xy_wide = xy_wide.merge(infl, on=COL_YEAR, how="left")
        xy_wide = xy_wide.rename(columns={"InflationAdjusted_USD": "Model X (inflation‑adj USD)"})
    else:
        xy_wide["Model X (inflation‑adj USD)"] = np.nan

    # Sort and clean
    xy_wide = xy_wide.dropna(subset=[COL_YEAR]).sort_values(COL_YEAR)

    if xy_wide.empty:
        st.info("No Model X or Model Y data available for this panel.")
    else:
        # ---- Chart: two lines (Model X vs Model Y) ----
        m = xy_wide.melt(
            id_vars=[COL_YEAR],
            value_vars=["Model X (USD)", "Model Y (USD)"],
            var_name="Series",
            value_name="PriceUSD"
        )

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

        # ---- Table: separate columns for X and Y ----
        cta, ctb = st.columns(2)
        with cta:
            st.markdown("**Model X & Model Y price table (separate columns)**")
            st.dataframe(xy_wide.rename(columns={COL_YEAR: "Year"}), use_container_width=True)
        with ctb:
            st.download_button(
                "⬇️ Download X & Y series (CSV)",
                xy_wide.rename(columns={COL_YEAR: "Year"}).to_csv(index=False).encode("utf-8"),
                file_name="tesla_model_x_and_model_y_series.csv"
            )
