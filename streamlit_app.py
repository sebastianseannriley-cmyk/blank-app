
# app.py
# Streamlit app that:
# - loads electricnotpetrol.csv (or clean_electricnotpetrol.csv if present),
# - purges Tesla Model X rows with Year < 2016,
# - forces all Tesla rows to Fuel Type = Electric,
# - makes ValueScore EV-aware,
# - uses Altair blocks formatted to avoid parenthesis/quote mismatches.

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------- Page -------------------------
st.set_page_config(
    page_title="Car Price Portfolio (EV-aware)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Columns -------------------------
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
# For Streamlit <=1.17, replace @st.cache_data with @st.cache
@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_and_clean(primary_path: str, clean_path: str = None):
