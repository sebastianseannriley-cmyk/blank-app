
# 📈 Yearly Price Forecast by Category (Streamlit)

A Streamlit app that ingests a CSV and predicts yearly price changes **per category** using Polynomial Regression (Ridge).  
Works out of the box with datasets like cars (columns: `Brand`, `Model`, `Year`, `Price`, etc).

## Features
- Upload CSV; map `Year`, `Price`, and choose one or **multiple** category columns
- Robust price aggregation (mean/median) and optional outlier capping (IQR)
- YoY % change visualization
- Per-category trend modeling & multi-year forecasts with confidence bands
- Download forecast CSV

## Quickstart
```bash
git clone https://github.com/your-username/price-forecast-app.git
cd price-forecast-app
pip install -r requirements.txt
streamlit run app.py
