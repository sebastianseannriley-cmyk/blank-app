
# 📈 Category Price Forecast (Streamlit)

A Streamlit app that ingests a CSV and predicts yearly price changes per category using polynomial regression (Ridge).

## Features
- Upload CSV; map date/year, category, and price columns
- Automatic year parsing from dates or strings
- Year×Category aggregation (mean/median)
- YoY % change visualization
- Per-category modeling & forecasts with confidence bands
- Download forecast CSV

## Quickstart
```bash
# Clone and enter the repo
git clone https://github.com/your-username/price-forecast-app.git
cd price-forecast-app

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
