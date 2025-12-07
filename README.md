# Car Price Portfolio (Streamlit)

An interactive portfolio of car prices organized by **Year** → **Brand**, with flexible categorization, charts, and a customizable **Value Score** that identifies the **Most** and **Least Worth Buying** vehicles per year.

## Features
- Upload CSV of vehicle listings (Year, Brand/Make, Model, Price, etc.).
- Explore prices by **Year** and **Category** (e.g., Brand, Body, Fuel).
- Interactive charts:
  - Average price by category per year (bar chart).
  - Price distribution per category per year (box plot).
  - Average price trend by brand (line chart).
- Custom **Value Score**:
  - Pick numeric features (Price, Mileage, MPG, Horsepower, SafetyRating…).
  - Set weights and directions (Minimize/Maximize).
  - Per-year normalization and scoring.
- Per-year **Most Worth Buying** and **Least Worth Buying** results.
- Filters, tables, and **CSV exports**.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
