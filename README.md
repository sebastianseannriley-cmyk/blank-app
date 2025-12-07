
# Car Price Portfolio — Fixed to `data.csv`

This Streamlit app is fully tailored to the provided dataset and does **not** allow external uploads.  
It organizes and explores car prices by **Year → Brand → Model/Trim**, with flexible categorization (Fuel, Transmission, Condition) and a per‑year **ValueScore** to identify the **Most** and **Least Worth Buying** vehicles.

## Dataset (columns)
`Car ID, Brand, Year, Engine Size, Fuel Type, Transmission, Mileage, Condition, Price, Model`  *(as found in `data.csv`)*. [1](https://usarmywestpoint-my.sharepoint.com/personal/sebastianseann_riley_westpoint_edu/_layouts/15/Doc.aspx?sourcedoc=%7B8EE93C14-B318-40B8-BBFC-158C221134B4%7D&file=data.csv&action=default&mobileredirect=true)

## Features
- Filter by **Year**, **Brand**, **Model/Trim**, and **Price range**.
- Choose primary/secondary grouping (Brand, Model/Trim, Fuel, Transmission, Condition).
- Charts:
  - Average price by group per year (bar).
  - Price distribution per group per year (box plot).
  - Brand average price trend over years (line).
- Custom **ValueScore** (per‑year normalization):
  - **Price** (Minimize), **Mileage** (Minimize), **Engine Size** (Minimize/Maximize configurable).
  - Tune weights from the sidebar.
- Per‑year **Best**/**Worst** vehicles + CSV exports.
- Full listings table with quick text filters and CSV export.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
