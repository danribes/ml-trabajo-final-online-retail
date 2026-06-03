import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
XLSX = HERE / "online_retail_II.xlsx"
PKL = HERE / "online_retail.pkl"

print("Loading Excel file (this takes ~30s)...")
xl = pd.ExcelFile(XLSX)
df = pd.concat([pd.read_excel(XLSX, sheet_name=s) for s in xl.sheet_names], ignore_index=True)
df.columns = [c.strip() for c in df.columns]
df.rename(columns={"Customer ID": "CustomerID"}, inplace=True)

print("Saving compatible pickle file...")
df.to_pickle(PKL)
print("Pickle updated successfully!")
