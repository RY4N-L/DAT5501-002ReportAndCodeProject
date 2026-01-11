import pandas as pd

def clean_price(path="data/raw/Price_table.csv"):
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = df.columns.str.lower().str.strip()

    # Filter for relevant columns needed to merge
    df = df[["genmodel_id", "year", "entry_price"]]

    # Convert year and price to float
    df["year"] = df["year"].astype(float)
    df["entry_price"] = df["entry_price"].fillna(0).astype(float)

    df.to_csv("data/processed/price.csv", index=False)
    return df

if __name__ == "__main__":
    clean_price()