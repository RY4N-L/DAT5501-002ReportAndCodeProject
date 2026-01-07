import pandas as pd

def clean_sales(path="data/raw/Sales_table.csv"):
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = df.columns.str.lower().str.strip()

    # Melt wide year columns into long format
    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(
        id_vars=["genmodel_id"],
        value_vars=year_cols,
        var_name="year",
        value_name="original_sales"
    )

    # Convert year to float and sales to int
    df_long["year"] = df_long["year"].astype(float)
    df_long["original_sales"] = df_long["original_sales"].fillna(0).astype(int)

    df_long.to_csv("data/processed/sales.csv", index=False)
    return df_long

if __name__ == "__main__":
    clean_sales()