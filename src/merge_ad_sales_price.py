import pandas as pd

def merge_ad_sales_price(
        ad_path="data/processed/ad.csv", 
        sales_path="data/processed/sales.csv", 
        price_path="data/processed/price.csv"
):
    # Merge cleaned advertisement, sales, and entry price datasets using genmodel_id and registration year (reg_year).

    ad_df = pd.read_csv(ad_path)
    sales_df = pd.read_csv(sales_path)
    
    price_df = pd.read_csv(price_path)
    print(sales_df.groupby(["genmodel_id", "year"]).size().sort_values(ascending=False).head(10))
    print(price_df.groupby(["genmodel_id", "year"]).size().sort_values(ascending=False).head(10))

    # Merge on genmodel_id + reg_year
    merged_df = ad_df.merge(
        sales_df,
        left_on=["genmodel_id", "reg_year"],
        right_on=["genmodel_id", "year"],
        how="left"
    ).drop(columns=["year"])

    merged_df = merged_df.merge(
        price_df,
        left_on=["genmodel_id", "reg_year"],
        right_on=["genmodel_id", "year"],
        how="left"
    ).drop(columns=["year"])

    # Fill missing values with 0
    merged_df["original_sales"] = merged_df["original_sales"].fillna(0).astype(int)
    merged_df["entry_price"] = merged_df["entry_price"].fillna(0).astype(float)

    merged_df.to_csv("data/processed/final_dataset.csv", index=False)
    return merged_df

if __name__ == "__main__":
    

    merge_ad_sales_price()

