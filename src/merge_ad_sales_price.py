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

    clean_merged_df = clean_merged_dataset(merged_df)

    clean_merged_df.to_csv("data/processed/final_dataset.csv", index=False)
    
    return merged_df, clean_merged_df


def clean_merged_dataset(merged_df: pd.DataFrame):
    # Drop missing values
    merged_df = merged_df.dropna(subset=["original_sales", "entry_price"]).copy()

    # Convert new columns to numerical
    merged_df["original_sales"] = merged_df["original_sales"].astype(int)
    merged_df["entry_price"] = merged_df["entry_price"].astype(float)

    #print((merged_df["original_sales"] == 0).sum())
    #print((merged_df["entry_price"] == 0).sum())

    # Filter for rows with sales and entry_price > 0
    merged_df = merged_df[
    (merged_df["original_sales"] > 0) &
    (merged_df["entry_price"] > 0)].copy()

    return merged_df

if __name__ == "__main__":
    _, clean_df = merge_ad_sales_price()

    # Check final dataset summary statistics
    numeric_cols = clean_df.select_dtypes(include='number').columns.tolist()
    print (clean_df[numeric_cols].describe().T)

