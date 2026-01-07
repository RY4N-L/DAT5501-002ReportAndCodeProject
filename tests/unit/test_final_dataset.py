import unittest as ut
import pandas as pd
from src.merge_ad_sales_price import merge_ad_sales_price

class TestMergeAdSalesPrice(ut.TestCase):

    def test_merge_row_count_matches_ad(self):
        # Final dataset should never have more rows than ad.csv
        merged_df = merge_ad_sales_price()
        ad_df = pd.read_csv("data/processed/ad.csv")

        self.assertEqual(
            len(merged_df),
            len(ad_df),
            "Merged dataset has a different number of rows than ad.csv"
        )

    def test_final_dataset_contains_all_expected_columns(self):
        # Ensure the final merged dataset contains all original ad columns and the new sales and price columns

        merged_df = merge_ad_sales_price()
        ad_df = pd.read_csv("data/processed/ad.csv")

        # Check all original ad columns must exist
        for col in ad_df.columns:
            self.assertIn(
                col,
                merged_df.columns,
                f"Column '{col}' from ad.csv is missing in final_dataset.csv"
            )

        # Check new columns exist
        self.assertIn("original_sales", merged_df.columns)
        self.assertIn("entry_price", merged_df.columns)


    def test_merge_no_nans_in_new_columns(self):
        # Ensure merged columns contain no NaN values
        merged_df = merge_ad_sales_price()

        self.assertFalse(
            merged_df["original_sales"].isna().any(),
            "NaN values found in original_sales column"
        )
        self.assertFalse(
            merged_df["entry_price"].isna().any(),
            "NaN values found in entry_price column"
        )

    def test_merge_sales_and_price_are_numeric(self):
        # Ensure merged numeric columns are numeric types
        merged_df = merge_ad_sales_price()

        self.assertTrue(pd.api.types.is_numeric_dtype(merged_df["original_sales"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(merged_df["entry_price"]))

    def test_merge_export(self):
        # Ensure exported final_dataset.csv exists and contains no NaNs
        df = pd.read_csv("data/processed/final_dataset.csv")
        self.assertFalse(df.isna().any().any(), "NaN values found in final_dataset.csv")

if __name__ == "__main__":
    ut.main()
