import unittest as ut
import pandas as pd
from src.cleaning.merge_ad_sales_price import merge_ad_sales_price

class TestMergeAdSalesPrice(ut.TestCase):
    def test_no_duplicate_rows_after_merge(self):
        # Should not produce duplicate rows after merging
        merged_df, _ = merge_ad_sales_price()
        self.assertEqual(len(merged_df), len(merged_df.drop_duplicates()))

    def test_merge_row_count_matches_ad(self):
        # Merged dataset (before cleaning) should not have more rows than ad.csv
        merged_df, _ = merge_ad_sales_price()
        ad_df = pd.read_csv("data/processed/ad.csv")

        self.assertEqual(
            len(merged_df),
            len(ad_df),
            "Merged dataset has a different number of rows than ad.csv"
        )

    def test_final_dataset_contains_all_expected_columns(self):
        # Ensure the final merged dataset contains all original ad columns and the new sales and price columns

        _ ,  clean_df = merge_ad_sales_price()
        ad_df = pd.read_csv("data/processed/ad.csv")

        # Check all original ad columns must exist
        for col in ad_df.columns:
            self.assertIn(
                col,
                clean_df.columns,
                f"Column '{col}' from ad.csv is missing in final_dataset.csv"
            )

        # Check new columns exist
        self.assertIn("original_sales", clean_df.columns)
        self.assertIn("entry_price", clean_df.columns)

    def test_merge_no_nans_in_new_columns(self):
        # Ensure final columns contain no NaN values
        _, clean_df = merge_ad_sales_price()

        self.assertFalse(
            clean_df["original_sales"].isna().any(),
            "NaN values found in original_sales column"
        )
        self.assertFalse(
            clean_df["entry_price"].isna().any(),
            "NaN values found in entry_price column"
        )

    def test_merge_sales_and_price_are_numeric(self):
        # Ensure merged numeric columns are numeric types
        _, clean_df = merge_ad_sales_price()

        self.assertTrue(pd.api.types.is_numeric_dtype(clean_df["original_sales"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(clean_df["entry_price"]))

    def test_merge_export(self):
        # Ensure exported final_dataset.csv exists and contains no NaNs
        df = pd.read_csv("data/processed/final_dataset.csv")
        self.assertFalse(df.isna().any().any(), "NaN values found in final_dataset.csv")

    def test_cleaned_dataset_has_no_zeros(self):
        # Cleaned dataset should not contain zero sales or zero price
        _, clean_df = merge_ad_sales_price()
        self.assertFalse((clean_df["original_sales"] == 0).any())
        self.assertFalse((clean_df["entry_price"] == 0).any())

    def test_cleaned_dataset_has_no_nans(self):
        # Cleaned dataset should contain no NaN values
        _, clean_df = merge_ad_sales_price()
        self.assertFalse(clean_df.isna().any().any())

    def test_genmodel_id_integrity(self):
        # genmodel_id should never be missing after merge
        merged_df, _ = merge_ad_sales_price()
        self.assertFalse(merged_df["genmodel_id"].isna().any())


if __name__ == "__main__":
    ut.main()
