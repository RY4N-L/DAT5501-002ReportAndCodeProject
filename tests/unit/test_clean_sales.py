import unittest as ut
import pandas as pd
from src.cleaning.clean_sales import clean_sales

class TestCleanSales(ut.TestCase):

    def test_clean_sales_basic_structure(self):
        # Ensure cleaned sales file contains required columns
        df = clean_sales()

        self.assertIn("genmodel_id", df.columns)
        self.assertIn("year", df.columns)
        self.assertIn("original_sales", df.columns)

    def test_clean_sales_no_nans(self):
        # Ensure no NaN values remain after cleaning
        df = clean_sales()
        self.assertFalse(df.isna().any().any(), "NaN values found in cleaned sales dataset")

    def test_clean_sales_types(self):
        # Ensure year is numeric and sales are integers
        df = clean_sales()

        self.assertTrue(pd.api.types.is_numeric_dtype(df["year"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["original_sales"]))

    def test_clean_sales_export(self):
        # Ensure CSV export works and contains no NaNs
        df = pd.read_csv("data/processed/sales.csv")
        self.assertFalse(df.isna().any().any(), "NaN values found in exported sales.csv")

    def test_unique_genmodel_year_pairs(self):
        # Ensure each (genmodel_id, year) pair appears only once after cleaning
        df = clean_sales()
        duplicates = df.duplicated(subset=["genmodel_id", "year"]).any()
        self.assertFalse(duplicates, "Duplicate (genmodel_id, year) pairs found in cleaned sales dataset")


if __name__ == "__main__":
    ut.main()
