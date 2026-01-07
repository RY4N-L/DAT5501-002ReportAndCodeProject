import unittest as ut
import pandas as pd
from src.cleaning.clean_price import clean_price

class TestCleanPrice(ut.TestCase):

    def test_clean_price_basic_structure(self):
        # Ensure cleaned price file contains required columns
        df = clean_price()
        self.assertIn("genmodel_id", df.columns)
        self.assertIn("year", df.columns)
        self.assertIn("entry_price", df.columns)

    def test_clean_price_no_nans(self):
        # Ensure no NaN values remain after cleaning
        df = clean_price()
        self.assertFalse(df.isna().any().any(), "NaN values found in cleaned price dataset")

    def test_clean_price_types(self):
        # Ensure year and entry_price are numeric
        df = clean_price()

        self.assertTrue(pd.api.types.is_numeric_dtype(df["year"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["entry_price"]))

    def test_clean_price_export(self):
        # Ensure CSV export works and contains no NaNs
        df = pd.read_csv("data/processed/price.csv")
        self.assertFalse(df.isna().any().any(), "NaN values found in exported price.csv")

    def test_unique_genmodel_year_pairs(self):
        # Ensure each (genmodel_id, year) pair appears only once after cleaning
        df = clean_price()
        duplicates = df.duplicated(subset=["genmodel_id", "year"]).any()
        self.assertFalse(duplicates, "Duplicate (genmodel_id, year) pairs found in cleaned price dataset")


if __name__ == "__main__":
    ut.main()
