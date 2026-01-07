import unittest as ut
import pandas as pd
from src.cleaning.clean_ad import *

class TestCleanAd (ut.TestCase):

    # -- Test final df and csv -- #

    def test_no_negative_values_in_processed_df(self):
        #Ensure the fully processed dataset contains no negative values.
        
        # Run the full cleaning pipeline
        df = preprocess_data()

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=["number"])

        # Assert all numeric values are >= 0
        self.assertTrue((numeric_df >= 0).all().all(), "Negative values found in numeric columns")

    def test_no_nans_in_processed_df(self):
        #Ensure the fully processed dataset contains no NaN values.
        
        # Run the full cleaning pipeline
        df = preprocess_data()

        # Assert no NaN values anywhere in the DataFrame
        self.assertFalse(df.isna().any().any(), "NaN values found in processed dataset")

    def test_no_negative_values_in_csv(self):
        # Ensure the exported CSV contains no negative numeric values to validate export step
        df = pd.read_csv("data/processed/ad.csv")

        numeric_df = df.select_dtypes(include=["number"])

        self.assertTrue((numeric_df >= 0).all().all(), "Negative values found in exported CSV")

    def test_no_nans_in_csv(self): 
        #Ensure the exported CSV contains no NaN values to validate export step
        df = pd.read_csv("data/processed/ad.csv") 
        
        # Assert no NaN values anywhere 
        self.assertFalse(df.isna().any().any(), "NaN values found in exported CSV")


    # -- Test remove_units() function -- #

    def test_remove_units_basic(self):
        # Ensure unit suffix is removed and values convert to float
        df = pd.DataFrame({"speed": ["120mph", "100mph"]})
        out = remove_units(df, "speed", "mph")
        self.assertEqual(out["speed"].tolist(), [120.0, 100.0])

    def test_remove_units_with_spaces(self):
        # Function should handle whitespace around values
        df = pd.DataFrame({"speed": ["120 mph", " 100mph "]})
        out = remove_units(df, "speed", "mph")
        self.assertEqual(out["speed"].tolist(), [120.0, 100.0])

    def test_remove_units_no_unit(self):
        # If no unit is present, values should still convert to float
        df = pd.DataFrame({"speed": ["120", "100"]})
        out = remove_units(df, "speed", "mph")
        self.assertEqual(out["speed"].tolist(), [120.0, 100.0])


    # -- Test flagged_rows() function -- #

    def test_flagged_rows(self):
        # Should create a boolean column marking rows containing the flag
        df = pd.DataFrame({"tax": ["150", "200*", "300"]})
        out = mark_flagged_rows(df, "tax", "*")
        
        # Check column exists
        self.assertIn("is_flagged", out.columns)

        # Only the row with '*' should be flagged
        self.assertEqual(out["is_flagged"].tolist(), [False, True, False])


    # -- Test calculate_vehicle_age() function -- #

    def test_vehicle_age_basic(self):
        # Age = adv_year - reg_year, negative ages should be removed
        df = pd.DataFrame({"adv_year": [2020, 2015], "reg_year": [2018, 2021]})
        out = calculate_vehicle_age(df, "adv_year", "reg_year")

        # Only the valid (non-negative) age should remain
        self.assertEqual(out["age"].tolist(), [2])


    
    # -- Test calculate_vehicle_usage() function -- #

    def test_usage_intensity_basic(self):
        # Should compute usage = miles / age in column usage_intensity
        df = pd.DataFrame({"age": [1, 2], "Runned_Miles": [10000, 50000]})
        out = calculate_vehicle_usage(df, "age", "Runned_Miles")
        
        # Check column exists
        self.assertIn("usage_intensity", out.columns)

        # Check calculation is correct
        self.assertAlmostEqual(out["usage_intensity"].iloc[0], 10000 / 1)
        self.assertAlmostEqual(out["usage_intensity"].iloc[1], 50000 / 2)

    def test_usage_intensity_zero_age(self):
        # age = 0 should be replaced with 0.5 to avoid division by zero
        df = pd.DataFrame({"age": [0], "mileage": [10000]})
        out = calculate_vehicle_usage(df, "age", "mileage")
        self.assertEqual(out["usage_intensity"].iloc[0], 10000 / 0.5)

    def test_usage_intensity_norm_range(self):
        # Normalised values should always fall between 0 and 1
        df = pd.DataFrame({"age": [1, 4], "mileage": [10000, 20000]})
        out = calculate_vehicle_usage(df, "age", "mileage")
        self.assertTrue(out["usage_intensity_norm"].between(0, 1).all())


    # -- Test format_column_names() function -- #

    def test_lowercase_and_underscores(self):
        # Should convert to lowercase and replace spaces with underscores
        df = pd.DataFrame({"Maker Name": [1]})
        out = format_column_names(df, {})
        self.assertIn("maker_name", out.columns)

    def test_renaming_map(self):
        # Should apply custom renaming map after formatting
        df = pd.DataFrame({"Maker": [1], "Seat_num": [5]})
        rename_map = {"maker": "brand", "seat_num": "seats"}
        out = format_column_names(df, rename_map)

        self.assertIn("brand", out.columns)
        self.assertIn("seats", out.columns)


if __name__ == '__main__':
    ut.main()

