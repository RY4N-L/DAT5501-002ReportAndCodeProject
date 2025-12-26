## -- Clean the Ad_table (extra).csv -- ##

# Import libraries
import pandas as pd
import re
import numpy as np

def clean_df():
    '''
    Runs functions to clean and extend datset with vehicle age and usage intensity scores.

    Parameters:
    None

    Returns:
    None
    
    '''

    # Load csv
    df = pd.read_csv('data/raw/Ad_table (extra).csv', delimiter = ',')

    # Remove rows with blank values (all columns will be needed as features)
    df = df.dropna()

    # Remove units
    df = remove_units(df, "Average_mpg", "mpg")
    df = remove_units(df, "Top_speed", "mph")
    df = remove_units(df, "Engin_size", "L")
    df = remove_units(df, "Runned_Miles", "mile")

    # Mark flagged values
    df = mark_flagged_rows(df, "Annual_Tax", "*")

    # Add column for vehicle age
    df = calculate_vehicle_age(df, "Adv_year", "Reg_year")

    # Calculate usage scores
    df = calculate_vehicle_usage(df, "Vehicle_age", "Runned_Miles")

    # Convert final df to csv
    df.to_csv("data/processed/ad.csv", index=False)

    

def remove_units(df, column_name: str, unit: str):
    '''
    Remove a unit suffix from a DataFrame column and convert the values to float.

    :param df: Pandas DataFrame containing the column to clean.
    :param column_name: The name of the column whose values include the unit suffix.
    :type column_name: str
    :param unit: The unit string to detect and remove (e.g., "mph", "mpg", "L").
    :type unit: str

    :return: The DataFrame with the specified column cleaned and converted to float.
    
    '''
    # Check all top speed values are measured in unit
    bool_list = (df[column_name].astype(str).str.contains(unit))
    
    if(bool_list.all()):
        print(f"All values in {column_name} are measured in {unit}")
    else:
        print(f"Not all values in {column_name} are measured in {unit}")

    # Strip unit from column name and convert to float
    df[column_name] = (
        df[column_name]
        .astype(str)
        .str.replace(unit, '', regex=False)
        .str.strip()
        .astype(float))

    print(f"Removed {unit} from all values in {column_name}")
    
    return df


def mark_flagged_rows(df, column_name: str, flag: str):
    '''
    Identify rows containing a given flag and mark them in a new Boolean column.

    :param df: The DataFrame containing the column to inspect.
    :param column_name: The name of the column to search for the flag.
    :type column_name: str
    :param flag: The substring used to identify flagged rows.
    :type flag: str

    :return: The DataFrame with a new Boolean column `is_flagged`.
    '''
    # Create a Boolean mask for rows that are flagged
    mask = df[column_name].astype(str).str.contains(flag, regex=False)

    # Store mask in new column
    df["is_flagged"] = mask
    print ("New column is_flagged created")
    
    # Remove rows which are flagged
    #flagged_df = df[~mask].rest_index(drop=True)

    #print(ad_unflagged_df)
    return df

def calculate_vehicle_age(df, ad_column_name: str, man_date_column_name: str):
    '''
    Calculates vehicle age using the date advertisde and the date or registration    
    
    :param df: Pandas Dataframe containing the date columns.
    :param ad_column_name: Description
    :type ad_column_name: column with date vehicle advertised
    :param man_date_column_name: column with date of vehicle registration
    :type man_date_column_name: str

    :return: The DataFrame with the added column called "Vehicle_age".
    '''
    df['Vehicle_age'] = df[ad_column_name].astype(int) - df[man_date_column_name].astype(float)
    df = remove_negatives(df, "Vehicle_age")
    return df

def remove_negatives(df, column_name: str):
    '''
    Remove rows where the specified column contains negative values.
    
    :param df: The DataFrame containing vehicle age and mileage columns.
    :param age_column: The name of the column containing vehicle age in years.
    :type age_column: str
    :param miles_column: The name of the column containing mileage values.
    :type miles_column: str
    :param new_col: Base name for the generated usage‑intensity columns.
                    Defaults to "Usage_intensity".
    :type new_col: str

    :return: The DataFrame with negative values removed from the specified column.
    '''
    df = df[df[column_name] >= 0]
    print("removed negatives")
    return df


def calculate_vehicle_usage(df, age_column:str , miles_column: str, new_col="Usage_intensity"):
    '''
    Calculate a normalised value for vehicle usage given the years and milage
    Uasage intensity = miles/age
    Lower = better

    :param df: Description
    :param ad_date_column: Description
    :type ad_date_column: str
    :param miles_column: Description
    :type miles_column: str
    '''
    print(df[df["Vehicle_age"] < 0][["Vehicle_age", "Adv_year", "Reg_year"]])

    age = df[age_column]

    # Replace age=0 with 0.5 years (half a year) to prevent division by 0
    safe_age = age.replace(0, 0.5)

    # raw score
    df[new_col] = (df[miles_column] / safe_age).round(5)

    # min-max normalisation (0 = best, 1 = worst)
    min_val = df[new_col].min()
    print(min_val)
    max_val = df[new_col].max()
    
    df[new_col + "_norm"] = ((df[new_col] - min_val) / (max_val - min_val)).round(5)
    df[new_col + "_norm_inv"] = 1 - df[new_col + "_norm"]

    # Check min and max is 0 and 1
    print(df[new_col + "_norm"].max())
    print(df[new_col + "_norm"].min())
    
    return df


if __name__ == "__main__":
    clean_df()