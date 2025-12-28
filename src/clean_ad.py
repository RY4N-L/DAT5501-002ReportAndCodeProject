## -- Clean the Ad_table (extra).csv -- ##

# Import libraries
import pandas as pd

def preprocess_data():
    
    # Load csv
    df = pd.read_csv('data/raw/Ad_table (extra).csv', delimiter = ',')

    df = clean_df(df) # remove blanks and units
    df = transform_df(df) # add additional columns used for analysis

    # Final tidy by formatting column names
    rename_map = {
        "maker": "brand",
        "engin_size": "engine",
        "engine_power": "power",
        "runned_miles": "mileage",
        "color": "colour",
        "seat_num": "seats",
        "door_num": "doors"
    }

    df = format_column_names(df, rename_map)
    
    # Convert final processed df to csv
    df.to_csv("data/processed/ad.csv", index=False)

    return df


def clean_df(df: pd.DataFrame):
    '''
    Runs functions to clean dataset
    
    :param df: Description
    :type df: pd.DataFrame
    '''

    # Removing rows with blank values (viable as it's a large dataset)
    df = df.dropna()

    # Remove units
    df = remove_units(df, "Average_mpg", "mpg")
    df = remove_units(df, "Top_speed", "mph")
    df = remove_units(df, "Engin_size", "L")
    df = remove_units(df, "Runned_Miles", "mile")
    
    return df 


def transform_df(df: pd.DataFrame):
    '''
    Runs functions to extend dataset with vehicle age and usage intensity scores.   
    :param df: Description
    '''
    # Mark flagged values
    df = mark_flagged_rows(df, "Annual_Tax", "*")

    # Add column for vehicle age
    df = calculate_vehicle_age(df, "Adv_year", "Reg_year")

    # Calculate usage scores
    df = calculate_vehicle_usage(df, "age", "Runned_Miles")
    
    return df


def remove_units(df: pd.DataFrame, column_name: str, unit: str):
    '''
    Remove a unit suffix from a DataFrame column and convert the values to float.

    :param df: Pandas DataFrame containing the column to clean.
    :type df: pd.DataFrame
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

    print(f"Removed {unit} from values in {column_name}")
    
    return df


def mark_flagged_rows(df: pd.DataFrame, column_name: str, flag: str, new_column_name="Is_flagged"):
    '''
    Identify rows containing a given flag and mark them in a new Boolean column.

    :param df: The DataFrame containing the column to inspect.
    :type df: pd.DataFrame
    :param column_name: The name of the column to search for the flag.
    :type column_name: str
    :param flag: The substring used to identify flagged rows.
    :type flag: str
    :param new_column_name: Base name for the generated is flagged column.
                    Defaults to "Is_flagged".
    :type new_column_name: str

    :return: The DataFrame with a new Boolean column `is_flagged`.
    '''
    # Create a Boolean mask for rows that are flagged
    mask = df[column_name].astype(str).str.contains(flag, regex=False)

    # Store mask in new column
    df[new_column_name] = mask
    print (f"New column {new_column_name} created")
    
    # Remove rows which are flagged
    #flagged_df = df[~mask].rest_index(drop=True)

    #print(ad_unflagged_df)
    return df

def calculate_vehicle_age(df: pd.DataFrame, ad_column_name: str, man_date_column_name: str, new_column_name="age"):
    '''
    Calculates vehicle age using the date advertised and the date of registration and removes any nregative values from errors in data
    
    :param df: Pandas Dataframe containing the date columns.
    :type df: pd.DataFrame
    :param ad_column_name: Description
    :type ad_column_name: column with date vehicle was advertised
    :param man_date_column_name: column with date of vehicle registration
    :type man_date_column_name: str
    :param new_column_name: Base name for the generated vehicle age column.
                    Defaults to "Vehicle_age".
    :type new_column_name: str

    :return: The DataFrame with the added vehicle age column
    '''
    df[new_column_name] = df[ad_column_name].astype(int) - df[man_date_column_name].astype(float)
    df = remove_negatives(df, new_column_name)
    return df

def remove_negatives(df: pd.DataFrame, column_name: str):
    '''
    Remove rows where the specified column contains negative values.

    :param df: The DataFrame containing the column to filter.
    :type df: pd.DataFrame
    :param column_name: The name of the column to check for negative values.
    :type column_name: str

    :return: The DataFrame with negative values removed from the specified column.
    '''
    df = df[df[column_name] >= 0]
    print("removed negatives")
    
    return df


def calculate_vehicle_usage(df: pd.DataFrame, age_column:str , miles_column: str, new_column_name="Usage_intensity"):
    '''
    Calculate a normalised value for vehicle usage given the years and milage
    Uasage intensity = miles/age
    Lower = better

    :param df: The DataFrame containing vehicle age and mileage columns.
    :type df: pd.DataFrame
    :param age_column: The name of the column containing vehicle age in years.
    :type age_column: str
    :param miles_column: The name of the column containing mileage values.
    :type miles_column: str
    :param new_column_name: Base name for the generated usage-intensity columns.
                    Defaults to "Usage_intensity".
    :type new_column_name: str

    :return: The DataFrame with added usage-intensity, normalised, and inverted columns.
    '''

    age = df[age_column]

    # Replace age=0 with 0.5 years (half a year) to prevent division by 0
    safe_age = age.replace(0, 0.5)

    # raw score
    df[new_column_name] = (df[miles_column] / safe_age).round(5)

    # min-max normalisation (0 = best, 1 = worst)
    min_val = df[new_column_name].min()
    print(min_val)
    max_val = df[new_column_name].max()
    
    df[new_column_name + "_norm"] = ((df[new_column_name] - min_val) / (max_val - min_val)).round(5)
    df[new_column_name + "_norm_inv"] = 1 - df[new_column_name + "_norm"]

    # Check min and max is 0 and 1
    print(df[new_column_name + "_norm"].max())
    print(df[new_column_name + "_norm"].min())
    
    return df

def format_column_names(df: pd.DataFrame, rename_map: dict):
    
    df.columns = ( 
        df.columns
         .str.strip()
         .str.lower()
         .str.replace(" ", "_")
         .str.replace("-", "_") )

    df = df.rename(columns=rename_map)
    
    print("Renamed columns:")
    for old, new in rename_map.items():
        print(f"  {old} → {new}")


    return df


if __name__ == "__main__":
    preprocess_data()