import pandas as pd
import re

# Load csv
trim_df = pd.read_csv('data/raw/Trim_table.csv', delimiter = ',')

# Filter rows conatining hp or ps (power)
filtered_df = trim_df[trim_df["Trim"].str.contains("hp|ps", case=False, na=False)]

# Create a new column for power and unit
extended_df = filtered_df
extended_df[["Power", "Unit"]] = filtered_df["Trim"].str.extract(r'(\d+)\s*(hp|ps|bhp)', flags=re.IGNORECASE)

# Convert power to float
extended_df["Power"] = extended_df["Power"].astype(float)

# Check new extended dataframe 
#print (extended_df)

# Check for ps and hp units
count_ps = (extended_df['Unit'].str.lower() == 'ps').sum()
print(count_ps)

count_hp = (extended_df['Unit'].str.lower() == 'hp').sum()
print(count_hp)

count_bhp = (extended_df['Unit'].str.lower() == 'bhp').sum()
print(count_bhp)

# Convert ps and hp to bhp
mask = extended_df['Unit'].str.lower() == 'ps'
extended_df.loc[mask, 'Power'] = extended_df.loc[mask, 'Power'] * 0.98632 
extended_df.loc[mask, 'Unit'] = 'bhp'
extended_df.loc[extended_df['Unit'].str.lower() == 'hp', 'Unit'] = 'bhp'

# Check new filtered dataframe has only bhp values
new_count_bhp = ((extended_df['Unit'].str.lower() == 'bhp').sum())

if (new_count_bhp == count_hp + count_ps + count_bhp):
    print("All power units converted to bhp")
else:
    print ("Not all power units converted to bhp")

# Remove any rows with blank values (may be caused from trim name containing hp or ps in other words)
final_extended_df = extended_df.dropna()

# Convert final df to csv
final_extended_df.to_csv("data/processed/trim.csv", index=False)


