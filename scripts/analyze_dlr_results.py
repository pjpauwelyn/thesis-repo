import pandas as pd

# Load the DLR results CSV file
dlr_file = 'dlr_data/dlr-results.csv'
df = pd.read_csv(dlr_file)

# Display basic information
print("DLR Results Analysis:")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# Display the first few rows to understand the structure
print("\nFirst few rows:")
print(df.head(3))

# Check the distribution of idx values
print("\nDistribution of idx values:")
print(df['idx'].value_counts())
