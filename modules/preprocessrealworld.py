import pandas as pd

# Load the CSV file
file_path = '../newdata/output.csv'  # Update this if the file is elsewhere
output_file = '../newdata/preprocessed_output.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Clean and preprocess
df['Category'] = df['Category'].str.strip()  # Remove leading/trailing spaces
df['Data points'] = df['Ways to Test Assumptions'].str.strip()  # Clean data points

# Drop rows with missing values in either column
df = df.dropna(subset=['Category', 'Data points'])

# Split multi-line data points into individual rows
df = df.assign(Data_points_split=df['Data points'].str.split('\n')).explode('Data_points_split')

# Remove empty or whitespace-only rows from Data_points_split
df['Data_points_split'] = df['Data_points_split'].str.strip()
df = df[df['Data_points_split'] != '']

# Rename the column for clarity
df = df.rename(columns={'Data_points_split': 'Cleaned Data Points'})

# Save the preprocessed data to a new CSV
df.to_csv(output_file, index=False)

print(f"Preprocessed data saved to {output_file}")