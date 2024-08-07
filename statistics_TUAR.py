import pandas as pd 
import os

directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'

all_files = os.listdir(directory_path)
# Filter out only EDF files
csv_files = [file for file in all_files if file.endswith('.csv')]

for file_name in sorted(csv_files):
    # Try reading with a different delimiter if needed, e.g., ';' or '\t'
    df = pd.read_csv('your_file.csv', delimiter=',', header=None, error_bad_lines=False, quotechar='"')

    # Extract the total duration from the third line and first column
    total_duration = df.iloc[2, 0]

    # Initialize a variable to keep the sum
    total_difference_sum = 0

    # Iterate over the rows starting from the eighth line (index 7)
    for index in range(7, len(df)):
        # Ensure columns exist before performing operations
        if len(df.columns) > 2:
            difference = df.iloc[index, 2] - df.iloc[index, 1]
            total_difference_sum += difference

# Print the results
print(f"Total Duration: {total_duration}")
print(f"Sum of differences from the eighth line onward: {total_difference_sum}")