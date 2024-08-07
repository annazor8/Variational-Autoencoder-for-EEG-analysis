import pandas as pd 
import os

directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar'

all_files = os.listdir(directory_path)
# Filter out only EDF files
csv_files = [file for file in all_files if file.endswith('.csv')]
total_duration=0
total_artifacts_duration=0
for file_name in sorted(csv_files):
    # Try reading with a different delimiter if needed, e.g., ';' or '\t'
    file_path = os.path.join(directory_path, file_name)
    df = pd.read_csv(file_path, nrows=5)
    eeg_duration = float(df.iloc[1,0].split("=")[1].strip().split(" ")[0])
    # Extract the total duration from the third line and first column
    total_duration=total_duration+eeg_duration
    df_artifact=pd.read_csv(file_path, skiprows=7)
    df_artifact['Difference'] = df_artifact.iloc[:, 2] - df_artifact.iloc[:, 1]
    # Initialize a variable to keep the sum
    total_artifacts_eeg = df_artifact['Difference'].sum()
    total_artifacts_duration=total_artifacts_duration + total_artifacts_eeg
    # Iterate over the rows starting from the eighth line (index 7)
    
perc_artifacts=total_artifacts_duration/(total_duration*22)*100
perc_clean_dataset=100-perc_artifacts
# Print the results
print(f"Percentage artifacts in the dataset: {perc_artifacts}")
print(f"Percentage clean eeg in the dataset: {perc_clean_dataset}")