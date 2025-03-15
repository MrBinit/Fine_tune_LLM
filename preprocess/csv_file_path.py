# this code will extract file path and there columns. 
import os
import pandas as pd
directory_path = '/home/binit/fine_tune_LLama/dataset'
output_csv_path = '/home/binit/fine_tune_LLama/csv_column_info.csv'

data = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.csv'):
            csv_file_path = os.path.join(root, file)
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            columns = df.columns.tolist()
            # Append the file path and columns to the data list
            data.append([csv_file_path, ', '.join(columns)])

df_info = pd.DataFrame(data, columns=['File Path', 'Column Names'])
df_info.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"Column names of all CSV files saved to {output_csv_path}")
