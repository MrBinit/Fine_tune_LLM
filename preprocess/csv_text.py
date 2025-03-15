# this code will extract the file path and the column names and check if the column name mentioned in the data files are present or not. if its present it will convert the column into .txt format. 
import os
import pandas as pd

output_csv_path = '/home/binit/fine_tune_LLama/csv_column_info.csv'
txt_file_path = '/home/binit/fine_tune_LLama/extracted_text.txt'
columns_to_select = ['question', 'answer', 'review', 'Article', 'Translated_Response', 'Translated_Context', 'Sentences', 'correct', 'incorrect' , 'article', 'ne', 'text', 'input', 'output', 'Nepali Text', 'article_summary', 'Question', 'Answer', 'target']

if os.path.exists(output_csv_path):
    column_info_df = pd.read_csv(output_csv_path, encoding='utf-8-sig')
else:
    print(f"Error: {output_csv_path} not found.")
    exit()

for _, row in column_info_df.iterrows():
    csv_file_path = row['File Path']
    column_names = row['Column Names'].split(', ') 
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    missing_columns = [col for col in columns_to_select if col not in df.columns]

    if missing_columns:
        print(f"Warning: These columns are not found in {csv_file_path}: {', '.join(missing_columns)}")
        print("Proceeding with the available columns...")

    df_selected = df[columns_to_select] if all(col in df.columns for col in columns_to_select) else df
    with open(txt_file_path, 'a', encoding='utf-8') as txt_file:
        for index, row in df_selected.iterrows():
            row_str = ' '.join(map(str, row.values))
            txt_file.write(row_str + '\n')

    print(f"CSV file {csv_file_path} converted to text format and appended to {txt_file_path}")
