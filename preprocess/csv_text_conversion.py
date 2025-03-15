import pandas as pd

csv_file_path = '/home/binit/fine_tune_LLama/dataset/test_nepali_corpus_part1.csv'
txt_file_path = '/home/binit/fine_tune_LLama/extracted_text.txt'

df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

columns_to_select = ['Article']

missing_columns = [col for col in columns_to_select if col not in df.columns]

if missing_columns:
    print(f"Warning: These columns are not found in the CSV: {', '.join(missing_columns)}")
    print("Proceeding with the available columns...")

df_selected = df[columns_to_select] if all(col in df.columns for col in columns_to_select) else df

with open(txt_file_path, 'a', encoding='utf-8') as txt_file:
    for index, row in df_selected.iterrows():
        row_str = ' '.join(map(str, row.values))
        txt_file.write(row_str + '\n')

print(f"CSV file converted to text format and appended to {txt_file_path}")
