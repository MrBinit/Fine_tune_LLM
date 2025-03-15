#convert csv to text format

import pandas as pd


csv_file_path = '/home/binit/fine_tune_LLama/dataset/extracted_text.csv'
df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
txt_file_path = '/home/binit/fine_tune_LLama/extracted_text.txt'

with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    for index, row in df.iterrows():
        row_str = ' '.join(map(str, row.values)) 
        txt_file.write(row_str + '\n')

print(f"CSV file converted to text format and saved at {txt_file_path}")
