# import pandas as pd
# file_path = '/home/binit/fine_tune_LLama/1/ne.txt'

# # Read the content of the text file
# with open(file_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# # Strip any leading/trailing whitespace and newline characters
# lines = [line.strip() for line in lines]

# # Create a DataFrame with the column name 'Nepali Text'
# df = pd.DataFrame(lines, columns=['Nepali Text'])

# # Define the path to save the CSV file
# csv_file_path = 'ne.csv'

# # Save the DataFrame to a CSV file
# df.to_csv(csv_file_path, index=False, encoding='utf-8')

# # Print the path to the saved CSV file
# print(f'CSV file saved at: {csv_file_path}')

import kagglehub

# Download latest version
path = kagglehub.dataset_download("reganmaharjan/nepali-corpus-and-tokenizer")

print("Path to dataset files:", path)