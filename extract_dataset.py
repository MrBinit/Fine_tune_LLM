#extraction 10% of the entire dataset. 
file_path = "/home/binit/fine_tune_LLama/split_nepali_text_output.txt"
output_path = "/home/binit/fine_tune_LLama/extracted_text_sample_test.txt"

#extract 10% of the total dataset
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
sample_size = max(1, int(len(lines) * 0.0002))
sampled_lines = lines[:sample_size]

with open(output_path, 'w', encoding='utf-8') as output_file:
    output_file.writelines(sampled_lines)

print(f"Saved first {sample_size} lines to: {output_path}")
