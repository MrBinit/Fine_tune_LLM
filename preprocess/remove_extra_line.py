file_path = '/home/binit/fine_tune_LLama/nepali.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    lines = [line for line in file if line.strip()]
with open(file_path, 'w', encoding='utf-8') as file:
    file.writelines(lines)

print("Unnecessary line spaces removed successfully.")