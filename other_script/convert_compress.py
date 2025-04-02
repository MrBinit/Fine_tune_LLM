import zipfile

# Input file path
input_file = '/home/binit/fine_tune_LLama/split_nepali_text_output.txt'

# Output .zip file path
output_file = '/home/binit/fine_tune_LLama/split_nepali_text_output.zip'

# Compress the file into .zip
with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(input_file, arcname='split_nepali_text_output.txt')

print(f"File compressed and saved as {output_file}")
