import re

# Define the file paths
input_file_path = '/home/binit/fine_tune_LLama/extracted_text.txt'
output_file_path = '/home/binit/fine_tune_LLama/split_nepali_text_output.txt'

# Function to clean text, split based on '।', and handle sentence length condition
def split_and_clean_text(input_file_path):
    cleaned_sentences = []
    temp_sentence = ''
    
    # Open the file and read line by line
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Clean the line by removing unwanted symbols (like ०, ००, etc.)
            line = line.strip()
            line = re.sub(r'[^अ-ह\s।]+', '', line)
            
            # Process each line for sentence splitting
            sentences = re.split(r'([।])', line)
            
            for part in sentences:
                # Skip empty parts or spaces
                if part.strip():
                    temp_sentence += part
                    # If '।' is found, it marks the end of the sentence
                    if '।' in part:
                        # Now, we need to check the sentence length
                        while len(temp_sentence) > 150:
                            # Split the sentence into chunks of 100-150 characters
                            chunk = temp_sentence[:150]
                            cleaned_sentences.append(chunk.strip())
                            temp_sentence = temp_sentence[150:]
                        if temp_sentence:
                            cleaned_sentences.append(temp_sentence.strip())
                        temp_sentence = ''

    return cleaned_sentences

# Get the cleaned and split sentences
cleaned_sentences = split_and_clean_text(input_file_path)

# Save the cleaned sentences to a new text file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for sentence in cleaned_sentences:
        output_file.write(sentence.strip() + '\n')  # Each sentence on a new line

print(f"Text has been split, cleaned, and saved to {output_file_path}")
