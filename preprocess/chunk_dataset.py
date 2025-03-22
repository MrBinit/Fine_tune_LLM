import os

def split_file_by_size(input_file, chunk_size_gb=10):
    """
    Splits a large text file into smaller chunks of specified size.

    Args:
        input_file (str): Path to the input large text file.
        chunk_size_gb (int): Size of each chunk in gigabytes. Default is 10GB.
    """
    chunk_size_bytes = chunk_size_gb * 1024 * 1024 * 1024 
    output_dir = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        chunk_number = 1
        chunk_file = os.path.join(output_dir, f"{base_name}_chunk_{chunk_number}.txt")
        outfile = open(chunk_file, 'w', encoding='utf-8')
        current_chunk_size = 0

        for line in infile:
            outfile.write(line)
            current_chunk_size += len(line.encode('utf-8'))  # Calculate size in bytes

            if current_chunk_size >= chunk_size_bytes:
                outfile.close()
                print(f"Created chunk: {chunk_file} ({current_chunk_size / (1024 * 1024 * 1024):.2f} GB)")
                chunk_number += 1
                chunk_file = os.path.join(output_dir, f"{base_name}_chunk_{chunk_number}.txt")
                outfile = open(chunk_file, 'w', encoding='utf-8')
                current_chunk_size = 0

        # Close the last chunk file
        outfile.close()
        print(f"Created chunk: {chunk_file} ({current_chunk_size / (1024 * 1024 * 1024):.2f} GB)")

input_file_path = '/home/binit/fine_tune_LLama/extracted_text.txt'
split_file_by_size(input_file_path, chunk_size_gb=10)
