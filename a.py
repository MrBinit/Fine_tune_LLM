import os
from pyspark.sql import SparkSession

# Create Spark session with optimized memory configurations
spark = SparkSession.builder \
    .appName("LargeFileSplitter") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \   
    .config("spark.executor.cores", "4") \   
    .config("spark.driver.cores", "4") \     
    .config("spark.sql.shuffle.partitions", "500") \ 
    .config("spark.memory.fraction", "0.8") \ 
    .config("spark.dynamicAllocation.enabled", "true") \
    .getOrCreate()
# Path to the large file
file_path = '/home/binit/fine_tune_LLama/extracted_text.txt'
output_dir = '/home/binit/fine_tune_LLama/dataset'

# Define how large each split file should be in bytes (e.g., 20GB for each piece)
split_size = 20 * 1024 * 1024 * 1024  # 20 GB

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the large text file into an RDD (resilient distributed dataset)
rdd = spark.sparkContext.textFile(file_path)

# Function to split the RDD into chunks of desired size
def split_rdd(rdd, split_size):
    # Convert RDD to a list of strings (each string is a line)
    lines = rdd.collect()  # Be careful with collect on large datasets; consider working in partitions
    total_size = 0
    chunk = []
    part_number = 1
    current_chunk_size = 0

    for line in lines:
        line_size = len(line.encode('utf-8'))
        current_chunk_size += line_size
        
        if current_chunk_size >= split_size:
            # Write the current chunk to a file
            output_file_path = os.path.join(output_dir, f'piece_{part_number}.txt')
            with open(output_file_path, 'w') as output_file:
                output_file.write("\n".join(chunk))
            print(f'Written: {output_file_path}')
            part_number += 1
            chunk = [line]  # Start a new chunk with the current line
            current_chunk_size = line_size
        else:
            chunk.append(line)

    # Write the final chunk if any lines remain
    if chunk:
        output_file_path = os.path.join(output_dir, f'piece_{part_number}.txt')
        with open(output_file_path, 'w') as output_file:
            output_file.write("\n".join(chunk))
        print(f'Written: {output_file_path}')

# Split the RDD into chunks
split_rdd(rdd, split_size)

# Stop the Spark session
spark.stop()

print('File splitting complete.')
