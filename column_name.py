from pyspark.sql import SparkSession
import os

def print_column_names(directory_path):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Print Column Names from CSVs") \
        .getOrCreate()

    # List all CSV files in the directory
    csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the directory.")
        spark.stop()
        return

    for csv_file in csv_files:
        try:
            # Read CSV file
            df = spark.read.csv(csv_file, header=True, inferSchema=True)
            # Print column names
            print(f"File: {csv_file}")
            print("Columns:", ", ".join(df.columns))
            print("-" * 50)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # Stop Spark session
    spark.stop()

# Example usage
dataset_path = "/home/binit/fine_tune_LLama/dataset"

print_column_names(dataset_path)
