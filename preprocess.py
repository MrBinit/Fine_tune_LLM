from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count
import os


def list_csv_columns(folder_paths):
    spark = SparkSession.builder \
        .appName("Preprocessing") \
        .getOrCreate()

    csv_files = [f for f in os.listdir(folder_paths) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the folder.")
        return
        
    for csv_file in csv_files:
        csv_path = os.path.join(folder_paths, csv_file)

        try:
            df = spark.read.csv(csv_path, header = True, inferSchema=True)
            print(f"Columns in {csv_file}: {df.columns}")

        except Exception as e:
            print(f"error reading {csv_file}: {e}")
    spark.stop()

list_csv_columns('/home/binit/fine_tune_LLama/dataset')
