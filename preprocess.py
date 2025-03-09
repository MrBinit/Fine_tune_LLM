from pyspark.sql import SparkSession
import os
import unicodedata
import re 
import emoji
import pandas as pd 

eng_to_nepali_numbers = str.maketrans("0123456789", "०१२३४५६७८९")

def preprocess(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\.pdf", "", text, flags=re.IGNORECASE)
    text = emoji.replace_emoji(text, replace='')
    text = text.translate(eng_to_nepali_numbers)
    text = re.sub(r"[^०१२३४५६७८९अ-ह‍ि-्ॐक़-ॡ০-৯a-zA-Z\s\.\,\?\!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_columns_to_txt(csv_path, columns_to_extract, output_txt_path):
    spark = SparkSession.builder \
        .appName("Preprocessing") \
        .getOrCreate()

    df = spark.read.csv(csv_path, header= True, inferSchema = True)

    missing_cols = [col for col in columns_to_extract if col not in df.columns]

    if missing_cols:
        print(f"Warning: These columns were not found in the CSV: {missing_cols}")
        return

    selected_df = df.select(columns_to_extract)
    text_rdd = selected_df.rdd.map(lambda row: preprocess(" ".join(str(item) for item in row)))
    extracted_text = text_rdd.collect()
    with open(output_txt_path, "a", encoding="utf-8") as txt_file:
        for line in extracted_text:
            txt_file.write(line + "\n")

    print(f"Extracted text saved to {output_txt_path}")
    spark.stop()


csv_path = "/home/binit/fine_tune_LLama/dataset/extracted_text.csv"
columns_to_extract = ["Nepali Text"]
output_txt_path = "/home/binit/fine_tune_LLama/output_text.txt"

extract_columns_to_txt(csv_path, columns_to_extract, output_txt_path)