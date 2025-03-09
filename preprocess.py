from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count
spark = SparkSession.builder \
    .appName("Preprocessing") \
    .getOrCreate()


path = "/home/binit/fine_tune_LLama/dataset/extracted_text.csv"
# df = spark.read.csv(path)

df = spark.read.option("delimiter", ";").option("header", True).csv(path)
print(df.show())
print(df.columns)

# remove null values
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
# df.na.drop().show(truncate=False)
# df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
