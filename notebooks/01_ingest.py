# Databricks notebook source
!pip install gdown

# COMMAND ----------

import os
import gdown
import zipfile

local_dir = "/dbfs/Users/rafael.pierre@databricks.com/bitcoin_tweets"
output = f"{local_dir}/bitcoin_tweets.zip"
url = "https://drive.google.com/uc?id=1kk63Nn9ROPjHn9I1JzQU8Ckd2WEx3R2i"

if not os.path.exists(local_dir):
  os.makedirs(local_dir)

if not os.path.exists(output):
  gdown.download(url, output, quiet=False)
  with zipfile.ZipFile(output, 'r') as zip_ref:
      zip_ref.extractall(local_dir)
      
!ls {local_dir}

# COMMAND ----------

!cat {local_dir}/Bitcoin_tweets.csv | tr "," "|" > {local_dir}/tweets.csv

# COMMAND ----------

!head {local_dir}/tweets.csv

# COMMAND ----------

from pyspark.sql.types import (
  StructType,
  StructField,
  StringType,
  IntegerType,
  TimestampType,
  BooleanType,
  ArrayType
)

schema = StructType([ \
    StructField("user_name", StringType(),True), \
    StructField("user_location", StringType(),True), \
    StructField("user_created", TimestampType(),True), \
    StructField("user_followers", IntegerType(), True), \
    StructField("user_friends", IntegerType(), True), \
    StructField("user_favorites", IntegerType(), True), \
    StructField("user_verified", BooleanType(), True), \
    StructField("date", TimestampType(), True), \
    StructField("text", StringType(), True), \
    StructField("hashtags", StringType(), True), \
    StructField("source", StringType(), True), \
    StructField("is_retweet", BooleanType(), True), \
  ])

spark_dir = f"{local_dir}".replace("/dbfs/", "/")
spark_path = "{}/Bitcoin_tweets.csv".format(spark_dir)
df_tweets = spark.read \
  .option("header", "true") \
  .csv(
    spark_path,
    enforceSchema = True,
    sep = ","
  )

# COMMAND ----------

display(df_tweets)

# COMMAND ----------

df_clean = df_tweets.dropna(subset = ["date"])

# COMMAND ----------

df_clean.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE DATABASE IF NOT EXISTS bitcointweetdb;
# MAGIC DROP TABLE IF EXISTS bitcointweetdb.raw_tweets;

# COMMAND ----------

df_clean.write.saveAsTable("bitcointweetdb.raw_tweets")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from bitcointweetdb.raw_tweets

# COMMAND ----------

df_bronze = df_clean.withColumn(
  "parsed_date",
  F.to_timestamp(
    F.col("date"),
    "yyyy-MM-dd HH:mm:ss"
  )
)\
  .dropna(
    subset = ["parsed_date"]
  )\
  .drop("user_name", "user_location", "user_description", "source", "parsed_date")

# COMMAND ----------

display(df_bronze)

# COMMAND ----------

df_bronze.write.saveAsTable("bitcointweetdb.bronze_tweets")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select max(date), min(date) from bitcointweetdb.bronze_tweets

# COMMAND ----------


