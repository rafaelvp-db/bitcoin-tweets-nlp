
from pyspark.sql.session import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType

from bitcoin_tweets_nlp.pipelines import inference

def test_sentiment_dist_inference():

    spark = SparkSession.builder.appName("pytest").getOrCreate()

    dist_pipeline = inference.SentimentDistributedPipeline()
    columns = ["text"]
    data = [Row("I think bitcoin is going down :(")]
    
    df = spark.createDataFrame(data=data, schema=columns)
    df = dist_pipeline.predict(df, column_name = "text")
    assert df.count() == 1
    assert "prediction" in df.columns