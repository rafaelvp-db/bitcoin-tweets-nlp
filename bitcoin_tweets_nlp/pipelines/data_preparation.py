import pandas as pd
from pyspark.sql import functions as F


class DataPreparationPipeline:
    def __init__(
        self,
        spark,
        db_name,
    ):

        self.spark = spark
        self.db_name = db_name