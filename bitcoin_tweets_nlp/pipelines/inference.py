import numpy as np
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List

class SentimentPipeline:

    def __init__(
        self,
        tokenizer_name_or_path: str = "vinai/bertweet-covid19-base-cased",
        model_name_or_path: str = "rabindralamsal/BERTsent",
        labels: List[str] = ["negative", "neutral", "positive"]
    ):

        self._tokenizer_name_or_path = tokenizer_name_or_path
        self._model_name_or_path = model_name_or_path
        self._labels = labels


    def predict(self, sentences: List[str]) -> List[Dict]:
        
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name_or_path)
        labels = self._labels

        tokenized = tokenizer(
            sentences,
            return_tensors = "pt",
            padding = True,
            truncation = True
        )
        sentiments = model(tokenized["input_ids"])
        return sentiments


class SentimentDistributedPipeline:

    def __init__(
        self,
        tokenizer_name_or_path: str = "vinai/bertweet-covid19-base-cased",
        model_name_or_path: str = "rabindralamsal/BERTsent",
        labels: List[str] = ["negative", "neutral", "positive"],
    ):
    
        self._tokenizer_name_or_path = tokenizer_name_or_path
        self._model_name_or_path = model_name_or_path

    def predict(self, df, column_name):

        @pandas_udf(IntegerType())
        def _predict_udf(text: pd.Series) -> pd.Series:
            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name_or_path)
            model = AutoModelForSequenceClassification.from_pretrained(self._tokenizer_name_or_path)
            result = []
            for item in text.values:
                tokenized = tokenizer(
                item,
                return_tensors = "pt",
                padding = True,
                truncation = True)
                prediction = model(tokenized["input_ids"])
                label = torch.argmax(prediction.logits).item()
                result.append(label)
            return pd.Series(result)

        result = df.withColumn("prediction", _predict_udf(column_name))
        return result