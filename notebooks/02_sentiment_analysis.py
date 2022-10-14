# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Transformer Models
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC <img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png" width="700"/>
# MAGIC <br></br>
# MAGIC <hr></hr>
# MAGIC 
# MAGIC ### What is a Transformer?
# MAGIC 
# MAGIC A **transformer** is a **deep learning model** that adopts the mechanism of **self-attention**, differentially weighting the significance of each part of the input data. It is used primarily in the fields of **natural language processing** (NLP) and **computer vision** (CV), although some researchers have recently investigated applications in time series forecasting recently.
# MAGIC 
# MAGIC Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not have to process one word at a time. This allows for more parallelization than RNNs and therefore reduces training times.[1]
# MAGIC 
# MAGIC Transformers were introduced in 2017 by a team at Google Brain[1] and are increasingly the model of choice for **NLP** problems, replacing **RNN** models such as long short-term memory (LSTM). The additional training parallelization allows training on larger datasets. This led to the development of pretrained systems such as **BERT** (Bidirectional Encoder Representations from Transformers) and **GPT** (Generative Pre-trained Transformer), which were trained with large language datasets, such as the Wikipedia Corpus and Common Crawl, and can be fine-tuned for specific tasks.
# MAGIC 
# MAGIC <br></br>

# COMMAND ----------

# DBTITLE 1,Example: Transformer Based Sentiment Analysis with Hugging Face
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("rabindralamsal/BERTsent")

labels = ["negative", "neutral", "positive"]

sentence = "thanks Elon I'm rich!"
tokenized = tokenizer(sentence, return_tensors = "pt")
sentiment = model(tokenized["input_ids"])
print(sentiment)
sentiment_label = labels[torch.argmax(sentiment.logits)]
result = {
  "sentence": sentence,
  "sentiment": sentiment_label
}

print(f"\n{result}")

# COMMAND ----------

torch.argmax(sentiment.logits).item()

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark_df = spark.sql("select text from bitcointweetdb.bronze_tweets")

# COMMAND ----------

@pandas_udf("float", PandasUDFType.SCALAR)
def predict_udf(text):
  tokenizer = AutoTokenizer.from_pretrained("/dbfs/Users/YOUR_USER/bitcoin_tweets/tokenizer")
  model = AutoModelForSequenceClassification.from_pretrained("/dbfs/Users/YOUR_USER/bitcoin_tweets/model")
  # here text is Series
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
  
df_predicted = spark_df.withColumn("predicted", predict_udf(col("text")))
display(df_predicted)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col, udf
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@pandas_udf("int")
def tok_udf(text: pd.Series) -> pd.Series:
    tokenizer = AutoTokenizer.from_pretrained("/dbfs/Users/YOUR_USER/bitcoin_tweets/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("/dbfs/Users/YOUR_USER/bitcoin_tweets/model")
    tokenized = tokenizer(
                text,
                return_tensors = "pt",
                padding = True,
                truncation = True)
    prediction = model(tokenized["input_ids"])
    result_tmp = torch.argmax(prediction.logits).item()
    print("HERE RESULTS TOKENIZER", result_tmp, "CHECK HERE")
    return result_tmp



# COMMAND ----------

text = df_pd.values[0][0]
tokenizer = AutoTokenizer.from_pretrained("/dbfs/Users/YOUR_USER/bitcoin_tweets/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("/dbfs/Users/YOUR_USER/bitcoin_tweets/model")
tokenized = tokenizer(
            text,
            return_tensors = "pt",
            padding = True,
            truncation = True)
prediction = model(tokenized["input_ids"])
#result_tmp = torch.argmax(prediction.logits).item()

# COMMAND ----------

np.array(prediction.logits.detach()[0])

# COMMAND ----------


