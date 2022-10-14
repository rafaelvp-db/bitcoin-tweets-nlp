import logging

from bitcoin_tweets_nlp.pipelines import inference


logger = logging.getLogger(__name__)
logger.setLevel("ERROR")

def test_sentiment_inference(caplog):

    sentences = [
        """don't think bitcoin is going 
        to do that well today 👎🏻 😢""",
        """thanks Elon I'm rich!"""
    ]

    pipeline = inference.SentimentPipeline()
    sentiments = [p['logits'] for p in pipeline.predict(sentences)]
    print(sentiments)
    
    assert sentiments[0] == 0
    assert sentiments[1] == 2


