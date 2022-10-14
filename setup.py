from setuptools import find_packages, setup
from bitcoin_tweets_nlp import __version__

setup(
    name="bitcoin_tweets_nlp",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="Bitcoin Tweets NLP",
    author="https://www.github.com/rafaelvp-db"
)