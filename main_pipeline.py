import os

from settings import *
from train_model.train_summary import FineTuningSummary
from train_model.sentiment_analysis import SentimentAnalysis


def run_pipeline():
    summary = FineTuningSummary(
        os.path.join(DATA_PATH, RAW_DATA_FILE),
        os.path.join(DATA_PATH, VALIDATION_DATA_FILE),
        os.path.join(OUTPUT_PATH_MODELS, SUMMARY_MODEL),
        os.path.join(OUTPUT_PATH_RESULTS, SUMMARY_OUTPUT)
    )

    sentiment_analysis = SentimentAnalysis(
        os.path.join(DATA_PATH, RAW_DATA_FILE),
        os.path.join(DATA_PATH, VALIDATION_DATA_FILE),
        os.path.join(OUTPUT_PATH_MODELS, SENTIMENTS_MODEL, SENTIMENT_MODEL_NAME)
    )

    # summary.inference_summary()
    sentiment_analysis.train_sentiments()


if __name__ == '__main__':
    run_pipeline()
