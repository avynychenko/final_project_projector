import os
import logging

from settings import *
from train_model.train_summary import FineTuningSummary
from train_model.sentiment_analysis import SentimentAnalysis
from train_model.keywords_inference import KeywordsExtraction
from train_model.ner_inference import NER_Extraction
from train_model.final_output import FinalOutput


def run_pipeline(train=False):
    # define classes for training and inference
    summary = FineTuningSummary(
        os.path.join(DATA_PATH, RAW_DATA_FILE),
        os.path.join(DATA_PATH, VALIDATION_DATA_FILE),
        os.path.join(OUTPUT_PATH_MODELS, SUMMARY_MODEL),
        os.path.join(OUTPUT_PATH_RESULTS, SUMMARY_OUTPUT)
    )

    sentiment_analysis = SentimentAnalysis(
        os.path.join(DATA_PATH, RAW_DATA_FILE),
        os.path.join(OUTPUT_PATH_RESULTS, SUMMARY_OUTPUT),
        os.path.join(OUTPUT_PATH_MODELS, VOCAB),
        os.path.join(OUTPUT_PATH_MODELS, SENTIMENTS_MODEL, SENTIMENT_MODEL_NAME),
        os.path.join(OUTPUT_PATH_RESULTS, SENTIMENTS_OUTPUT)
    )

    keywords_extraction = KeywordsExtraction(
        os.path.join(OUTPUT_PATH_RESULTS, SENTIMENTS_OUTPUT),
        os.path.join(OUTPUT_PATH_RESULTS, KEYWORDS_OUTPUT)
    )

    ner_extraction = NER_Extraction(
        os.path.join(OUTPUT_PATH_RESULTS, KEYWORDS_OUTPUT),
        os.path.join(OUTPUT_PATH_RESULTS, NER_OUTPUT)
    )

    final_output = FinalOutput(
        os.path.join(OUTPUT_PATH_RESULTS, NER_OUTPUT),
        os.path.join(OUTPUT_PATH_RESULTS, FINAL_OUTPUT)
    )

    if train:
        logging.warning('Start training summary model')
        summary.fine_tuning_model()

        logging.warning('Start training sentiment model')
        sentiment_analysis.train_sentiments()

    else:
        logging.warning('Start summary inference')
        summary.inference_summary()

        logging.warning('Start sentiments inference')
        sentiment_analysis.inference_sentiments()

        logging.warning('Start keywords extraction inference')
        keywords_extraction.keywords_inference()

        logging.warning('Start NER extraction inference')
        ner_extraction.ner_inference()

        logging.warning('Final output generation')
        final_output.generate_results()


if __name__ == '__main__':
    run_pipeline(train=False)
