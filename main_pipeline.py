import os

from settings import *
from train_model.train_summary import FineTuningSummary

def run_pipeline():
    train_summary = FineTuningSummary(
        os.path.join(DATA_PATH, RAW_DATA_FILE),
        os.path.join(OUTPUT_PATH_MODELS, SUMMARY_MODEL)
    )

    train_summary.text_preprocessing()


if __name__ == '__main__':
    run_pipeline()