import os

basedir = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = f'{basedir}/data'

RAW_DATA_FILE = 'train_reviews.csv'
SUMMARY_MODEL = 'summary'
OUTPUT_PATH_MODELS = f'{DATA_PATH}/saved_models'
