import os

basedir = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = f'{basedir}/data'

RAW_DATA_FILE = 'train_reviews.csv'
VALIDATION_DATA_FILE = 'validation_reviews.csv'

SUMMARY_MODEL = 'summary'
SENTIMENTS_MODEL = 'sentiments'
SENTIMENT_MODEL_NAME = 'best_sentiments.pt'
OUTPUT_PATH_MODELS = f'{DATA_PATH}/saved_models'
OUTPUT_PATH_RESULTS = f'{DATA_PATH}/processed'

VOCAB = 'saved_vocab.pkl'

SUMMARY_OUTPUT = 'summary_preliminary.csv'
SENTIMENTS_OUTPUT = 'sentiments_preliminary.csv'
KEYWORDS_OUTPUT = 'keywords_preliminary.csv'
NER_OUTPUT = 'ner_preliminary.csv'

FINAL_OUTPUT = 'final_results.xlsx'
