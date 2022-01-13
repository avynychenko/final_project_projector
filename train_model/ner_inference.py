import stanza
from tqdm import tqdm
import pandas as pd


class NER_Extraction:
    def __init__(
            self,
            validation_data,
            ner_output
    ):
        """
        # :param dir:
        """
        self.validation_data = validation_data
        self.ner_output = ner_output

    @staticmethod
    def generate_ner_objects(model, text):
        doc = model(text)

        locations = [ent.text for ent in doc.ents if ent.type in {'LOC', 'B-LOC', 'I-LOC', 'E-LOC'}]
        organizations = [ent.text for ent in doc.ents if ent.type in {'ORG', 'B-ORG', 'I-ORG', 'E-ORG'}]

        res = list(set(locations + organizations))

        return res

    def ner_inference(self):

        df = pd.read_csv(self.validation_data)

        nlp_ner = stanza.Pipeline(lang='en', processors='tokenize,ner')

        tqdm.pandas(desc='Extracting keywords from NER Model: ORG, LOC')
        df['ner_keywords_hashtags'] = df['Text'].progress_apply(lambda text: self.generate_ner_objects(nlp_ner, text))
        df['ner_keywords_hashtags'] = df['ner_keywords_hashtags'].apply(lambda x: ['#' + '_'.join(y.lower().split()) for y in x])

        print('Locations, Organizations entities were successfully extracted, saving...')
        df.to_csv(self.ner_output, index=False)
