import pandas as pd
import numpy as np


class FinalOutput:
    def __init__(
            self,
            input_path,
            output_path
    ):
        """
        # :param dir:
        """
        self.input_path = input_path
        self.output_path = output_path

    def generate_results(self):

        df = pd.read_csv(self.input_path)

        df = df.rename(columns={'Score': 'True sentiment'})
        df['True sentiment'] = np.where(df['True sentiment'] == 1, 'Positive', 'Negative')
        df['Predicted sentiment'] = np.where(df['Predicted sentiment'] == 1, 'Positive', 'Negative')

        # place columns in a proper order
        df = df[['Text', 'Predicted sentiment', 'True sentiment', 'Generated Summary', 'Summary', 'keywords_hashtags',
                 'ner_keywords_hashtags']]

        # remove empty lists
        df['ner_keywords_hashtags'] = df['ner_keywords_hashtags'].apply(lambda x: x.replace('[]', ''))
        df['keywords_hashtags'] = df['keywords_hashtags'].apply(lambda x: x.replace('[]', ''))

        # sorting df before saving
        df = df.sort_values(by='Predicted sentiment', ascending=False)

        print('Final output were successfully generated, saving to: ' + self.output_path)
        df.to_excel(self.output_path, index=False)