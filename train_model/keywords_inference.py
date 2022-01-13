import yake
from tqdm import tqdm
import pandas as pd


class KeywordsExtraction:
    def __init__(
            self,
            validation_data,
            keywords_output
    ):
        """
        # :param dir:
        """
        self.validation_data = validation_data
        self.keywords_output = keywords_output

    def keywords_inference(self):
        df = pd.read_csv(self.validation_data)
        yake_list = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Key words extraction with YAKE approch'):
            kw_extractor = yake.KeywordExtractor(n=3, top=10)
            keywords = kw_extractor.extract_keywords(row['Text'])
            # The lower the score, the more relevant the keyword is. So, sorting is applied.
            keywords.sort(key=lambda x: x[1])
            keywords = [x[0] for x in keywords]
            yake_list.append(keywords)

        df['keywords_hashtags'] = yake_list
        df['keywords_hashtags'] = df['keywords_hashtags'].apply(lambda x: ['#' + '_'.join(y.lower().split()) for y in x])

        print('Key words were successfully generated, saving...')
        df.to_csv(self.keywords_output, index=False)
