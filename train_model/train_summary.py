import pandas as pd
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from train_model.dataset_preparation_for_sum import DatasetPreparation


class FineTuningSummary:
    def __init__(
            self,
            raw_data,
            output_summary
    ):
        """
        # :param dir:
        """

        self.raw_data = raw_data
        self.output_summary = output_summary
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", truncation=True)

        self.MAX_LEN = 512
        self.SUMMARY_LEN = 150
        self.BATCH_SIZE = 5
        self.TRAIN_EPOCHS = 1
        # self.VAL_EPOCHS = 1
        self.LEARNING_RATE = 1e-4
        self.RANDOM_SEED = 42

    def train(self, model, epoch, loader, optimizer):
        model.train()
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(self.device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(self.device, dtype=torch.long)
            mask = data['source_mask'].to(self.device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        return loss.item()

    def validate(self, model, loader):
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, data in enumerate(loader, 0):
                y = data['target_ids'].to(self.device, dtype=torch.long)
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=150,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
                preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                         generated_ids]
                target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                          y]
                print(generated_ids)
                print(preds)
                print(target)

                predictions.extend(preds)
                actuals.extend(target)
        return predictions, actuals

    def text_preprocessing(self, ):
        """
        :return:
        """

        df = pd.read_csv(self.raw_data, usecols=['Summary', 'Text'])

        df_summary = df.rename(columns={'Text': 'text', 'Summary': 'ctext'})
        df_summary['ctext'] = 'summarize: ' + df_summary.ctext

        # replace('<br />') token to whitespace
        df_summary['text'] = df_summary['text'].apply(lambda x: x.replace('<br />', ' '))

        return df_summary

    def fine_tuning_model(self, df):
        # read data
        data = self.text_preprocessing()

        # prepare dataset in proper view for model
        training_set = DatasetPreparation(data, self.tokenizer, self.MAX_LEN, self.SUMMARY_LEN)

        train_params = {
            'batch_size': self.BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }

        training_loader = DataLoader(training_set, **train_params)

        # load pretrained T5 model
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        model = model.to(self.device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)

        training = self.train(model=model,
                              epoch=self.TRAIN_EPOCHS,
                              loader=training_loader,
                              optimizer=optimizer)

        train_loss = []
        for epoch in range(self.TRAIN_EPOCHS):
            loss = training.train(epoch, training_loader)
            train_loss.append(loss)

        # Saving the model after training
        print('Saving fine-tuned summary model')
        model.save_pretrained(self.output_summary)
        # tokenizer.save_pretrained(path)

        # load model

        # from transformers import AutoModel
        #
        # model = AutoModel.from_pretrained(save_directory, from_tf=True)

        # model =.from_pretrained("path/to/awesome-name-you-picked").
