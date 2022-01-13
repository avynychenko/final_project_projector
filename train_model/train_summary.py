import pandas as pd
import torch
import os

from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from train_model.dataset_preparation_for_sum import DatasetPreparation
from transformers import AutoModelWithLMHead

import warnings
warnings.filterwarnings("ignore")


class FineTuningSummary:
    def __init__(
            self,
            raw_data,
            validation_data,
            output_summary,
            file_output
    ):
        """
        # :param dir:
        """

        self.raw_data = raw_data
        self.validation_data = validation_data
        self.output_summary = output_summary
        self.file_output = file_output
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", truncation=True)

        self.MAX_LEN = 180
        self.SUMMARY_LEN = 40
        self.BATCH_SIZE = 5
        self.TRAIN_EPOCHS = 2
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
                    max_length=self.MAX_LEN,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
                preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                         generated_ids]
                target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                          y]

                predictions.extend(preds)
                actuals.extend(target)
        return predictions, actuals

    def text_preprocessing(self, ):
        """
        :return:
        """

        df = pd.read_csv(self.raw_data, usecols=['Summary', 'Text'])

        df_summary = df.rename(columns={'Text': 'ctext', 'Summary': 'text'})
        df_summary['ctext'] = 'summarize: ' + df_summary.ctext

        # replace('<br />') token to whitespace
        df_summary['ctext'] = df_summary['ctext'].apply(lambda x: x.replace('<br />', ' '))

        return df_summary

    def text_preprocessing_val(self):

        df = pd.read_csv(self.validation_data, usecols=['Summary', 'Text'])

        df_summary = df.rename(columns={'Text': 'ctext', 'Summary': 'text'})
        df_summary['ctext'] = 'summarize: ' + df_summary.ctext

        # replace('<br />') token to whitespace
        df_summary['ctext'] = df_summary['ctext'].apply(lambda x: x.replace('<br />', ' '))

        return df_summary

    def fine_tuning_model(self):
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
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model = model.to(self.device)
        print(self.device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)

        train_loss = []
        for epoch in range(self.TRAIN_EPOCHS):
            loss = self.train(model=model,
                              epoch=self.TRAIN_EPOCHS,
                              loader=training_loader,
                              optimizer=optimizer)
            train_loss.append(loss)

        # Saving the model after training
        print('Saving fine-tuned summary model to: ' + self.output_summary)
        model.save_pretrained(self.output_summary)

    def inference_summary(self):
        # load model
        model = AutoModelWithLMHead.from_pretrained(self.output_summary)
        model = model.to(self.device)

        val_data = self.text_preprocessing_val()

        df = pd.read_csv(self.validation_data)

        # prepare dataset in proper view for model
        val_set = DatasetPreparation(val_data, self.tokenizer, self.MAX_LEN, self.SUMMARY_LEN)

        val_params = {
            'batch_size': self.BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0
        }

        val_loader = DataLoader(val_set, **val_params)

        predictions, actuals = self.validate(model, val_loader)
        final_df = pd.DataFrame({'Generated Summary': predictions, 'Actual Text': actuals})

        final_df = pd.concat([df[['Text', 'Score', 'Summary']], final_df[['Generated Summary']]], axis=1)
        # save predictions
        path = '/'.join(self.file_output.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

        print('Summaries were successfully generated, saving...')
        final_df.to_csv(self.file_output, index=False)
