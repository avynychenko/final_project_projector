import os
import torch
import random
import spacy
import pickle

import pandas as pd
from tqdm import tqdm
from torchtext.legacy import data
import torch.optim as optim
import torch.nn as nn
from train_model.model_sentiments import classifier

import warnings
warnings.filterwarnings("ignore")


class SentimentAnalysis:
    def __init__(
            self,
            raw_data,
            validation_data,
            vocab_path,
            sentiments_model_path,
            sentiments_output
    ):
        """
        # :param dir:
        """

        self.raw_data = raw_data
        self.validation_data = validation_data
        self.vocab_path = vocab_path
        self.sentiments_model_path = sentiments_model_path
        self.sentiments_output = sentiments_output

        self.SEED = 2000
        self.BATCH_SIZE = 64

        # define hyperparameters
        self.embedding_dim = 100
        self.num_hidden_nodes = 32
        self.num_output_nodes = 1
        self.num_layers = 2
        self.bidirection = True
        self.dropout = 0.2
        self.N_EPOCHS = 5

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define metric
    @staticmethod
    def binary_accuracy(preds, y):
        # round predictions to the closest integer
        rounded_preds = torch.round(preds)

        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self, model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        # set the model in training phase
        model.train()

        for batch in iterator:
            # resets the gradients after every batch
            optimizer.zero_grad()

            # retrieve text and no. of words
            text, text_lengths = batch.Text

            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze()

            # compute the loss
            loss = criterion(predictions, batch.Score)

            # compute the binary accuracy
            acc = self.binary_accuracy(predictions, batch.Score)

            # backpropage the loss and compute the gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, model, iterator, criterion):

        epoch_loss = 0
        epoch_acc = 0

        # deactivating dropout layers
        model.eval()

        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.Text

                # convert to 1d tensor
                predictions = model(text, text_lengths).squeeze()

                # compute loss and accuracy
                loss = criterion(predictions, batch.Score)
                acc = self.binary_accuracy(predictions, batch.Score)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def load_dataset(self):

        torch.manual_seed(self.SEED)

        TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
        LABEL = data.LabelField(dtype=torch.float, batch_first=True)

        fields = [('Score', LABEL), ('Summary', None), ('Text', TEXT)]

        # loading custom dataset
        training_data = data.TabularDataset(path=self.raw_data, format='csv', fields=fields, skip_header=True)

        train_data, valid_data = training_data.split(split_ratio=0.8, random_state=random.seed(self.SEED))

        # initialize glove embeddings
        TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
        LABEL.build_vocab(train_data)

        with open(self.vocab_path, 'wb') as f:
            pickle.dump(TEXT.vocab.stoi, f)

        return TEXT, train_data, valid_data

    def predict(self, model, sentence, vocab):
        nlp = spacy.load("en_core_web_sm")
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  # tokenize the sentence
        indexed = [vocab[t] for t in tokenized]  # convert to integer sequence
        length = [len(indexed)]  # compute no. of words
        tensor = torch.LongTensor(indexed).to(self.device)  # convert to tensor
        tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
        length_tensor = torch.LongTensor(length)  # convert to tensor
        prediction = model(tensor, length_tensor)
        res = round(prediction.item(), 0)
        if res == 0:
            res = 1
        else:
            res = 0
        return res

    def train_sentiments(self):

        TEXT, train_data, valid_data = self.load_dataset()

        # create data iterators
        train_iterator, valid_iterator = data.BucketIterator.splits(
            (train_data, valid_data),
            batch_size=self.BATCH_SIZE,
            sort_key=lambda x: len(x.Text),
            sort_within_batch=True,
            device=self.device)

        size_of_vocab = len(TEXT.vocab)
        print('Size of vocab: ' + str(size_of_vocab))

        # instantiate the model
        model = classifier(size_of_vocab, self.embedding_dim, self.num_hidden_nodes,
                           self.num_output_nodes, self.num_layers,
                           bidirectional=True, dropout=self.dropout)

        # model architecture and # of trainable parameters
        print(model)
        print(f'The model has {self.count_parameters(model):,} trainable parameters')

        # Initialize the pretrained embedding
        pretrained_embeddings = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

        # define optimizer and loss
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCELoss()

        # push to cuda if available
        model = model.to(self.device)
        criterion = criterion.to(self.device)

        best_valid_loss = float('inf')

        print('Start training sentiment model')
        for epoch in range(self.N_EPOCHS):

            train_loss, train_acc = self.train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate(model, valid_iterator, criterion)

            # save the best model
            path = '/'.join(self.sentiments_model_path.split('/')[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), self.sentiments_model_path)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    def inference_sentiments(self):

        # read text vocabulary
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        size_of_vocab = len(vocab)

        # instantiate the model
        model = classifier(size_of_vocab, self.embedding_dim, self.num_hidden_nodes,
                           self.num_output_nodes, self.num_layers,
                           bidirectional=True, dropout=self.dropout)
        model.load_state_dict(torch.load(self.sentiments_model_path))
        model = model.to(self.device)
        model.eval()

        df = pd.read_csv(self.validation_data)
        tqdm.pandas(desc='Sentiments generation')
        df['Predicted sentiment'] = df['Text'].progress_apply(lambda x: self.predict(model, x, vocab))

        print('Sentiments were successfully generated, saving...')
        df.to_csv(self.sentiments_output, index=False)




