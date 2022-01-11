from torchtext.legacy import data
import torch
import pandas as pd



class SentimentAnalysis:
    def __init__(
            self,
            raw_data,
            validation_data,
            sentiments_model_path
    ):
        """
        # :param dir:
        """

        self.raw_data = raw_data
        self.validation_data = validation_data
        self.sentiments_model_path = sentiments_model_path

    # define metric
    @staticmethod
    def binary_accuracy(preds, y):
        # round predictions to the closest integer
        rounded_preds = torch.round(preds)

        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    def train(self, model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        # set the model in training phase
        model.train()

        for batch in iterator:
            # resets the gradients after every batch
            optimizer.zero_grad()

            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze()

            # compute the loss
            loss = criterion(predictions, batch.label)

            # compute the binary accuracy
            acc = self.binary_accuracy(predictions, batch.label)

            # backpropage the loss and compute the gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, model, iterator, criterion):

        # initialize every epoch
        epoch_loss = 0
        epoch_acc = 0

        # deactivating dropout layers
        model.eval()

        # deactivates autograd
        with torch.no_grad():
            for batch in iterator:
                # retrieve text and no. of words
                text, text_lengths = batch.text

                # convert to 1d tensor
                predictions = model(text, text_lengths).squeeze()

                # compute loss and accuracy
                loss = criterion(predictions, batch.label)
                acc = self.binary_accuracy(predictions, batch.label)

                # keep track of loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train_sentiments(self):

        # df = pd.read_csv(self.raw_data)
        #
        # # rules for suppressing ratings
        # # 1. If the rating is 1 or 2, then the reviewText is treated as the negative, labeled - 1.
        # # 2. If the rating is 4 or 5, then the reviewText is treated as the positive, labeled + 1.
        # # 3. Ignore all reviewsText with the rating 3. Since they belong to a neutral sentiment.
        #
        # df = df[df["Score"] != '3']  # need datatype=object
        # df["label"] = df["Score"].apply(lambda rating: +1 if str(rating) > '3' else -1)

        # # Split dataset into test and validation
        # from sklearn.model_selection import train_test_split
        #
        # X = pd.DataFrame(df, columns=["Text"])
        # y = pd.DataFrame(df, columns=["label"])
        #
        # train_X, test_X, trian_y, test_y = train_test_split(X, y, random_state=50)

        SEED = 2019
        # Torch
        torch.manual_seed(SEED)

        TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
        LABEL = data.LabelField(dtype=torch.float, batch_first=True)

        fields = [('Score', LABEL), (None, None), ('Text', TEXT)]

        # loading custom dataset
        training_data = data.TabularDataset(path=self.raw_data, format='csv', fields=fields, skip_header=True)

        # print preprocessed text

        import random
        train_data, valid_data = training_data.split(split_ratio=0.8, random_state=random.seed(SEED))

        # initialize glove embeddings
        TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
        LABEL.build_vocab(train_data)

        # check whether cuda is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set batch size
        BATCH_SIZE = 64

        # Load an iterator
        train_iterator, valid_iterator = data.BucketIterator.splits(
            (train_data, valid_data),
            batch_size=BATCH_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=device)


        # define hyperparameters
        size_of_vocab = len(TEXT.vocab)
        embedding_dim = 100
        num_hidden_nodes = 32
        num_output_nodes = 1
        num_layers = 2
        bidirection = True
        dropout = 0.2

        from train_model.model_sentiments import classifier
        # instantiate the model
        model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                           bidirectional=True, dropout=dropout)

        # architecture
        print(model)

        # No. of trianable parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'The model has {count_parameters(model):,} trainable parameters')

        # Initialize the pretrained embedding
        pretrained_embeddings = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

        print(pretrained_embeddings.shape)

        import torch.optim as optim
        import torch.nn as nn

        # define optimizer and loss
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCELoss()

        # push to cuda if available
        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 5
        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            # train the model
            train_loss, train_acc = self.train(model, train_iterator, optimizer, criterion)

            # evaluate the model
            valid_loss, valid_acc = self.evaluate(model, valid_iterator, criterion)

            # save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), self.sentiments_model_path)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')




