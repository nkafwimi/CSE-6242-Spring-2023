# !/usr/bin/env python3
# -*- coding: utf-8 -*-


from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

torch.manual_seed(0)
np.random.seed(0)
import torch
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string

import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('.data/train.csv')
test_df = pd.read_csv('.data/test.csv')

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=1)

train_df = train_df.drop(["keyword", "location"], axis=1)
val_df = val_df.drop(["keyword", "location"], axis=1)
test_df = test_df.drop(["keyword", "location"], axis=1)


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer):
        texts = df.text.values.tolist()
        texts = [self._preprocess(text) for text in texts]
        self._print_random_samples(texts)
        self.texts = [tokenizer(text, padding='max_length',
                                max_length = 150,
                                truncation = True,
                                return_tensors = 'pt')
                      for text in texts]
        if 'target' in df:
            classes = df.target.values.tolist()
            self.labels = classes

    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")


    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)

        # text = self._lowercase(text)
        text = self._remove_punctuation(text)
        # text = self._remove_numbers(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        # text_tokens = self._stemming(text_tokens)
        text = self._stitch_text_tokens_together(text_tokens)

        return text.strip()

    def _remove_amp(self, text):
        return text.replace("&amp;", " ")

    def _remove_mentions(self, text):
        return re.sub(r'(@.*?)[\s]', ' ', text)

    def _remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text)

    def _remove_retweets(self, text):
        return re.sub(r'^RT[\s]+', ' ', text)

    def _remove_links(self, text):
        return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

    def _remove_hashes(self, text):
        return re.sub(r'#', ' ', text)

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words('english')

        return [token for token in text_tokens if token not in stop_words]

    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    def _remove_numbers(self, text):
        return re.sub(r'\d+', ' ', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return ''.join(character for character in text if character not in string.punctuation)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label


class TweetClassifier(nn.Module):
    def __init__(self, base_model):
        super(TweetClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # get the CLS token of embedding size, [batch_size*sequence_len*embedding_dim] to [batch_size*embedding_dim]
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:,0]
        # convert the CLS token to label 0 or 1
        x = self.fc1(bert_out)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    best_val_los  = float('inf')
    early_stopping_threshold_count = 0
    print('cuda yes? ', torch.cuda.is_available())
    device = torch.device('cuda')

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_input, train_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            train_label = train_label.to(device)

            output = model(input_ids, attention_mask)
            loss = criterion(output, train_label.float().unsqueeze(1))
            total_loss_train += loss.item()

            acc = ((output>=0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            total_loss_val = 0

            model.eval()

            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)
                loss = criterion(output, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  )

            if best_val_los > total_loss_val:
                best_val_los = total_loss_val
                torch.save(model, f'best_model.pt')
                print('saved model')
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1

            if early_stopping_threshold_count >= 1:
                print('early stopping')
                break



BERT_MODEL = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
base_model = AutoModel.from_pretrained(BERT_MODEL)

train_dataloader = DataLoader(TweetDataset(train_df, tokenizer), batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(TweetDataset(val_df, tokenizer), batch_size=8, num_workers=0)

model = TweetClassifier(base_model)

learning_rate = 1e-5
epochs = 5

train(model, train_dataloader, val_dataloader, learning_rate, epochs)

#####
