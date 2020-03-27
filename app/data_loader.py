import os
import random
import torch
from torchtext import data


class DataLoader:
    def __init__(self, path: str, text_field: str, label_field):
        self._path = path
        self._text = data.Field(tokenize='spacy', lower=True, batch_first=True, include_lengths=True)
        self._label = data.LabelField(dtype=torch.float, batch_first=True)
        self._fields = [(text_field, self._text), (label_field, self._label)]

    def _split_data(self):
        file_format = os.path.splitext(self._path)[1]
        dataset = data.TabularDataset(path=self._path, format=file_format, fields=self._fields, skip_header=True)
        train_data, test_data = dataset.split(split_ratio=0.2, random_state=random.seed(42))
        return train_data, test_data

    def get_train_test_iters(self, embeddings_name: str, batch_size: int):
        train_data, test_data = self._split_data()
        self._text.build_vocab(train_data, min_freq=3, vectors=embeddings_name)
        self._label.build_vocab(train_data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_iterator, test_iterator = data.BucketIterator.splits(
            datasets=(train_data, test_data), batch_size=batch_size, device=device)
        return train_iterator, test_iterator
