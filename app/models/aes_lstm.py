import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class AesLstm:
    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(AesLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)  # [batch size, sent_len, emb dim]
        # we will get text_length from torchtext batch
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)  # hidden = [batch size, 2 * num layers, hidden_dim]

        # concat final forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch size, 2 * hidden_dim]
        dense_outputs = self.fc(hidden)
        outputs = self.act(dense_outputs)
        return outputs
