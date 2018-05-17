import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_size, num_filters, filter_sizes, dropout):
        super().__init__()
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv1d(embed_size, num_filters, fs) for fs in filter_sizes])
        self.drop = nn.Dropout(dropout)
        code_size = len(filter_sizes) * num_filters
        self.labeler = nn.Linear(code_size, num_classes)

    def forward(self, inputs):
        # emb: batch_size x embed_size x length
        emb = self.drop(self.lookup(inputs).permute(0, 2, 1))
        # convolution and max pooling over time (length) for each filter
        codes = [F.relu(conv(emb).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        code = self.drop(torch.cat(codes, dim=1))
        # probs of shape: batch_size x num_classes
        return self.labeler(code)
