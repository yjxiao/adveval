import os
from collections import Counter
import pickle
import numpy as np
import torch

PAD_ID, UNK_ID = 0, 1


class Corpus:
    def __init__(self, datadir, min_n=2, max_vocab_size=None, max_length=None):
        self.min_n = min_n
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        filenames = ['train.txt', 'valid.txt', 'test.txt']
        self.datapaths = [os.path.join(datadir, x) for x in filenames]
        self._construct_vocab()
        self.num_classes = len(self.idx2label)
        self.train, self.valid, self.test = [
            Data(dp, (self.word2idx, self.label2idx), max_length) \
            for dp in self.datapaths]

    def _construct_vocab(self):
        self._vocab = Counter()
        labels = []
        for datapath in self.datapaths[:2]:
            with open(datapath) as f:
                # parse data files to construct vocabulary            
                for line in f:
                    label, text = line.strip().split('\t')
                    if label not in labels:
                        labels.append(label)
                    self._vocab.update(text.lower().split())
        vocab_size = len([x for x in self._vocab if self._vocab[x] >= self.min_n])
        self.idx2word = ['_PAD', '_UNK'] + list(next(zip(*self._vocab.most_common(self.max_vocab_size)))[:vocab_size])
        self.word2idx = dict((w, i) for (i, w) in enumerate(self.idx2word))
        self.idx2label = sorted(labels)
        self.label2idx = dict((w, i) for (i, w) in enumerate(self.idx2label))

        
class Data:
    def __init__(self, datapath, vocabs, max_length=None):
        word2idx, label2idx = vocabs
        self.vocab_size = len(word2idx)
        texts = []
        labels = []
        with open(datapath) as f:
            for line in f:
                label, text = line.strip().split('\t')
                words = text.lower().split()
                if max_length is not None:
                    words = words[:max_length]
                indices = [word2idx.get(x, UNK_ID) for x in words]
                texts.append(indices)
                labels.append(label2idx[label])
        self.texts = np.array(texts)
        self.labels = np.array(labels)
            
    def shuffle(self):
        # no need to shuffle. during training, batchs are randomly sampled
        perm = np.random.permutation(self.size)
        self.texts = self.texts[perm]
        self.labels = None if self.labels is None else self.labels[perm]
        self.topics = self.topics[perm]
        
    @property
    def size(self):
        return len(self.texts)
    
    def get_batch(self, batch_size, start_id=None):
        if start_id is None:
            batch_idx = np.random.choice(np.arange(self.size), batch_size)
        else:
            batch_idx = np.arange(start_id, start_id + batch_size)
        batch_texts = self.texts[batch_idx]
        batch_labels = self.labels[batch_idx]
        max_len = max([len(x) for x in batch_texts]) 
        text_tensor = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
        for i, x in enumerate(batch_texts):
            n = len(x)
            text_tensor[i][:n] = torch.from_numpy(np.array(x))
        label_tensor = torch.from_numpy(batch_labels)
        return text_tensor, label_tensor
