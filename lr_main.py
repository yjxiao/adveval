import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

from model import LogisticRegressor
import cnn_main

parser = argparse.ArgumentParser(description="Action Item Detection SVM/LR Classifier")
parser.add_argument('--data', type=str, default='./data',
                    help="location of the data folder")
parser.add_argument('--gendata', type=str, default='generated.txt',
                    help="location of the generated data to be evaluated")
parser.add_argument('--model', type=str, default='LR',
                    help="type of classifier (LR, SVM)")
parser.add_argument('--max_ngram', type=int, default=1,
                    help="maximum number of ngrams to use")
parser.add_argument('--batch_size', type=int, default=32,
                    help="batch size")
parser.add_argument('--tau', type=float, default=0,
                    help="probability of using the generated data in training")
parser.add_argument('--epochs', type=int, default=40,
                    help="number of epochs")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of steps in each epoch")
parser.add_argument('--lr', type=float, default=1e-1,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=1e-3,
                    help="weight decay")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use cuda")
args = parser.parse_args()


class Data:
    def __init__(self, data, labels):
        self.data = data.toarray().astype(np.float32)
        self.feat_size = self.data.shape[1]
        self.labels = np.array(labels)

    @property
    def size(self):
        return len(self.labels)

    def get_batch(self, batch_size, start_id=None):
        if start_id is None:
            batch_idx = np.random.choice(np.arange(self.size), batch_size)
        else:
            batch_idx = np.arange(start_id, start_id + batch_size)
        batch_data = self.data[batch_idx]
        batch_labels = self.labels[batch_idx]
        return torch.from_numpy(batch_data), torch.from_numpy(batch_labels)


def get_data(filepath):
    data = []
    labels = []
    with open(filepath) as f:
        for line in f:
            label, text = line.strip().split('\t')
            data.append(text)
            labels.append(int(label))
    return data, labels


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    return './saves/{0}.tau{1:.1f}.lr.{2}.pt'.format(args.gendata, args.tau, dataset)


def main(args):
    filenames = ['train.txt', 'valid.txt', 'test.txt', args.gendata]
    datapaths = [os.path.join(args.data, fn) for fn in filenames]

    train_data, train_labels = get_data(datapaths[0])
    valid_data, valid_labels = get_data(datapaths[1])
    test_data, test_labels = get_data(datapaths[2])
    gen_data, gen_labels = get_data(datapaths[3])
    num_classes = len(set(train_labels))
    
    print("Number of training examples: ", len(train_data))
    print("Number of validation examples: ", len(valid_data))
    print("Number of test examples: ", len(test_data))
    print("Number of generated examples: ", len(gen_data))    
    vect = TfidfVectorizer(ngram_range=(1, args.max_ngram),
                           stop_words='english', min_df=2,
                           max_features=20000)
    train_tfidf = vect.fit_transform(train_data)
    valid_tfidf = vect.transform(valid_data)
    test_tfidf = vect.transform(test_data)
    gen_tfidf = vect.transform(gen_data)
    feat_size = train_tfidf.shape[1]
    train, valid, test, gen = [Data(tfidf, labels) for tfidf, labels in zip([train_tfidf, valid_tfidf, test_tfidf, gen_tfidf], [train_labels, valid_labels, test_labels, gen_labels])]

    device = torch.device('cpu' if args.nocuda else 'cuda')
    model = LogisticRegressor(feat_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_acc = 0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss, train_acc = cnn_main.train([train, gen], args.tau, model,
                                               criterion, optimizer, device)
            valid_loss, valid_acc = cnn_main.evaluate(valid, model, criterion, device)
            print('-' * 80)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time() - epoch_start_time)
            print(meta + "| train loss {:5.2f} | train acc {:4.1f} ".format(
                train_loss, train_acc * 100))
            print(len(meta) * ' ' + "| valid loss {:5.2f} | valid acc {:4.1f}".format(
                valid_loss, valid_acc * 100), flush=True)
            
            if valid_acc > best_acc:
                with open(get_savepath(args), 'wb') as f:
                    torch.save(model, f)
                    best_acc = valid_acc
                    
    except KeyboardInterrupt:
        print('-' * 80)
                
    with open(get_savepath(args), 'rb') as f:
        model = torch.load(f)

    test_loss, test_acc = cnn_main.evaluate(test, model, criterion, device)
    print('=' * 80)
    print("| End of training | test loss {:5.2f} | test acc {:4.1f}".format(
        test_loss, test_acc * 100))
    print('=' * 80)
                            

if __name__ == '__main__':
    main(args)
    
