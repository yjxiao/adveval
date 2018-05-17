import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

from model import CNNClassifier
import data

parser = argparse.ArgumentParser(description="Adversarial Evaluation")
parser.add_argument('--data', type=str, default='./data/yahoo',
                    help="location of the data folder")
parser.add_argument('--gendata', type=str, default='generated.txt',
                    help="name of the generated data to be evaluated")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="max size of vocabulary")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum text length")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--num_filters', type=int, default=128,
                    help="number of filters for CNN")
parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                    help="list of filter sizes as string separated by commas")
parser.add_argument('--lr', type=float, default=3e-3,
                    help="initial learning rate")
parser.add_argument('--tau', type=float, default=0,
                    help="probability of using the generated data in training")
parser.add_argument('--epochs', type=int, default=40,
                    help="maximum training epochs")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of steps in each epoch")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--wd', type=float, default=1e-4,
                    help="weight decay used for regularization")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--nocuda', action='store_true',
                    help="use CUDA")
args = parser.parse_args()
torch.manual_seed(args.seed)


def evaluate(data_source, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, targets = data_source.get_batch(batch_size, i)
        texts = texts.to(device)
        targets = targets.to(device)
        outputs = model(texts)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        predicts = outputs.max(1)[1]
        total_correct += (predicts == targets).sum().item()
    return total_loss / data_source.size, total_correct / data_source.size


def train(data_sources, tau, model, criterion, optimizer, device):
    """Training for a single epoch. """
    model.train()
    total_loss = 0.0
    total_correct = 0
    p = [1 - tau, tau]
    for i in range(args.epoch_size):
        # random choose from train set and generated set
        data_source = np.random.choice(data_sources, p=p)
        texts, targets = data_source.get_batch(args.batch_size)
        texts = texts.to(device)
        targets = targets.to(device)
        outputs = model(texts)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicts = outputs.max(1)[1]
        total_correct += (predicts == targets).sum().item()
    epoch_loss = total_loss / data_source.size
    return epoch_loss, total_correct / data_source.size


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    return './saves/{0}.tau{1:.1f}.{2}.ptb'.format(args.gendata, args.tau, dataset)


def main(args):
    device = torch.device('cpu' if args.nocuda else 'cuda')
    # load corpus and construct model
    corpus = data.Corpus(args.data,
                         max_vocab_size=args.max_vocab,
                         max_length=args.max_length)
    vocab_size = len(corpus.word2idx)
    num_classes = corpus.num_classes
    gendatapath = os.path.join(args.data, args.gendata)
    genset = data.Data(gendatapath, [corpus.word2idx, corpus.label2idx], args.max_length)
    print("\ttraining data size: ", corpus.train.size)
    print("\tgenerated data size: ", genset.size)
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)
    filter_sizes = list(map(int, args.filter_sizes.strip().split(',')))
    model = CNNClassifier(vocab_size, num_classes, args.embed_size,
                          args.num_filters, filter_sizes, args.dropout).to(device)

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_acc = 0

    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss, train_acc = train([corpus.train, genset], args.tau,
                                          model, criterion, optimizer, device)
            valid_loss, valid_acc = evaluate(corpus.valid, model, criterion, device)
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
        print('Exiting from training early')

    with open(get_savepath(args), 'rb') as f:
        model = torch.load(f)

    test_loss, test_acc = evaluate(corpus.test, model, criterion, device)
    print('=' * 80)
    print("| End of training | test loss {:5.2f} | test acc {:4.1f}".format(
        test_loss, test_acc * 100))
    print('=' * 80)


if __name__ == '__main__':
    main(args)
            
