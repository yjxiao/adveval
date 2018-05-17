import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data/yahoo',
                    help="location of the data folder")
parser.add_argument('--tpc', action='store_true',
                    help="whether topics are used")
parser.add_argument('--joint', action='store_true',
                    help="use joint model, otherwise marginal model. Only used when tpc is True")
parser.add_argument('--bow', action='store_true',
                    help="add bow loss during training")
parser.add_argument('--lr', action='store_true',
                    help="use logistic regression instead of cnn classifier")
args = parser.parse_args()

gendata = '{0}samples.txt'.format('bow.' if args.bow else '')
if args.tpc:
    prefix = 'jt.' if args.joint else 'mg.'
    gendata = prefix + gendata

command_base = 'python {0}main.py --data={1} --gendata={2} --tau={3}'
dataset = args.data.rstrip('/').split('/')[-1]
logpath_base = 'logs/{0}/{1}.tau{2:.1f}.{3}.log'
lines = []
for tau in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    command = command_base.format('lr_' if args.lr else 'cnn_', args.data, gendata, tau)
    logpath = logpath_base.format(dataset, gendata, tau, 'lr' if args.lr else 'cnn')
>>>>>>> 79c3894b4261bc42860552577c05877983aaf8d6
    lines.append(command + ' > ' + logpath)

script_name = '{0}.{1}.{2}.sh'.format(dataset, gendata, 'lr' if args.lr else 'cnn')
with open(script_name, 'w') as f:
    f.write('\n'.join(lines))
    
