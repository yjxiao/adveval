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
args = parser.parse_args()

gendata = '{0}samples.txt'.format('bow.' if args.bow else '')
if args.tpc:
    prefix = 'jt.' if args.joint else 'mg.'
    gendata = prefix + gendata

command_base = 'python main.py --data={0} --gendata={1} --tau={2}'
dataset = args.data.rstrip('/').split('/')[-1]
logpath_base = 'logs/{0}/{1}.tau{2:.1f}.log'
lines = []
for tau in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    command = command_base.format(args.data, gendata, tau)
    logpath = logpath_base.format(dataset, gendata, tau)
    lines.append(command + ' > ' + logpath)

script_name = '{0}.{1}.sh'.format(dataset, gendata)
with open(script_name, 'w') as f:
    f.write('\n'.join(lines))
    
