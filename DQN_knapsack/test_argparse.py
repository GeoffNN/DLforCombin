import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr_multiplier', type=float)
parser.add_argument('--exp_name', type=str)
parser.add_argument('--boltzmann_exploration', action='store_true')
parser.add_argument('--batch_size', '-b', type=int, default=1000)
parser.add_argument('--n_layers', '-l', type=int, default=1)
parser.add_argument('--size', '-s', type=int, default=32)
args = parser.parse_args()

print(args.exp_name)
print(args.lr_multiplier)
print(args.boltzmann_exploration)