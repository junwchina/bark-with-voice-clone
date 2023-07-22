from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--mode', required=True, help='The mode to use', choices=['prepare', 'prepare2', 'train', 'test'])
parser.add_argument('--train-save-epochs', default=1, type=int, help='The amount of epochs to train before saving')

args = parser.parse_args()