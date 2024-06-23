import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(""))
from lib import dataset
from lib.resnet import load_resnet
from lib.train import test


def parse_args():
    parser = argparse.ArgumentParser(description='Test the original accuracy of a ResNet model on CIFAR10')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')

    return parser.parse_args()


def test_original_accuracy(args):
    resnet_model = args
    # Load the CIFAR10 dataset
    test_set = dataset.CIFAR10(f'./pic/test')
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)

    # Load the saved model
    model = load_resnet(resnet_model)
    pretrained_model_path = f'./pretrained_model/{resnet_model}/saved_model.pth'
    if not os.path.exists(pretrained_model_path):
        print(f'Model not found at {pretrained_model_path}')
        return 0
    else:
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()
        print(f'Model loaded from {resnet_model}/saved_model.pth')

    # Test the model on the original test set
    print('Start testing')
    original_accuracy = test(model, test_loader)

    # Print the original test accuracy
    print(f'{resnet_model}, Original Test Accuracy: {original_accuracy:.2f}%')


# run:
if __name__ == '__main__':
    args = parse_args()
    test_original_accuracy(args.model)
