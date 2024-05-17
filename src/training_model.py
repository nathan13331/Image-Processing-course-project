import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(".."))
from lib.train import train
from lib import dataset
from lib.resnet import load_resnet


def parse_args():
    parser = argparse.ArgumentParser(description='Test a ResNet model on CIFAR10 dataset '
                                                 'with adversarial perturbations.')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')
    parser.add_argument('--image_transform', type=str, default=None,
                        help='Specify which image transformation to apply (e.g., blur, noise).')

    return parser.parse_args()


def training_model(args):
    resnet_model = args.model
    # Load CIFAR10 dataset
    train_set = dataset.CIFAR10(f'../pic/train')
    test_set = dataset.CIFAR10(f'../pic/test')
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)

    # Load ResNet model
    model = load_resnet(resnet_model)

    # Train the model
    train(train_loader, test_loader, model)
    print('Training complete')

    # Save the model
    torch.save(model.state_dict(), f'../pretrained_model/{resnet_model}_original_CIFAR10.pth')
    print(f'Model saved to ../pretrained_model/{resnet_model}_original_CIFAR10.pth')


# run:
if __name__ == '__main__':
    args = parse_args()
    training_model(args.model)
