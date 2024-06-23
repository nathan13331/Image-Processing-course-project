import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(""))
from lib.train import train_model as train
from lib import dataset
from lib.resnet import load_resnet


def parse_args():
    parser = argparse.ArgumentParser(description='Training ResNet on CIFAR10')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs to train the model.')

    return parser.parse_args()


def training_model(args):
    resnet_model = args.model
    epoch = args.epoch
    # Load CIFAR10 dataset
    train_set = dataset.CIFAR10(f'./pic/train')
    test_set = dataset.CIFAR10(f'./pic/test')
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)

    # Load ResNet model and load pretrained model if available
    model = load_resnet(resnet_model)
    if os.path.exists(f'./pretrained_model/{resnet_model}/saved_model.pth'):
        print(f'Finding pretrained model in ./pretrained_model/{resnet_model}/saved_model.pth, load it? (y/n)')
        choice = input()
        if choice == 'y':
            print(f'Loading pretrained model from ./pretrained_model/{resnet_model}/saved_model.pth')
            pretrained_model_path = f'./pretrained_model/{resnet_model}/saved_model.pth'
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            print('Training from scratch')
    else:
        print(f'No pretrained model found in ./pretrained_model/{resnet_model}/original_CIFAR10.pth, train from scratch')
        if not os.path.exists(f'./pretrained_model/{resnet_model}'):
            os.makedirs(f'./pretrained_model/{resnet_model}')
        print(f'Directory ./pretrained_model/{resnet_model} created')

    # Train the model
    train(train_loader, test_loader, model, criterion=nn.CrossEntropyLoss(), epochs=epoch)
    print('Training complete')

    # Save the model
    torch.save(model.state_dict(), f'./pretrained_model/{resnet_model}/saved_model.pth')
    print(f'Model saved to ./pretrained_model/{resnet_model}/saved_model.pth')


# run:
if __name__ == '__main__':
    args = parse_args()
    training_model(args)
