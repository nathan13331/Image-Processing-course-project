import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(".."))
from lib import dataset
from lib.resnet import load_resnet
from lib.train import test


def parse_args():
    parser = argparse.ArgumentParser(description='Compute transform accuracy on CIFAR10')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')
    parser.add_argument('--image_transform', type=str, default='normalized',
                        help='Specify which image transformation to apply.')

    return parser.parse_args()


def test_transform_accuracy(args):
    resnet_model = args.model
    image_transform = args.image_transform
    # Load the CIFAR10 dataset
    test_set = dataset.CIFAR10(f'../pic/test')
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)

    # Load the saved model
    model = load_resnet(resnet_model)
    pretrained_model_path = f'../pretrained_model/{resnet_model}/original_CIFAR10.pth'
    if not os.path.exists(pretrained_model_path):
        print(f'Model not found at {pretrained_model_path}')
        return 0
    else:
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()
        print(f'Model loaded from {resnet_model}/original_CIFAR10.pth')

    # Check if the perturbed test set is generated
    output_folder = f'../pic/adversarial_images/{resnet_model}'
    if not os.path.exists(output_folder):
        print(f'Perturbed test set not found at {output_folder}')
        return 0
    else:
        print(f'Perturbed test set found at {output_folder}')

    # Check if the transformed test set is generated
    output_folder = f'../pic/adversarial_transformed_images/{resnet_model}'
    if not os.path.exists(output_folder):
        print(f'Transformed test set not found at {output_folder}')
        return 0
    else:
         print(f'Transformed test set found at {output_folder}')

    # Load the perturbed image dataset and transform perturbed images dataset
    per_test_set = dataset.CIFAR10(f'../pic/adversarial_images/{resnet_model}/')
    per_test_loader = DataLoader(per_test_set, batch_size=128, shuffle=False, num_workers=0)
    per_transformed_test_set = dataset.CIFAR10(output_folder)
    per_transformed_test_loader = DataLoader(per_transformed_test_set, batch_size=128, shuffle=False, num_workers=0)
    print(f'{image_transform} perturbed test set loaded from {output_folder}')

    # Test both model on each dataset
    print('Start testing')
    original_accuracy = test(model, test_loader)
    perturbed_accuracy = test(model, per_test_loader)
    transformed_per_accuracy = test(model, per_transformed_test_loader)

    print(f'{resnet_model}, Original Test Accuracy: {original_accuracy:.2f}%')
    print(f'{resnet_model}, Perturbed Test Accuracy: {perturbed_accuracy:.2f}%')
    print(f'{resnet_model}, Transformed Test Accuracy: {transformed_per_accuracy:.2f}%')


if __name__ == '__main__':
    args = parse_args()
    test_transform_accuracy(args)
