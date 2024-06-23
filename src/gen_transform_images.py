import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(".."))
from lib import dataset
from lib.resnet import load_resnet
from lib.image_transformations import gen_transformed_file


def parse_args():
    parser = argparse.ArgumentParser(description='Generate transformed images for adversarial attacks')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')
    parser.add_argument('--image_transform', type=str, default='normalized',
                        help='Specify which image transformation to apply (e.g., blur, noise).')

    return parser.parse_args()


def gen_transformed_images(args):
    resnet_model = args.model
    image_transform = args.image_transform

    # Load the saved model
    model = load_resnet(resnet_model)
    pretrained_model_path = f'../pretrained_model/{resnet_model}/saved_model.pth'
    if not os.path.exists(pretrained_model_path):
        print(f'Model not found at {pretrained_model_path}')
        return 0
    else:
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()
        print(f'Model loaded from {resnet_model}/saved_model.pth')

    # Check if the perturbed test set is generated
    output_folder = f'../pic/adversarial_images/{resnet_model}/'
    if not os.path.exists(output_folder):
        print(f'Perturbed test set not found at {output_folder}')
        return 0
    else:
        print(f'Perturbed test set found at {output_folder}')

    # Generate transformed perturbed images if not generated yet
    output_folder = f'../pic/adversarial_transformed_images/{resnet_model}/'
    if not os.path.exists(output_folder):
        print(f'{image_transform} test set not found at {output_folder}')
        print(f'Generating {image_transform} images')
        os.makedirs(output_folder)
        per_test_set = dataset.CIFAR10(f'../pic/adversarial_images/{resnet_model}/')
        per_test_loader = DataLoader(per_test_set, batch_size=1, shuffle=False, num_workers=0)
        gen_transformed_file(per_test_loader, output_folder, image_transform)
        print(f'Saved {image_transform} images to {output_folder}')
    else:
        print(f'{image_transform} test set found at {output_folder}')
        print(f'delete {output_folder} to generate new {image_transform} images')


if __name__ == '__main__':
    args = parse_args()
    gen_transformed_images(args)
