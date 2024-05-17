import os
import sys
import argparse
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(".."))
from lib import dataset
from lib.image_transformations import gen_transformed_file


def parse_args():
    parser = argparse.ArgumentParser(description='Test a ResNet model on CIFAR10 dataset '
                                                 'with adversarial perturbations.')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')
    parser.add_argument('--image_transform', type=str, default=None,
                        help='Specify which image transformation to apply (e.g., blur, noise).')

    return parser.parse_args()


def gen_transformed_images(args):
    resnet_model = args.resnet_model
    image_transform = args.image_transform
    # Load the test dataset
    test_set = dataset.CIFAR10(f'../pic/test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # Create the folder to save smoothed images if it doesn't exist
    output_folder = f'../pic/{resnet_model}_{image_transform}_test'
    if os.path.exists(output_folder):
        print(f'{image_transform} test set already found at {output_folder}')
        print(f'Still generating {image_transform} images')
        gen_transformed_file(test_loader, output_folder, image_transform)
        print(f'Saved {image_transform} images to {output_folder}')
    else:
        os.makedirs(output_folder)
        print(f'Created {output_folder} to save {image_transform} test set')
        gen_transformed_file(test_loader, output_folder, image_transform)
        print(f'Saved {image_transform} images to {output_folder}')


if __name__ == '__main__':
    args = parse_args()
    gen_transformed_images(args)
