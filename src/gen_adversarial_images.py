import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(".."))
from lib import dataset
from lib.resnet import load_resnet
from lib.fgsm_attack import  gen_perturbation_file


def parse_args():
    parser = argparse.ArgumentParser(description='Test a ResNet model on CIFAR10 dataset '
                                                 'with adversarial perturbations.')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')

    return parser.parse_args()


def gen_adversarial_images(args):
    resnet_model = args
    # Load the test dataset
    test_set = dataset.CIFAR10(f'../pic/test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # Load the saved model
    model = load_resnet(resnet_model)
    model.load_state_dict(torch.load(f'../pretrained_model/{resnet_model}_original_CIFAR10.pth'))
    model.eval()

    output_folder = f'../pic/{resnet_model}_per_test'
    if os.path.exists(output_folder):
        print(f'Perturbed test set found at {output_folder}')
        print('Still generating perturbed images? (y/n)')
        ans = input()
        if ans.lower() == 'n':
            exit()
    else:
        os.makedirs(output_folder)

    print('Generating perturbed images')
    test_loader_1 = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    gen_perturbation_file(model, test_loader_1, output_folder)
    print(f'Saved perturbed images to {output_folder}')


# run:
if __name__ == '__main__':
    args = parse_args()
    gen_adversarial_images(args.model)
