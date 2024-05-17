import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.abspath(".."))
from lib import dataset
from lib.resnet import load_resnet
from lib.train import test, train
from lib.fgsm_attack import gen_perturbation_file


def parse_args():
    parser = argparse.ArgumentParser(description='Test a ResNet model on CIFAR10 dataset '
                                                 'with adversarial perturbations.')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Specify which ResNet model to use.')

    return parser.parse_args()


def test_adversarial_accuracy(args):
    resnet_model = args
    # Load the CIFAR10 dataset
    train_set = dataset.CIFAR10(f'../pic/train')
    test_set = dataset.CIFAR10(f'../pic/test')
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)

    # Load the original model and train it if not trained yet
    model = load_resnet(resnet_model)
    pretrained_model_path = f'../pretrained_model/{resnet_model}_original_CIFAR10.pth'
    if not os.path.exists(pretrained_model_path):
        print('Starting training')
        train(train_loader, test_loader, model)
        print('Training complete')
        torch.save(model.state_dict(), pretrained_model_path)
        print(f'Model saved to {resnet_model}_original_CIFAR10.pth')
    else:
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()
        print(f'Model loaded from {resnet_model}_original_CIFAR10.pth')

    # Generate perturbed images if not generated yet
    output_folder = f'../pic/{resnet_model}_per_test'
    if not os.path.exists(output_folder):
        print(f'Perturbed test set not found at {output_folder}')
        print('Generating perturbed images')
        os.makedirs(output_folder)
        test_loader_1 = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        gen_perturbation_file(model, test_loader_1, output_folder)
        print(f'Saved perturbed images to {output_folder}')

    # Load the perturbed datasets
    per_test_set = dataset.CIFAR10(f'../pic/{resnet_model}_per_test')
    per_test_loader = DataLoader(per_test_set, batch_size=128, shuffle=False, num_workers=0)
    print(f'Perturbed test set loaded from ../pic/{resnet_model}_per_test')

    # Test the model on the original test set and perturbed test set
    print('Start testing')
    original_accuracy = test(model, test_loader)
    perturbed_accuracy = test(model, per_test_loader)

    # Compare the accuracies
    print(f'{resnet_model}, Original Test Accuracy: {original_accuracy:.2f}%')
    print(f'{resnet_model}, Perturbed Test Accuracy: {perturbed_accuracy:.2f}%')


# run:
if __name__ == '__main__':
    args = parse_args()
    test_adversarial_accuracy(args.model)
