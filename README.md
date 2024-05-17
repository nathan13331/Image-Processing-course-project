Here's a README file for the scripts `training_model`, `gen_adversarial_images`, and `gen_transformed_images`. This README will provide an overview of each script, how to use them, and their respective command-line arguments.
# CIFAR-10 ResNet Training and Adversarial Example Generation

This repository contains scripts for training ResNet models on the CIFAR-10 dataset, generating adversarial examples, and applying various image transformations to test the model robustness.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- tqdm

## Scripts

### 1. Training ResNet Models (`training_model.py`)

This script trains a ResNet model on the CIFAR-10 dataset and saves the trained model.

#### Usage

```bash
python training_model.py --model <model_name>
```

#### Arguments

- `--model`: Specify which ResNet model to use. Options are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Default is `resnet18`.

#### Example

```bash
python training_model.py --model resnet50
```

### 2. Generating Adversarial Examples (`gen_adversarial_images.py`)

This script generates adversarial examples for a trained ResNet model using the Fast Gradient Sign Method (FGSM).

#### Usage

```bash
python gen_adversarial_images.py --model <model_name>
```

#### Arguments

- `--model`: Specify which ResNet model to use. Options are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Default is `resnet18`.

#### Example

```bash
python gen_adversarial_images.py --model resnet50
```

### 3. Generating Transformed Images (`gen_transformed_images.py`)

This script applies specified image transformations to the CIFAR-10 test set and saves the transformed images.

#### Usage

```bash
python gen_transformed_images.py --model <model_name> --image_transform <transformation>
```

#### Arguments

- `--model`: Specify which ResNet model to use. Options are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Default is `resnet18`.
- `--image_transform`: Specify which image transformation to apply (e.g., blur, noise). This argument is optional.

#### Example

```bash
python gen_transformed_images.py --model resnet50 --image_transform blur
```


### 4. Testing Transform Accuracy (`compute_transform_accuracy.py`)

This script tests the accuracy of a ResNet model on the CIFAR-10 dataset with specific image transformations applied.

#### Usage

```bash
python compute_transform_accuracy.py --model <model_name> --image_transform <transformation>
```

#### Arguments

- `--model`: Specify which ResNet model to use. Options are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Default is `resnet18`.
- `--image_transform`: Specify which image transformation to apply (e.g., blur, noise).

#### Example

```bash
python compute_transform_accuracy.py --model resnet50 --image_transform blur
```

### 5. Testing Adversarial Accuracy (`compute_adversarial_accuracy.py`)

This script tests the accuracy of a ResNet model on the CIFAR-10 dataset with adversarial perturbations.

#### Usage

```bash
python compute_adversarial_accuracy.py --model <model_name>
```

#### Arguments

- `--model`: Specify which ResNet model to use. Options are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Default is `resnet18`.

#### Example

```bash
python compute_adversarial_accuracy.py --model resnet50
```

## Directory Structure
- 'src/': Contains the scripts for training, generating adversarial examples, and applying image transformations.
- `lib/`: Contains the necessary libraries and modules for training, loading ResNet models, and generating adversarial or transformed images.
- `pretrained_model/`: Directory where the trained models will be saved.
- `pic/`: Directory containing the CIFAR-10 dataset and generated images.


## Notes

- Ensure the CIFAR-10 dataset is available in the `../pic/` directory before running the scripts.
- The `lib` directory should contain the necessary modules for training
