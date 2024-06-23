# CIFAR-10 ResNet Training and Adversarial Example Generation

This repository contains scripts for training ResNet models on the CIFAR-10 dataset, generating adversarial examples, and applying various image transformations to test the model robustness.

------ This project uses only CPU since my laptop does not have a GPU for demo purposes. ------

## Directory Structure
- `src/`: Contains the scripts for training, generating adversarial examples, and applying image transformations.
- `lib/`: Contains the necessary libraries and modules for training, loading ResNet models, and generating adversarial or transformed images.
- `pretrained_model/`: Directory where the trained models will be saved.
- `pic/`: Directory containing the CIFAR-10 dataset and generated images.
- `datasets/`: Directory containing the CIFAR-10 dataset in binary format.
- `requirements.txt`: Contains the required packages and their versions.

```bash
├── README.md
├── src/
│   ├── __init__.py
│   ├── training_model.py
│   ├── compute_original_accuracy.py
│   ├── gen_adversarial_images_dataset.py
│   ├── compute_adversarial_accuracy.py
│   ├── gen_transform_images_dataset.py
│   └── compute_transform_accuracy.py
├── lib/
│   ├── __init__.py
│   ├── dataset.py
│   ├── resnet.py
│   ├── train.py
│   ├── fgsm_attack.py
│   └── image_transformations.py
├── pretrained_model/
├── pic/
│   ├── test/
│   ├── train/
│   ├── adversarial_images/
│   └── adversarial_transformed_images/
├── datasets/
├── requirements.txt
└── ...
```


## Environment Setup

To setup the environment, follow the steps below:

1. Clone the repository

```bash
git clone https://github.com/nathan13331/adversarial-examples-cifar10.git
```

2. Install the required packages

```bash
cd adversarial-examples-cifar10
pip install -r requirements.txt
```

## Using CIFAR-10 Dataset

To use the CIFAR-10 dataset, follow the steps below:

1. Download the CIFAR-10 dataset

The CIFAR-10 dataset is preprocessed by dividing it into training and test sets. The training set is used to train the ResNet models, and the test set is used to evaluate the model's performance on adversarial examples and transformed images.

Download the CIFAR-10 dataset from the official website https://www.cs.toronto.edu/~kriz/cifar.html .For this project, choose the `CIFAR-10 python version` and extract the file.Then, move all ".bin" files(such as `data_batch_1.bin`, `data_batch_2.bin`,...) into the `datasets/` directory.

```bash
├── datasets/
│   ├── data_batch_1.bin
│   ├── data_batch_2.bin
│   ├── ...
│   └── test_batch.bin
└── ...
```
2. Preprocess the CIFAR-10 dataset

The ".bin" files needs to be preprocessed before training the ResNet models. This is done by executing the script 'unpickle_CIFAR10.py' provided in the repository.This script extracts the images and labels from the CIFAR-10 dataset and saves them in the 'pic/test' and 'pic/train' directories.The file's name from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 represents the label of the image.

```bash
python unpickle_CIFAR10.py
```
```bash
pic/
├── test/
│   ├── 0/
│   ├── 1/
│   ├── 2/    
│   ├── 3/
│   ├──...
│   └── 9/
└── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/    
│   ├── 3/
│   ├──...
│   └── 9/
└── ...
```

## To costumize the dataset

To customize the dataset, you can save them in the 'pic/test' and 'pic/train' directories with the same structure as the CIFAR-10 dataset. The file's name should represent the label of the image (must be a integer from 0 to infinite). It is suitable for any numbers of images and any size of images.

```bash
pic/
├── test/
│   ├── 0/
│   ├── 1/
│   ├── 2/    
│   ├── 3/
│   ├──...
│   └── infinite/
└── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/    
│   ├── 3/
│   ├──...
│   └── infinite/
└── ...
```


## Scripts

### 1. Training ResNet Models (`training_model.py`)

This script trains a ResNet model on the CIFAR-10 dataset and saves the trained model.

#### Usage

```bash
python training_model.py --model '<model_name>' --epochs '<num_epochs>' --lr '<learning_rate>'
```

#### Arguments

- `--model`: Specify which ResNet model to use. Options are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Default is `resnet18`.
- `--epochs`: Specify the number of epochs to train the model. Default is `10`.
- `--lr`: Specify the learning rate for the model. Default is `0.2`.

#### Example

```bash
python training_model.py --model 'resnet18' --epochs 200 --lr 0.2
```

### 2. Testing Model's on Test Set (`test_model.py`)

This script tests the accuracy of a trained ResNet model on the CIFAR-10 dataset.

#### Usage

```bash
python test_model.py --model '<model_name>'
```

### 3. Generating Adversarial Examples (`gen_adversarial_images.py`)

This script generates adversarial examples for a trained ResNet model and dataset using the Fast Gradient Sign Method (FGSM).

#### Usage

```bash
python gen_adversarial_images.py --model '<model_name>'
```
#### Directory Structure
The adversarial examples are saved in the `pic/adversarial_images/{resnet_model_name}` directory.
```bash
pic/
├── adversarial_images/
│   ├── resnet18/
│   ├── resnet34/
│   ├── ...
│   └── resnet152/
└── ...
```

### 4. Testing Model on Adversarial Images (`compute_adversarial_accuracy.py`)

This script tests the accuracy of a ResNet model on the CIFAR-10 dataset with adversarial perturbations.

#### Usage

```bash
python compute_adversarial_accuracy.py --model '<model_name>'
```

### 5. Generating Transformed Images (`gen_transformed_images.py`)

This script applies specified image transformations to the CIFAR-10 test set and saves the transformed images.

The transformed images are saved in the `pic/adversarial_transformed_images/{resnet_model_name}` directory.

#### Usage

```bash
python gen_transform_images.py --model '<model_name>' --image_transform '<transformation>'
```

#### Arguments
- `--image_transform`: Specify which image transformation to apply 

Options are: 
- `median_smooth`, 
- `color_depth_reduction_{bit_depth}`,
- `spatial_smoothing_{kernel_size}`, 
- `non_local_mean`, 
- `sharpen`, 
- `enhance_color`, 
- `enhance_color`, 
- `normalized`, 
- `bit_quantization_{bit_depth}`. 

#### Example

```bash
python gen_transform_images.py --model 'resnet18' --image_transform 'bit_quantization_8'
```

### 6. Testing Model on Transformed Images (`compute_transform_accuracy.py`)

This script tests the accuracy of a ResNet model on the CIFAR-10 dataset with specific image transformations applied.

#### Usage

```bash
python compute_transform_accuracy.py --model '<model_name>' --image_transform '<transformation>'
```
