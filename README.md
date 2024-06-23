# CIFAR-10 ResNet Training and Adversarial Example Generation

This repository contains scripts for training ResNet models on the CIFAR-10 dataset, generating adversarial examples, and applying various image transformations to test the model robustness.

------ This project uses only CPU since my laptop does not have a GPU for demo purposes and CUDA is not available on MacOS. ------

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
├── venv/
└── ...
```


## Environment Setup

To setup the environment, follow the steps below:

1. Clone the repository

```bash
git clone https://github.com/nathan13331/Image-Processing-course-project.git
```

2. Install the required packages
Note that the project uses Python 3.9. If you are using a different version of Python, make sure to install the required packages accordingly.

```bash
# Create a virtual environment if you don't have one yet
# python3 -m venv venv
# source venv/bin/activate
cd Image-Processing-course-project
python.exe -m pip install --upgrade pip
pip3 install -r requirements.txt
```

3. Install PyTorch and torchvision for CPU.
For MacOS users, torch and torchvision can be installed using the following command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For windows users, you can install the required packages using the following command:

```bash
pip3 install torch torchvision torchaudio
```

Note that Pytorch and torchvision used to have different versions for different operating systems. Make sure to install the correct version for your operating system.

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
python3 training_model.py --model '<model_name>' --epochs '<num_epochs>' --lr '<learning_rate>'
```

#### Arguments

- `--model`: Specify which ResNet model to use. Options are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Default is `resnet18`.
- `--epochs`: Specify the number of epochs to train the model. Default is `10`.

#### Example

```bash
python3 training_model.py --model 'resnet18' --epochs 200
```

### 2. Testing Model's on Test Set (`test_model.py`)

This script tests the accuracy of a trained ResNet model on the CIFAR-10 dataset.

#### Usage

```bash
python3 test_model.py --model '<model_name>'
```

### 3. Generating Adversarial Examples (`gen_adversarial_images.py`)

This script generates adversarial examples for a trained ResNet model and dataset using the Fast Gradient Sign Method (FGSM).

#### Usage

```bash
python3 gen_adversarial_images.py --model '<model_name>'
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
python3 compute_adversarial_accuracy.py --model '<model_name>'
```

### 5. Generating Transformed Images (`gen_transformed_images.py`)

This script applies specified image transformations to the CIFAR-10 test set and saves the transformed images.

The transformed images are saved in the `pic/adversarial_transformed_images/{resnet_model_name}` directory.

#### Usage

```bash
python3 gen_transform_images.py --model '<model_name>' --image_transform '<transformation>'
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
python3 gen_transform_images.py --model 'resnet18' --image_transform 'bit_quantization_8'
```

### 6. Testing Model on Transformed Images (`compute_transform_accuracy.py`)

This script tests the accuracy of a ResNet model on the CIFAR-10 dataset with specific image transformations applied.

#### Usage

```bash
python3 compute_transform_accuracy.py --model '<model_name>' --image_transform '<transformation>'
```
