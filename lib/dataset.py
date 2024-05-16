import os
import torch
import torchvision.transforms as transforms
from cv2 import imread
from torch.utils.data import Dataset


class CIFAR10(Dataset):
    def __init__(self, path):
        self.data = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for label in os.listdir(path):
            for pic in os.listdir(path + '\\' + label):
                cv_pic = imread(f'{path}\\{label}\\{pic}')
                self.data.append([cv_pic, int(label)])

    def __getitem__(self, index):
        datas = self.transform(self.data[index][0])
        labels = torch.tensor(self.data[index][1])
        return datas, labels

    def __len__(self):
        return len(self.data)
