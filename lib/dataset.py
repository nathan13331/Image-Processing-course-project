import os
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset


class CIFAR10(Dataset):
    def __init__(self, path):
        self.data = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        for label in os.listdir(path):
            for pic in os.listdir(path + '\\' + label):
                cv_pic = cv2.imread(os.path.join(path, label, pic))
                self.data.append([cv_pic, int(label)])

    def __getitem__(self, index):
        datas = self.transform(self.data[index][0])
        labels = torch.tensor(self.data[index][1])

        return datas, labels

    def __len__(self):
        return len(self.data)
