from torch.utils.data import TensorDataset
from torchvision.transforms import transforms
from PIL import Image
import glob
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class Dataset(TensorDataset):
    def __init__(self, data_path, same_prob = 0.8):
        self.same_prob = same_prob

        self.datasets = glob.glob(f'{data_path}/*.*g')

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5 ] )
        ])

    def __getitem__(self, item):

        image_path = self.datasets[item]
        name = os.path.split(image_path)[1]

        Xs = cv2.imread(image_path)
        cv2.imwrite("../outputs/sample_out.jpg", Xs)
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets)
            Xt = cv2.imread(image_path)
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1

        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return len(self.datasets)


class Dataset_Sample(TensorDataset):
    def __init__(self, data_path):

        self.datasets = glob.glob(f'{data_path}/*_s.*g')
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5 ] )
        ])

    def __getitem__(self, item):

        image_path = self.datasets[item]
        name = os.path.split(image_path)[1]

        Xs = cv2.imread(image_path)
        Xs = Image.fromarray(Xs)
        

        image_path =image_path.split("\\")[0] + "\\"+ image_path.split("\\")[-1].split("_s")[0] + "_t."+ image_path.split("\\")[-1].split(".")[-1]
        Xt = cv2.imread(image_path)
        Xt = Image.fromarray(Xt)
        same_person = 0

        return self.transforms(Xs), self.transforms(Xt), same_person, image_path.split("\\")[-1].split("_t")[0]

    def __len__(self):
        return len(self.datasets)


if __name__ == "__main__":
    dataset = Dataset_Sample("../inputs/processed")
    Xs,Xt, _, _ = dataset[0]
    print(Xs)