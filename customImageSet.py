import os
import pandas as pd
import torch
from torch.utils.data import Dataset  # helps create mini-batches of data to train on
from skimage import io

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):  # root dir of images, transform optional
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  # number of images, currently 12

    def __getitem__(self, index):  # going to return specific image + corresponding target
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])  # row i, col 0
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))  # annotation without s?

        if self.transform:  # if there's a transform, apply it
            image = self.transform(image)

        return (image, y_label)  # this class loads 1 image and its corresponding target