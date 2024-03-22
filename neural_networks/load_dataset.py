import sys
import os
import glob
import zipfile

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CImgDataset(Dataset):
    def __init__(self, zip_name):
        self.zip_name = zip_name
        self.zfile = zipfile.ZipFile(
            zip_name, mode="r", compression=zipfile.ZIP_DEFLATED
        )
        self.images = [x for x in self.zfile.namelist() if x.endswith(".png")]
        self.transforms = torch.nn.Sequential(
            transforms.RandomInvert(),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(degrees=(0, 180), expand=False),
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images[idx]
        with self.zfile.open(img_name, mode="r") as f:
            img = Image.open(f)
            img = np.asarray(img) / 255.0
        # img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        is_corner = 1 if "_valid_" in img_name else 0

        return self.transforms(img).float(), torch.Tensor(is_corner).float()


class CImgDataLoader(DataLoader):
    pass


def main():
    ds = CImgDataset("./test.zip")
    loader = CImgDataLoader(ds, batch_size=10, shuffle=True)

    for sample in loader:
        x, y = sample
        print(x)
        print(y)
        break


if __name__ == "__main__":
    main()
