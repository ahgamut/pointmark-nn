import sys
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torchvision

from load_dataset import CImgDataset, CImgDataLoader


class SimpleModel(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.l1 = nn.Linear(np.prod(in_shape), 1)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x)
        x = self.ac(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    model = SimpleModel(in_shape=(25, 25))
    ds = CImgDataset("./example.zip")
    loader = CImgDataLoader(ds, batch_size=10, shuffle=True)
    loss_fn = nn.MSELoss()

    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    stime = time.time()
    for t in range(epochs):
        ctime = time.time() - stime
        print(f"{ctime:3.4f} Epoch {t+1}\n-------------------------------")
        train_loop(loader, model, loss_fn, optimizer)


if __name__ == "__main__":
    main()
