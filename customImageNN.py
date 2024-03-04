# Imports
import torch
import torch.nn as nn  # nn stuff and loss
import torch.optim as optim  # optimization
import torch.nn.functional as F  # relu, tanh, functions with no params (nn also has)
from torch.utils.data import DataLoader  # helps create mini-batches of data to train on
import torchvision.datasets as datasets  # provides standard datasets from Pytorch
import torchvision.transforms as transforms  # helpful transforms
from customImageSet import CustomImageDataset
import zipfile

# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # input size 784 since 28x28 images
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 46)  # 2 layers, 46 nodes, col of mat2
        self.fc2 = nn.Linear(46, num_classes)  # 12 is num of examples to run, can control mat1 and mat2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 625  # row of mat2
num_classes = 10
learning_rate = 0.001
batch_size = 12  # controls row of map1 if correct size or less
num_epochs = 1

# Load data
# Since going to load as image, convert to tensor
dataset = CustomImageDataset(root_dir='test/data', transform=transforms.ToTensor())

train_num = int(dataset.dataCount - dataset.dataCount / 2)
test_num = dataset.dataCount - train_num - 1
train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])  #train_num is row of map1
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):  # one epoch = network has seen all images in dataset
    for batch_idx, (data, targets) in enumerate(train_loader):

        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # get to correct shape
        # print(data.shape)  # torch.Size([# of images, 1 input channel (RGB is 3) since black/white, height, width])
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()  # set gradients to 0 for each batch to not store backprop calc from previous forwards
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check training accuracy
def check_accuracy(loader, model, is_training):
    if is_training:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)

            #  Shape of scores is 64 (originally) images * 10
            #  Want to know which one is the maximum of those 10 digits.
            #  I.e. if max value is first one, digit 0.
            _, predictions = scores.max(1)  # max of second dimension
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)  # size of first dimension

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    #  return float(num_correct)/float(num_samples)*100

check_accuracy(train_loader, model, True)
check_accuracy(test_loader, model, False)
