import torch
import torch.nn as nn  # nn stuff and loss
import torch.optim as optim  # optimization
import torch.nn.functional as F  # relu, tanh, functions with no params (nn also has)
from torch.utils.data import DataLoader  # helps create mini-batches of data to train on
import torchvision.transforms as transforms  # helpful transforms
from customImageSet import CustomImageDataset
from load_dataset import CImgDataset


class CNN(nn.Module):
    def __init__(
        self, input_size=1, num_classes=40
    ):  # input size 784 since 28x28 images
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_size,
            out_channels=10,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # stride and padding are standard
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=num_classes,  # out_channels here controls col of mat1
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(
            72, num_classes
        )  # fully connected layer, row of mat2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # number of examples sent in
        x = self.fc1(x)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 1  # row of mat2
num_classes = 2
learning_rate = 0.001
batch_size = 25  # controls row of map1 if correct size or less, controls how many samples are tested together,
# so lower is more accurate but slower
num_epochs = 10

# Load data
# Since going to load as image, convert to tensor
# dataset = CustomImageDataset(root_dir="D:/test/data", transform=transforms.ToTensor())
dataset = CImgDataset("../test51.zip")

train_set, test_set = torch.utils.data.random_split(
    dataset, [0.8, 0.2]
)  # first is row of map1
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN(input_size=input_size, num_classes=num_classes).to(device)
# model = CNN()
x = torch.randn(100, 1, 25, 25)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # could try MSELoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network

for epoch in range(num_epochs):  # one epoch = network has seen all images in dataset
    print(epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):

        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

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

            scores = model(x)

            #  Shape of scores is 64 (originally) images * 10
            #  Want to know which one is the maximum of those 10 digits.
            #  I.e. if max value is first one, digit 0.
            _, predictions = scores.max(1)  #  max of second dimension
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)  #  size of first dimension

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()
    #  return float(num_correct)/float(num_samples)*100


check_accuracy(train_loader, model, True)
check_accuracy(test_loader, model, False)

torch.save(model.state_dict(), "../model_weights_convolutional")
model = CNN(input_size=input_size, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("../model_weights_convolutional"))
model.eval()

check_accuracy(test_loader, model, False)
