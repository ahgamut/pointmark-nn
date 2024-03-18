import torch
import torch.nn as nn  # nn stuff and loss
import torch.optim as optim  # optimization
import torch.nn.functional as F  # relu, tanh, functions with no params (nn also has)
from torch.utils.data import DataLoader  # helps create mini-batches of data to train on
import torchvision.transforms as transforms  # helpful transforms
from customImageSet import CustomImageDataset

# GRU stands for Gated Recurrent Unit, a specialized version of a Recurrent Neural Network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  # input size 625 since 25x25 images
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
                                                            # might remove batch_first
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Propagation
        out, _ = self.rnn(x, h0)  # don't need to store hidden state
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 25  # row of mat2
sequence_length = 25
num_layers = 2
hidden_size = 231
num_classes = 2
learning_rate = 0.001  # note, for a regular RNN, change this to 0.001
batch_size = 12  # controls row of map1 if correct size or less
num_epochs = 3

# Load data
# Since going to load as image, convert to tensor
dataset = CustomImageDataset(root_dir="D:/test/data", transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])  #first is row of map1
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # could try MSELoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):  # one epoch = network has seen all images in dataset
    for batch_idx, (data, targets) in enumerate(train_loader):

        # get data to cuda if possible
        data = data.to(device=device).squeeze(1)
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)

            #  Shape of scores is 64 (originally) images * 10
            #  Want to know which one is the maximum of those 10 digits.
            #  I.e. if max value is first one, digit 0.
            _, predictions = scores.max(1)  #  max of second dimension
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)  #  size of first dimension

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    #  return float(num_correct)/float(num_samples)*100

check_accuracy(train_loader, model, True)
check_accuracy(test_loader, model, False)
