import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 10 * 13 + 64 * 10 * 13 * 20, 128)
        self.fc2 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.pool(torch.relu(self.conv1(x1)))
        x1 = self.pool(torch.relu(self.conv2(x1)))
        x1 = x1.view(-1, 64 * 10 * 13)
        x2 = [self.pool(torch.relu(self.conv1(x))) for x in x2]
        x2 = [self.pool(torch.relu(self.conv2(x))) for x in x2]
        x2 = [x.view(-1, 64 * 10 * 13) for x in x2]
        x2 = torch.cat(x2, dim=1)
        x = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
