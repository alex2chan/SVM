import torch.nn as nn


class LinearSVM(nn.Module):
    """ Support Vector Machine:
        1 fully connected linear layer: 240 -> 1
    """

    def __init__(self):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(240, 1)

    def forward(self, x):
        h = self.fc(x)
        return h
