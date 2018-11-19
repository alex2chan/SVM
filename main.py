import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# For debugging, will be removed in the final version:
import pdb


'''
TO DO:

(Maybe later on) Putting functions into separate files
Fix training in batches and displaying

'''


class PSD_Dataset(Dataset):
    """ A class to convert the input data into tensors
        and store them in a Dataset object so that it
        can be read later by a Dataloader function.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['pulses'])

    def __getitem__(self, index):
        pulses = torch.FloatTensor(self.data['pulses'][index])
        classes = torch.tensor(self.data['class'][index])
        dataset = {'pulses': pulses, 'class': classes}

        return dataset


class LinearSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(240, 1)

    def forward(self, x):
        h = self.fc(x)
        return h


def get_data():
    # Gamma pulses
    gamma_loose = 'forML_gamma_LOOSE.txt'
    gamma_tight = 'forML_gamma_TIGHT.txt'
    gamma_supertight = 'forML_gamma_SUPERTIGHT.txt'

    # Neutron pulses
    neutron_loose = 'forML_neutron_LOOSE.txt'
    neutron_tight = 'forML_neutron_TIGHT.txt'
    neutron_supertight = 'forML_neutron_SUPERTIGHT.txt'

    # Training Dataset
    input_pulses_fname = [gamma_loose, gamma_tight, neutron_loose, neutron_tight]
    input_pulses = []
    input_class = []
    for fname in input_pulses_fname:
        input_pulses += read_data(fname, args.tnum)['pulses']
        # input_pulses = (input_pulses - input_pulses.mean()) / input_pulses.std()
        input_class += read_data(fname, args.tnum)['class']
        input_training_data = {'pulses': Normalization(input_pulses), 'class': input_class}

    # Validation Dataset
    validation_pulses_fname = [gamma_supertight, neutron_supertight]
    validation_pulses = []
    validation_class = []
    for fname in validation_pulses_fname:
        validation_pulses += read_data(fname, args.vnum)['pulses']
        # validation_pulses = (validation_pulses - validation_pulses.mean()) / validation_pulses.std()
        validation_class += read_data(fname, args.vnum)['class']
        validation_training_data = {'pulses': Normalization(validation_pulses), 'class': validation_class}

    # Datasets and Loaders ready for training
    train_dataset = PSD_Dataset(input_training_data)
    validation_dataset = PSD_Dataset(validation_training_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.tbs, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=args.vbs, shuffle=False)

    return train_loader, validation_loader


def Normalization(data):
    # Input data is list type
    pulse = np.asarray(data)
    pulse_mean = np.mean(pulse, axis=1)
    pulse_std = np.std(pulse, axis=1)
    data_normalized = (((pulse.transpose() - pulse_mean) / pulse_std).transpose()).tolist()
    return data_normalized


def read_data(filename, limit):
    with open(filename, 'r') as file:
        # Reading and double checking all lines to have 240 elements
        data_list = [list(map(int, lines.split())) for i, lines in enumerate(file.readlines()) if len(lines.split()) == 240 and i < limit]
        # The above single line is equivalent to the following:
        # data_list = []
        # for i, lines in enumerate(file.readlines()):
        #     if len(lines.split()) == 240 and i < limit:
        #         data_list.append(list(map(int, lines.split)))

        # Classifying neutrons into '1' and gammas into '-1'
        if 'neutron' in filename:
            data_dict = {'pulses': data_list, 'class': np.ones(len(data_list)).tolist()}
        elif 'gamma' in filename:
            data_dict = {'pulses': data_list, 'class': (-1 * np.ones(len(data_list))).tolist()}

    return data_dict


def train(model, data_loader, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epoch):
        sum_loss = 0
        for i, data in enumerate(data_loader):

            input_pulses = data['pulses']
            classes = data['class']

            # If CUDA is available
            if torch.cuda.is_available():
                input_pulses = input_pulses.cuda()
                classes = data['class'].cuda()

            optimizer.zero_grad()
            output = model(input_pulses)

            # Loss Function: Need to understand this...
            loss = torch.mean(torch.clamp(1 - output.t() * classes, min=0))  # hinge loss
            loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
            loss.backward()

            optimizer.step()
            sum_loss += loss.data.cpu().numpy()

            # if (i + 1) % 1 == 0:
            #     print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // train_batch_size, sum_loss))

        print("Epoch:{:4d}\tloss:{}".format(epoch + 1, sum_loss / len(data['class'])))


def validation(model, data_loader, args):
    model.eval()
    output_data = []

    for i, data in enumerate(data_loader):

        input_pulses = data['pulses']
        classes = data['class']

        # If CUDA is available
        if torch.cuda.is_available():
            input_pulses = input_pulses.cuda()
            classes = data['class'].cuda()

        output = model(input_pulses)

        # Take data from output
        output_data.append(output)

        print("pulse:{} class:{} predicted_class:{}".format(i + 1, classes.data.cpu().numpy()[0], output.data.cpu().numpy()[0][0]))


def main(args):
    train_loader, validation_loader = get_data()

    # Training
    model = LinearSVM()
    if torch.cuda.is_available():
        model.cuda()

    print("Training...")
    train(model, train_loader, args)

    # Testing
    print("Validating...")
    validation(model, validation_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01, help='regularization')
    parser.add_argument("--lr", type=float, default=0.01, help='learning_rate')
    parser.add_argument("--tbs", type=int, default=10, help='training batch size')
    parser.add_argument("--vbs", type=int, default=1, help='validation batch size')
    parser.add_argument("--epoch", type=int, default=30, help='number of epochs')
    parser.add_argument("--tnum", type=int, default=100, help='number of pulses to be read for training')
    parser.add_argument("--vnum", type=int, default=100, help='number of pulses to be read for validation')
    parser.add_argument("--ve", type=int, default=30, help='validation at epoch')
    args = parser.parse_args()

    """
    Usage in command line: python main.py

    To change hyper parameters simply type the two dashes followed by the parameter you want to change for e.g.:
    To change learning rate: python main.py --lr 0.0001
    To change number of epochs: python main.py --epoch 30

    For help in the available parameters to change: python main.py --help

    """
    main(args)
