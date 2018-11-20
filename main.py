import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

# For debugging, will be removed in the final version:
import pdb


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


def get_data(args):
    # Gamma pulses
    gamma_loose = 'forML_gamma_LOOSE.txt'
    gamma_tight = 'forML_gamma_TIGHT.txt'
    gamma_supertight = 'forML_gamma_SUPERTIGHT.txt'

    # Neutron pulses
    neutron_loose = 'forML_neutron_LOOSE.txt'
    neutron_tight = 'forML_neutron_TIGHT.txt'
    neutron_supertight = 'forML_neutron_SUPERTIGHT.txt'

    # Training Dataset
    input_pulses_fname = [gamma_supertight, neutron_supertight]
    print("Training Database: {}".format(input_pulses_fname))
    input_pulses = []
    input_class = []
    for fname in input_pulses_fname:
        input_pulses += read_data(fname, int(args.tnum / len(input_pulses_fname)))['pulses']
        input_class += read_data(fname, int(args.tnum / len(input_pulses_fname)))['class']
        input_training_data = {'pulses': Normalization(input_pulses), 'class': input_class}

    # Validation Dataset
    validation_pulses_fname = [gamma_loose, gamma_tight, neutron_loose, neutron_tight]
    print("Validation Database: {}".format(validation_pulses_fname))
    validation_pulses = []
    validation_class = []
    for fname in validation_pulses_fname:
        validation_pulses += read_data(fname, int(args.vnum / len(validation_pulses_fname)))['pulses']
        validation_class += read_data(fname, int(args.vnum / len(validation_pulses_fname)))['class']
        validation_training_data = {'pulses': Normalization(validation_pulses), 'class': validation_class}

    # Datasets and Loaders ready for training
    train_dataset = PSD_Dataset(input_training_data)
    validation_dataset = PSD_Dataset(validation_training_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.tbs, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=args.vbs, shuffle=False)

    print_args(args)

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
    print("CNN Model: {}".format(model.__class__.__name__))
    print("Optimizer: {}".format(optimizer.__class__.__name__))

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

            # Loss Function:
            loss = torch.mean(torch.clamp(1 - output.t() * classes, min=0))  # hinge loss
            loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
            loss.backward()

            optimizer.step()
            sum_loss += loss.data.cpu().numpy()

        print("Epoch [{}/{}], Loss: {}".format(epoch + 1, args.epoch, sum_loss / args.tnum))


def validation(model, data_loader, args):
    model.eval()
    correct_predictions = args.vnum
    actual_predictions_clamped = []

    for i, data in enumerate(data_loader):

        input_pulses = data['pulses']
        classes = data['class']

        # If CUDA is available
        if torch.cuda.is_available():
            input_pulses = input_pulses.cuda()
            classes = data['class'].cuda()

        output = model(input_pulses)

        # Comparing output data and class
        output_data = output.data.cpu().numpy()[0][0]
        classes_data = classes.data.cpu().numpy()[0]

        output_data_class = np.sign(output_data)
        if output_data_class != classes_data:
            correct_predictions -= 1

        confidence = 1
        if output_data > -1 and output_data < 1:
            confidence = np.absolute(output_data)

        actual_predictions_clamped.append(np.clip(output_data, -1, 1))

        print("pulse:{:>4}| class:{:>4}| predicted class:{:>4}| actual prediction:{:>+.4f}| confidence:{:>.2%}"
              .format(i + 1, classes_data, output_data_class, output_data, confidence))

    print("\nAccuracy for {} pulses: {:.2%}".format(args.vnum, correct_predictions / args.vnum))
    return np.asarray(actual_predictions_clamped)


def plot(confidence_levels):
    # x = np.linspace(-1, 1, len(confidence_levels))
    # plt.plot(x, norm.pdf(confidence_levels), 'r-', lw=2, alpha=0.6, label='norm pdf')
    # plt.hist(confidence_levels, 100, facecolor='green', alpha=0.75)
    f, axes = plt.subplots(1, 2)
    f.set_size_inches(9, 6)

    sns.distplot(confidence_levels, hist=True, kde=False, norm_hist=False,
                 bins=1000, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 2, 'bw': 0.25}, ax=axes[0])
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Confidence")
    axes[0].set_title("Neutron = 1, Gamma = -1")

    sns.distplot(confidence_levels, hist=False, kde=True, norm_hist=False,
                 bins=1000, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 2, 'bw': 0.25}, ax=axes[1])
    axes[1].set_ylabel("Density")
    axes[1].set_xlabel("Confidence")
    axes[1].set_title("Neutron = 1, Gamma = -1")

    plt.show()


def print_args(args):
    print("Training Dataset Size: {}".format(args.tnum))
    print("Training Batch Size: {}".format(args.tbs))
    print("Validation Dataset Size: {}".format(args.vnum))
    print("Validation Batch Size: {}".format(args.vbs))
    print("No. of Epochs: {}".format(args.epoch))
    print("Learning Rate: {}".format(args.lr))
    print("Regularization: {}".format(args.c))


def main(args):
    # Data Retrieval and Loading
    start_time = time.time()
    print("Retrieving data...")
    train_loader, validation_loader = get_data(args)
    get_data_time = time.time()
    print("Data Loaded Successfully!\nData Loading Time: {:.3f} secs\n".format(get_data_time - start_time))

    # Training
    model = LinearSVM()
    if torch.cuda.is_available():
        model.cuda()

    print("Training...")
    train(model, train_loader, args)
    training_time = time.time()
    print("End of training\nTraining Time: {:.3f} secs\n".format(training_time - get_data_time))

    # Testing
    print("Validating...")
    confidence_levels = validation(model, validation_loader, args)
    validation_time = time.time()
    print("End of validation\nValidation Time: {:.3f} secs\n".format(validation_time - training_time))
    print("Total Time Elapsed: {:.3f}secs\n".format(time.time() - start_time))
    print("{0:-^31}\n".format("END"))

    # Plotting
    plot(confidence_levels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01, help='regularization')
    parser.add_argument("--lr", type=float, default=0.01, help='learning_rate')
    parser.add_argument("--tbs", type=int, default=10, help='training batch size')
    parser.add_argument("--vbs", type=int, default=1, help='validation batch size')
    parser.add_argument("--epoch", type=int, default=10, help='number of epochs')
    parser.add_argument("--tnum", type=int, default=500, help='number of pulses to be read for training')
    parser.add_argument("--vnum", type=int, default=100, help='number of pulses to be read for validation')
    parser.add_argument("--ve", type=int, default=30, help='validation at epoch')
    args = parser.parse_args()

    """
    Usage in command line: python main.py

    Usage Examples:
    To change hyper parameters simply type the two dashes followed by the parameter you want to change for e.g.:
    To change learning rate: python main.py --lr 0.0001
    To change number of epochs: python main.py --epoch 30

    To write the results to a text file: python main.py > filename.txt
    To write the results and append the same file: python main.py >> filename.txt

    For help in the available parameters to change: python main.py --help

    """
    main(args)
