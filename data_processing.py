import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from main import print_args


class PSD_Dataset(Dataset):
    """ A class to convert the input data into tensors in torch
        and store them in a Dataset object so that it
        can be read later by a torch's Dataloader function.
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


def read_data(filename, limit):
    with open(filename, 'r') as file:
        # Reading and double checking all lines to have 240 elements
        data_list = [list(map(int, lines.split())) for i, lines in enumerate(file.readlines()) if len(lines.split()) == 240 and i < limit]
        # The above single line is equivalent to the following:
        # data_list = []
        # for i, lines in enumerate(file.readlines()):
        #     if len(lines.split()) == 240 and i < limit:
        #         data_list.append(list(map(int, lines.split)))

        # Classifying neutrons into '1' and gammas into '-1' and putting them into a dictionary
        if 'neutron' in filename:
            data_dict = {'pulses': data_list, 'class': np.ones(len(data_list)).tolist()}
        elif 'gamma' in filename:
            data_dict = {'pulses': data_list, 'class': (-1 * np.ones(len(data_list))).tolist()}

    return data_dict


def Normalization(data):
    # Input data is list type -> Output data is list type
    pulse = np.asarray(data)
    pulse_mean = np.mean(pulse, axis=1)
    pulse_std = np.std(pulse, axis=1)
    data_normalized = (((pulse.transpose() - pulse_mean) / pulse_std).transpose()).tolist()
    return data_normalized


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
    training_pulses_fname = [gamma_supertight, neutron_supertight]
    print("Training Database: {}".format(training_pulses_fname))
    training_pulses = []
    training_class = []
    for fname in training_pulses_fname:
        training_pulses += read_data(fname, int(args.tnum / len(training_pulses_fname)))['pulses']
        training_class += read_data(fname, int(args.tnum / len(training_pulses_fname)))['class']
        training_data = {'pulses': Normalization(training_pulses), 'class': training_class}

    # Validation Dataset
    validation_pulses_fname = [gamma_loose, gamma_tight, neutron_loose, neutron_tight]
    print("Validation Database: {}".format(validation_pulses_fname))
    validation_pulses = []
    validation_class = []
    for fname in validation_pulses_fname:
        validation_pulses += read_data(fname, int(args.vnum / len(validation_pulses_fname)))['pulses']
        validation_class += read_data(fname, int(args.vnum / len(validation_pulses_fname)))['class']
        validation_data = {'pulses': Normalization(validation_pulses), 'class': validation_class}

    # Datasets and Loaders ready for training
    train_dataset = PSD_Dataset(training_data)
    validation_dataset = PSD_Dataset(validation_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.tbs, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=args.vbs, shuffle=False)

    # Printing Experiment Details
    print_args(args)

    return train_loader, validation_loader
