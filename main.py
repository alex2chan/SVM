import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from data_processing import get_data
from model import LinearSVM

# For debugging, will be removed in the final version:
# import pdb


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
    # Data Processing
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

    If you're using Linux (displaying and writing): python main.py | tee filename.txt

    For help in the available parameters to change: python main.py --help

    """
    main(args)
