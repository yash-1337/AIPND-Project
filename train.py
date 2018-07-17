# Imports here
import argparse
from helper import *
from torch import nn
from torch import optim


def get_input_args():
    '''
        Get Input Arguments From Command Line.

        Available Arguments:
            --save_dir : Directory for Checkpoint
            --arch : Architecture for Model
            --learning_rate : Learning Rate for Training
            --hidden_units : Number of Hidden Units in Model Classifier
            --epochs : Number of Epochs to Train for
            --gpu : Enable GPU Acceleration
    '''

    parser = argparse.ArgumentParser(
        description='Parameter Options for Training the Neural Network')

    parser.add_argument('data_directory', action='store')
    parser.add_argument('--save_dir', action='store',
                        type=str, dest='save_dir', default='')
    parser.add_argument('--arch', action='store', type=str,
                        dest='arch', default='densenet121')
    parser.add_argument('--learning_rate', action='store',
                        type=float, dest='learning_rate', default=0.003)
    parser.add_argument('--hidden_units', action='store',
                        type=int, dest='hidden_units', default=512)
    parser.add_argument('--epochs', action='store',
                        type=int, dest='epochs', default=5)
    parser.add_argument('--gpu', action='store_true', dest='gpu')

    return parser.parse_args()


def main():
    input_args = get_input_args()

    data_dir = input_args.data_directory
    save_dir = input_args.save_dir
    arch = input_args.arch
    learning_rate = input_args.learning_rate
    hidden_units = input_args.hidden_units
    epochs = input_args.epochs
    gpu_is_enabled = input_args.gpu

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and gpu_is_enabled else "cpu")

    train_data, trainloader, validloader = LoadData(data_dir, batch_size=64)

    model, classifier_input_size, dropout = CreateModel(arch, hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    TrainNetwork(model, trainloader, train_data, validloader,
                 criterion, optimizer, epochs, device)

    SaveCheckpoint(model, arch, learning_rate, epochs, classifier_input_size, dropout,
                   hidden_units, train_data.class_to_idx, save_dir)


if __name__ == '__main__':
    main()
