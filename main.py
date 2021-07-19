# Author: Victor Dachet
from scipy import stats
import numpy as np
import pandas as pd
import argparse
import time
import os

import util as u
from model import *

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epoch', help='Number of epoch',
                        type=int, default=10)
    parser.add_argument('-batch', '--batch_size', help='Batch size', type=int,
                        default=32)
    parser.add_argument('-lr', '--learning_rate', help='Actor learning rate', 
                        type=float, default=1e-4)
    parser.add_argument('--hidden_size', help='Size of hidden layers', type=int,
                        default=64)
    parser.add_argument('--num_layers', help='Number of layers in the RNN', 
                        type=int, default=1)
    parser.add_argument('-d','--dataset_creation', help='Create the dataset', 
                        action="store_true")
    parser.add_argument('-t','--training', help='Train the model', 
                        action="store_true")
    parser.add_argument('--rnn', help='RNN type: LSTM GRU BRC nBRC', type=str)
    parser.add_argument('-c_t','--continue_training',
        help='Continue the training. Requires the path of the model to train', 
        required=False, default=None, type=str)

    args = parser.parse_args()

    rnns = {"LSTM":LSTM, "GRU":GRU, "BRC":BRC, "nBRC":nBRC}

    dataset_creation = args.dataset_creation
    model_training = args.training
    data = None

    if dataset_creation:
        u.create_dataset(vervose=False)
        df = pd.read_csv("data/dataset.csv")
        data = u.get_random_split_dataset(df)
        u.write_split_dataset(data)


    if model_training:
        device = torch.device('cpu')
        batch_size = args.batch_size
        checkpoint = 2
        
        # Data Loading
        if not data:
            data = u.load_split_dataset()

        X_train = torch.Tensor(data["X_train"])
        y_train = torch.Tensor(data["y_train"])
        train_set_len = X_train.shape[0]

        X_valid = torch.Tensor(data["X_valid"])
        y_valid = torch.Tensor(data["y_valid"])
        val_set_len = X_valid.shape[0]

        train = TensorDataset(X_train, y_train)
        val = TensorDataset(X_valid, y_valid)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

        # Model Parameters
        input_size = X_train.shape[2] # number of expected features in the input x
        hidden_size = args.hidden_size # Num of units in the RNN
        num_layers = args.num_layers # Number of recurrent layers
        output_size = y_train.shape[1]
        seq_length = X_train.shape[1]

        rnn = rnns[args.rnn]

        
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
                     seq_length=seq_length, output_size=output_size)

        if args.continue_training:
            print("We load the model to continue the training...")
            model.load_state_dict(torch.load(args.continue_training, map_location=torch.device(device)), strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        model.to(device)

        losses = []
        with open('results.csv', 'w') as f:
            f.write('epoch,train_loss,valid_loss,time\n')

        # Training Loop
        for e in range(args.epoch):
            start = time.time()
            mean_loss = 0.

            for x_batch, y_batch in train_loader:
                # print(x_batch.shape)
                # print(y_batch.shape)
                output = model(x_batch)
                
                # print(output.shape)
                loss = F.mse_loss(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                mean_loss += loss.item()

                # Checkpoint
                if e % checkpoint == 0 or e == args.epoch - 1:
                    if not os.path.isdir('model'):
                        os.mkdir("model")
                    if not os.path.isdir("model/" + args.rnn):
                        os.mkdir("model/" + args.rnn)
                    torch.save(model.state_dict(), 
                               f"model/" + args.rnn + f"/{e + 1}.model")

                break
            mean_loss = mean_loss / train_set_len
            losses.append(mean_loss)

            mean_loss_valid = 0.
            # Validation Loss
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output = model(x_batch)
                    loss_valid = F.mse_loss(output, y_batch)
                    mean_loss_valid += loss_valid
                    break
            mean_loss_valid = mean_loss_valid / val_set_len
            
            duration = time.time() - start

            if not os.path.isdir('results'):
                os.mkdir("results")

            with open('results/' + args.rnn + '.csv', 'a') as f:
                f.write('{},{},{},{}\n'.format(e + 1, mean_loss, mean_loss_valid, duration))
            print("Epoch {} MSE Train Loss: {:.4f} MSE Valid Loss: {:.4f} Duration: {:.2f}".format(e + 1, 
                mean_loss, mean_loss_valid, duration))


