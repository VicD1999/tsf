# Author: Victor Dachet

from scipy import stats
import numpy as np
import pandas as pd
import argparse
import time
import os

import util as u
from model import *
from attention import Attention_Net

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument('-d','--dataset_creation', help='Create the dataset', 
                        action="store_true")
    parser.add_argument('--forecast_size', help='Size of the forecast window', 
                        type=int, default=60)
    parser.add_argument('--window_size', help='Size of the input window', 
                        type=int, default=120)

    # Training args
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
    parser.add_argument('-t','--training', help='Train the model', 
                        action="store_true")
    parser.add_argument('--rnn', help='RNN type: LSTM GRU BRC nBRC attn', type=str)
    parser.add_argument('-c_t','--continue_training',
        help='Continue the training. Requires the path of the model to train', 
        required=False, default=None, type=str)

    # Model Evaluation args
    parser.add_argument('-e','--evaluation', help='Eval model', 
                        type=str, default=None)

    # Model Comparison
    parser.add_argument('-c','--comparison', help='Compare the models', 
                        action="store_true")

    args = parser.parse_args()

    rnns = {"LSTM":LSTM, "GRU":GRU, "BRC":BRC, "nBRC":nBRC, 
            "attn":Attention_Net, "simple_rnn":simple_rnn}

    dataset_creation = args.dataset_creation
    model_training = args.training
    data = None

    if dataset_creation:
        u.create_dataset(vervose=False)
        df = pd.read_csv("data/dataset.csv")
        # data = u.get_random_split_dataset(df, window_size=args.window_size, 
        #                      forecast_size=args.forecast_size, add_forecast=True)
        # u.write_split_dataset(data, path="data/{}_{}.pkl".format(
        #     args.window_size, args.forecast_size))


    if model_training:
        device = torch.device('cpu')
        batch_size = args.batch_size
        checkpoint = 1
        
        # Data Loading
        if not data:
            data = u.load_split_dataset(path="data/{}_{}.pkl".format(
                args.window_size, args.forecast_size))

        X_train = torch.Tensor(data["X_train"])
        y_train = torch.Tensor(data["y_train"])
        train_set_len = X_train.shape[0]

        X_valid = torch.Tensor(data["X_valid"])
        y_valid = torch.Tensor(data["y_valid"])
        val_set_len = X_valid.shape[0]

        train = TensorDataset(X_train, y_train)
        val = TensorDataset(X_valid, y_valid)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)

        # Model Parameters
        input_size = X_train.shape[2] # number of expected features for input x
        hidden_size = args.hidden_size # Num of units in the RNN
        num_layers = args.num_layers # Number of recurrent layers
        output_size = y_train.shape[1]
        seq_length = X_train.shape[1]

        rnn = rnns[args.rnn]

        
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
                     seq_length=seq_length, output_size=output_size)

        if args.continue_training:
            print("We load the model to continue the training...")
            model.load_state_dict(torch.load(args.continue_training, 
                map_location=torch.device(device)), strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        model.to(device)

        if not os.path.isdir('results'):
                os.mkdir("results")

        if not args.continue_training:
            with open('results/' + args.rnn + '.csv', 'w') as f:
                f.write('epoch,train_loss,valid_loss,time\n')
            restart_epoch = 0

        else:
            df = pd.read_csv('results/' + args.rnn + '.csv')
            restart_epoch = len(df)
            del df

        losses = []
        std_losses = []
        # Training Loop
        for e in range(restart_epoch, restart_epoch + args.epoch):
            start = time.time()
            mean_loss = 0.

            losses_train = None
            for x_batch, y_batch in train_loader:
                # print(x_batch.shape)
                # print(y_batch.shape)
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                
                # print(output.shape)
                loss = F.mse_loss(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    loss_train = F.mse_loss(output, y_batch, reduction="none")
                    mse_tensor = torch.mean(loss_train, dim=1)
                    losses_train = mse_tensor if losses_train is None else \
                                   torch.cat((losses_train, mse_tensor), dim=0)

                # Checkpoint
                if e % checkpoint == 0 or e == restart_epoch + args.epoch - 1:
                    if not os.path.isdir('model'):
                        os.mkdir("model")
                    if not os.path.isdir("model/" + args.rnn):
                        os.mkdir("model/" + args.rnn)
                    torch.save(model.state_dict(), 
                               "model/{}/{}_{}_{}.model".format(args.rnn, 
                                                                args.rnn,
                                                                hidden_size, 
                                                                e + 1))

            mean_loss = torch.mean(losses_train, dim=0)
            std_loss = torch.std(losses_train, dim=0)

            # Validation Loss
            losses_valid = None
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    output = model(x_batch)
                    # loss_valid = (output_i - y_batch_i)^2
                    # is a 2D Tensor
                    loss_valid = F.mse_loss(output, y_batch, reduction="none")
                    # mse_tensor is one vector containing the mse of
                    # each sample in the batch
                    mse_tensor = torch.mean(loss_valid, dim=1)

                    # losses_valid contains the mse of each batch already
                    # done
                    losses_valid = mse_tensor if losses_valid is None else \
                                   torch.cat((losses_valid, mse_tensor), dim=0)

            # Get the mean of all the MSE
            mean_loss_valid = torch.mean(losses_valid, dim=0)

            # Get the standard deviation of all the MSE
            std_loss_valid = torch.std(losses_valid, dim=0)

            duration = time.time() - start

            # Write Results
            with open('results/' + args.rnn + '.csv', 'a') as f:
                f.write('{},{},{},{}\n'.format(e + 1, mean_loss, 
                                               mean_loss_valid, duration))

            print("Epoch {} MSE Train Loss: {:.4f} +- {:.4f} MSE Valid Loss: \
                {:.4f} +- {:.4f} Duration: {:.2f}".format(e + 1, 
                    mean_loss, std_loss, mean_loss_valid, std_loss_valid, duration))

    if args.evaluation:
        if not os.path.isdir("results/figure/"):
            os.mkdir("results/figure/")

        df = pd.read_csv("results/" + args.rnn + ".csv")
        print(df)
        u.plot_curve_losses(df, save_path=f"results/figure/{args.rnn}_curve_loss.png")

        data = u.load_split_dataset(path="data/{}_{}.pkl".format(args.window_size, args.forecast_size))

        X_valid = torch.Tensor(data["X_valid"])
        y_valid = torch.Tensor(data["y_valid"])

        seq_length = X_valid.shape[1]

        print(args.hidden_size, seq_length, args.forecast_size)
        model = rnns[args.rnn](input_size=5, hidden_size=args.hidden_size, 
                               seq_length=seq_length, output_size=args.forecast_size)
        model.load_state_dict(torch.load(f"model/{args.rnn}/{args.evaluation}.model", map_location=torch.device('cpu')), strict=False)

        indexes = [0, 100, 200, 250]

        for idx in indexes:
            u.plot_results(model, X_valid[idx,:,:], y_valid[idx,:], save_path=f"results/figure/{args.rnn}_{idx}.png")

    if args.comparison:
        # Data Loading
        if not data:
            data = u.load_split_dataset(path="data/{}_{}.pkl".format(
                args.window_size, args.forecast_size))

        device = torch.device('cpu')
        batch_size = 16

        X_valid = torch.Tensor(data["X_valid"])
        y_valid = torch.Tensor(data["y_valid"])

        val_set_len = X_valid.shape[0]
        seq_length = X_valid.shape[1]
        input_size = X_valid.shape[2]

        val = TensorDataset(X_valid, y_valid)

        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

        model_names = [ "LSTM_64_7" ,"GRU_64_11", "BRC_64_9", 
                        "nBRC_64_5", "attn_64_5"]

        for rnn, model_name in zip(rnns, model_names):
            model = rnns[rnn](input_size=input_size, hidden_size=args.hidden_size, 
                                   seq_length=seq_length, output_size=args.forecast_size)
            model.load_state_dict(torch.load(f"model/{rnn}/{model_name}.model", map_location=device), strict=False)

            mean_loss_valid = 0
            # Validation Loss
            losses_valid = None
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    output = model(x_batch)
                    # loss_valid = (output_i - y_batch_i)^2
                    # is a 2D Tensor
                    loss_valid = F.mse_loss(output, y_batch, reduction="none")
                    # mse_tensor is one vector containing the mse of
                    # each sample in the batch
                    mse_tensor = torch.mean(loss_valid, dim=1)

                    # losses_valid contains the mse of each batch already
                    # done
                    losses_valid = mse_tensor if losses_valid is None else \
                                   torch.cat((losses_valid, mse_tensor), dim=0)
            
            # Get the mean of all the MSE
            mean_loss_valid = torch.mean(losses_valid, dim=0)

            # Get the standard deviation of all the MSE
            std_loss_valid = torch.std(losses_valid, dim=0)

            print(model_name)
            print(f"mean +- std: {mean_loss_valid} +- {std_loss_valid}")
            # break
                
        """
        plt.figure(figsize=(7,5))
        plt.bar(model_names, mean_loss_valid, yerr = std_loss_valid)
        """

