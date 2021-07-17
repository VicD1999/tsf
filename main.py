# Author: Victor Dachet
from scipy import stats
import numpy as np
import pandas as pd
import argparse

import util as u
from model import *

from torch.utils.data import TensorDataset, DataLoader

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
    parser.add_argument('--num_layers', help='Number of layers in the RNN', type=int,
                        default=1)

    parser.add_argument('-d','--dataset_creation', help='Create the dataset', 
                        action="store_true")
    parser.add_argument('-t','--training', help='Train the model', 
                        action="store_true")

    
    args = parser.parse_args()

    dataset_creation = args.dataset_creation
    model_training = args.training
    data = None

    if dataset_creation:
        u.create_dataset(vervose=False)
        
        df = pd.read_csv("data/dataset.csv")

        data = u.get_random_split_dataset(df)

        u.write_split_dataset(data)
        

    if model_training:
        batch_size = args.batch_size
        
        # Data Loading
        if not data:
            data = u.load_split_dataset()

        X_train = torch.Tensor(data["X_train"])
        y_train = torch.Tensor(data["y_train"])

        X_valid = torch.Tensor(data["X_valid"])
        y_valid = torch.Tensor(data["y_valid"])

        train = TensorDataset(X_train, y_train)
        val = TensorDataset(X_valid, y_valid)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

        # Model Parameters
        input_size = X_train.shape[2] # number of expected features in the input x
        hidden_size = args.hidden_size
        num_layers = args.num_layers # Number of recurrent layers
        output_size = y_train.shape[1]
        seq_length = X_train.shape[1]

        lstm_model = LSTM(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=num_layers, seq_length=seq_length, 
                          output_size=output_size)

        for x_batch, y_batch in train_loader:
            print(x_batch.shape)
            # print(y_batch.shape)
            output = lstm_model(x_batch)
            
            print(output.shape)
            break


