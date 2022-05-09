# Author: Victor Dachet

import pandas as pd
import argparse
import time
import os

import util as u
from model import LSTM, GRU, BRC, nBRC, simple_rnn, architecture, architecture_history_forecast
from attention import Attention_Net
from transformer import TransformerModel, Transformer_enc_dec, generate_square_subsequent_mask, TransformerModelWithoutMask

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch

from pytorch_forecasting.metrics import RMSE, MAPE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument('--forecast_size', help='Size of the forecast window', 
                        type=int, default=60)
    parser.add_argument('--dataset_size', help='Eval model', 
                        type=str, default="small")

    # Training args
    parser.add_argument('-ep', '--epoch', help='Number of epoch',
                        type=int, default=10)
    parser.add_argument('-batch', '--batch_size', help='Batch size', type=int,
                        default=32)
    parser.add_argument('-lr', '--learning_rate', help='Actor learning rate', 
                        type=float, default=1e-4)

    # Model args
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
            "attn":Attention_Net, "simple_rnn":simple_rnn, 
            "architecture":architecture, 
            "architecture_history_forecast":architecture_history_forecast,
            "TransformerModel": TransformerModel,
            "Transformer_enc_dec": Transformer_enc_dec,
            "TransformerModelWithoutMask": TransformerModelWithoutMask}

    model_training = args.training
    data = None

    ######### COMMON PART ##########
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("device", device)
    batch_size = args.batch_size
    checkpoint = 1

    day=96
    farm=0
    gap=48
    history_size=96
    forecast_horizon=96
    X_train, y_train = u.get_dataset_rnn(day=day, farm=farm, type_data="train", gap=gap, 
                               history_size=history_size, forecast_horizon=forecast_horizon, size=args.dataset_size)
    X_valid, y_valid = u.get_dataset_rnn(day=day, farm=farm, type_data="valid", gap=gap, 
                               history_size=history_size, forecast_horizon=forecast_horizon, size=args.dataset_size)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    train_set_len = X_train.shape[0]

    print(f"X_train {X_train.shape}, y_train {y_train.shape}")

    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).float()
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

    src_mask = None
    if rnn == architecture:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
                     seq_length=seq_length, output_size=output_size,
                     gap_length=gap)
    elif rnn == architecture_history_forecast:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
            output_size=output_size, histo_length=history_size, gap_length=gap)
    elif rnn == TransformerModel or rnn == TransformerModelWithoutMask:
        nhead = input_size 
        model = rnn(d_model=input_size, nhead=nhead, d_hid=hidden_size, nlayers=num_layers, dropout=0.2)
        src_mask = generate_square_subsequent_mask(seq_length)
        src_mask = src_mask.to(device)
        print("src_mask", src_mask.shape)
    elif rnn == Transformer_enc_dec:
        nhead = input_size 
        model = Transformer_enc_dec(d_model=input_size, nhead=nhead, d_hid=hidden_size, nlayers=num_layers, dropout=0.2, device=device)
        src_mask = generate_square_subsequent_mask(seq_length)
        src_mask = src_mask.to(device)
        print("src_mask", src_mask.shape)

    else:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
                     seq_length=seq_length, output_size=output_size)

    ### MODEL TRAINING ###
    if model_training:

        if args.continue_training:
            print("Loading the model to continue the training...")
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
                if rnn == TransformerModel or rnn == Transformer_enc_dec or rnn == TransformerModelWithoutMask:
                    output = model(x_batch, src_mask)
                else:
                    output = model(x_batch)
                output = output.to(device)
                # print("output", output.device)
                # print("y_batch", y_batch.device)
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
                if not os.path.isdir('model'):
                    os.mkdir("model")
                if not os.path.isdir("model/" + args.rnn):
                    os.mkdir("model/" + args.rnn)
                torch.save(model.state_dict(), 
                           "model/{}/{}_{}_{}.model".format(args.rnn, 
                                                            args.rnn,
                                                            hidden_size, 
                                                            e + 1))

            # print("losses_train", losses_train.shape)
            mean_loss = torch.mean(losses_train, dim=0)
            std_loss = torch.std(losses_train, dim=0)

            # Validation Loss
            losses_valid = None
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    if rnn == TransformerModel or rnn == Transformer_enc_dec or rnn == TransformerModelWithoutMask:
                        output = model(x_batch, src_mask)
                    else:
                        output = model(x_batch)
                    output = output.to(device)
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

    ### MODEL EVALUATION ###
    if args.evaluation:
        
        if not os.path.isdir("results/figure/"):
            os.mkdir("results/figure/")

        indexes = [0, 1, 2, 3]

        df = pd.read_csv("results/" + args.rnn + ".csv")
        print(df)
        u.plot_curve_losses(df, save_path=f"results/figure/{args.rnn}_curve_loss.png")
        
        model.load_state_dict(torch.load(f"model/{args.rnn}/{args.evaluation}.model", map_location=torch.device('cpu')), strict=False)

        rmse = RMSE(reduction="none")

        losses_train = None
        for x_batch, y_batch in train_loader:
            if rnn == TransformerModel or rnn == Transformer_enc_dec or rnn == TransformerModelWithoutMask:
                output = model(x_batch, src_mask)
            else:
                output = model(x_batch)
            with torch.no_grad():
                loss_train = rmse(output, y_batch)
                rmse_tensor = torch.sqrt(torch.mean(loss_train,axis=1)) #  torch.mean(loss_train, dim=1)
                losses_train = rmse_tensor if losses_train is None else \
                               torch.cat((losses_train, rmse_tensor), dim=0)

        print(f"RMSE TRAIN: {torch.mean(losses_train):.4f} \pm {torch.std(losses_train):.4f}")
        best = torch.argmin(losses_train)
        print("Best", best)
        u.plot_results(model, X_train[best,:,:], y_train[best,:], save_path=f"results/figure/{args.rnn}_best_train.png", src_mask=src_mask)

        for idx in indexes:
            u.plot_results(model, X_train[idx,:,:], y_train[idx,:], save_path=f"results/figure/{args.rnn}_{idx}_train.png", src_mask=src_mask)

        losses_train = None
        for x_batch, y_batch in val_loader:
            if rnn == TransformerModel or rnn == Transformer_enc_dec or rnn == TransformerModelWithoutMask:
                output = model(x_batch, src_mask)
            else:
                output = model(x_batch)
            with torch.no_grad():
                loss_train = rmse(output, y_batch)
                print("loss_train", loss_train.shape)
                rmse_tensor = torch.sqrt(torch.mean(loss_train,axis=1))
                print("rmse_tensor", rmse_tensor.shape)
                losses_train = rmse_tensor if losses_train is None else \
                               torch.cat((losses_train, rmse_tensor), dim=0)

        print(f"RMSE VALID: {torch.mean(losses_train):.4f} \pm {torch.std(losses_train):.4f}")
        best = torch.argmin(losses_train)
        print("Best", best)
        u.plot_results(model, X_valid[best,:,:], y_valid[best,:], save_path=f"results/figure/{args.rnn}_best_valid.png", src_mask=src_mask)
        for idx in indexes:
            u.plot_results(model, X_valid[idx,:,:], y_valid[idx,:], save_path=f"results/figure/{args.rnn}_{idx}.png", src_mask=src_mask)

    # if args.comparison:
    #     # Data Loading
    #     day = 95
    #     X_train, y_train = u.get_dataset_rnn(day=day, farm=0, type_data="train", gap=48, 
    #                                history_size=96, forecast_horizon=96, size=args.dataset_size)
    #     X_valid, y_valid = u.get_dataset_rnn(day=day, farm=0, type_data="valid", gap=48, 
    #                                history_size=96, forecast_horizon=96, size=args.dataset_size)

    #     X_train = torch.from_numpy(X_train)
    #     y_train = torch.from_numpy(y_train)

    #     X_valid = torch.from_numpy(X_valid)
    #     y_valid = torch.from_numpy(y_valid)

    #     device = torch.device('cpu')
    #     batch_size = 16

    #     val_set_len = X_valid.shape[0]
    #     seq_length = X_valid.shape[1]
    #     input_size = X_valid.shape[2]

    #     val = TensorDataset(X_valid, y_valid)

    #     val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

    #     model_names = [ "simple_rnn_128_20" ]

    #     for rnn, model_name in zip(rnns, model_names):
    #         model = rnns[rnn](input_size=input_size, hidden_size=args.hidden_size, 
    #                                seq_length=seq_length, output_size=args.forecast_size)
    #         model.load_state_dict(torch.load(f"model/{rnn}/{model_name}.model", map_location=device), strict=False)

    #         mean_loss_valid = 0
    #         # Validation Loss
    #         losses_valid = None
    #         with torch.no_grad():
    #             for x_batch, y_batch in val_loader:
    #                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    #                 output = model(x_batch)
    #                 # loss_valid = (output_i - y_batch_i)^2
    #                 # is a 2D Tensor
    #                 loss_valid = F.mse_loss(output, y_batch, reduction="none")
    #                 # mse_tensor is one vector containing the mse of
    #                 # each sample in the batch
    #                 mse_tensor = torch.mean(loss_valid, dim=1)

    #                 # losses_valid contains the mse of each batch already
    #                 # done
    #                 losses_valid = mse_tensor if losses_valid is None else \
    #                                torch.cat((losses_valid, mse_tensor), dim=0)
            
    #         # Get the mean of all the MSE
    #         mean_loss_valid = torch.mean(losses_valid, dim=0)

    #         # Get the standard deviation of all the MSE
    #         std_loss_valid = torch.std(losses_valid, dim=0)

    #         print(model_name)
    #         print(f"mean +- std: {mean_loss_valid} +- {std_loss_valid}")
            # break
                
        """
        plt.figure(figsize=(7,5))
        plt.bar(model_names, mean_loss_valid, yerr = std_loss_valid)
        """

