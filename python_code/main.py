# Author: Victor Dachet

import pandas as pd
import argparse
import time
import os

import util as u

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch

from pytorch_forecasting.metrics import RMSE, MAE, SMAPE


if __name__ == '__main__':
    rnns = u.rnns
    cells = u.cells
    train_losses = u.train_losses

    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument('--forecast_size', help='Size of the forecast window', 
                        type=int, default=60)
    parser.add_argument('--dataset_size', help='Eval model', 
                        type=str, default="small")

    # Training args
    parser.add_argument('-ep', '--epoch', help='Number of epoch',
                        type=int, default=10)
    parser.add_argument('--farm', help='Farm',
                        type=int, default=0, choices=[0,1,2])
    parser.add_argument('-batch', '--batch_size', help='Batch size', type=int,
                        default=32)
    parser.add_argument('-lr', '--learning_rate', help='Actor learning rate', 
                        type=float, default=1e-4)
    parser.add_argument('--loss', help='Loss to train the model', 
        required=False, default="MSE", type=str, choices=train_losses.keys())    

    # Model args
    parser.add_argument('--hidden_size', help='Size of hidden layers', type=int,
                        default=64)
    parser.add_argument('--num_layers', help='Number of layers in the RNN', 
                        type=int, default=1)
    parser.add_argument('-t','--training', help='Train the model', 
                        action="store_true")
    parser.add_argument('--rnn', help='RNN type: LSTM GRU BRC nBRC attn', 
                        type=str, choices=rnns.keys())
    parser.add_argument('-c_t','--continue_training',
        help='Continue the training. Requires the path of the model to train', 
        required=False, default=None, type=str)
    parser.add_argument('--cell',
        help='Type of cell in rnn architecture_history_forecast', 
        required=False, default="", type=str, choices=cells.keys())    

    # Model Evaluation args
    parser.add_argument('-e','--evaluation', help='Eval model', 
                        action="store_true")

    # Model Comparison
    parser.add_argument('-c','--comparison', help='Compare the models', 
                        action="store_true")

    args = parser.parse_args()

    

    model_training = args.training

    model_name = f"{args.rnn}_{args.cell}_{args.loss}_{args.hidden_size}"

    data = None

    ######### COMMON PART ##########

    # Constant Parameters definition
    device = torch.device("cuda:0") if torch.cuda.is_available() \
                                    else torch.device("cpu")
    print("device", device)
    batch_size = args.batch_size
    checkpoint = 1
    quarter=96
    farm=args.farm
    gap=48
    history_size=96
    forecast_horizon=96

    # Retrieve dataset
    X_train, y_train = u.get_dataset_rnn(quarter=quarter, farm=farm, 
                                         type_data="train", gap=gap, 
                                         history_size=history_size, 
                                         forecast_horizon=forecast_horizon, 
                                         size=args.dataset_size, tensor=True)
    X_valid, y_valid = u.get_dataset_rnn(quarter=quarter, farm=farm, 
                                         type_data="valid", gap=gap, 
                                         history_size=history_size, 
                                         forecast_horizon=forecast_horizon, 
                                         size=args.dataset_size, tensor=True)

    print(f"X_train {X_train.shape}, y_train {y_train.shape}")

    shuffle = True if args.training else False
    print("shuffle", shuffle)
    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size=batch_size, shuffle=shuffle, 
                              drop_last=False)
    val_loader = DataLoader(TensorDataset(X_valid, y_valid), 
                            batch_size=batch_size, shuffle=shuffle, 
                            drop_last=False)

    # Model Parameters
    input_size = X_train.shape[2] # number of expected features for input x
    hidden_size = args.hidden_size # Num of units in the RNN
    num_layers = args.num_layers # Number of recurrent layers
    output_size = y_train.shape[1]
    seq_length = X_train.shape[1]

    rnn = rnns[args.rnn]
    transformer_with_decoder = (rnn == rnns["TransformerEncoderDecoder"])

    cell_name = None if args.cell == "" else args.cell
    print("cell_name", cell_name)

    model = u.init_model(rnn=rnn, input_size=input_size, 
                         hidden_size=hidden_size, seq_length=seq_length, 
                         output_size=output_size, gap_length=gap, 
                         histo_length=history_size, nhead=num_layers, 
                         nlayers=num_layers, device=device, 
                         cell_name=cell_name)


    path_model = args 
    path_csv = 'results/' + model_name + '.csv'

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
            with open(path_csv, 'w') as f:
                f.write('epoch,train_loss,valid_loss,time\n')
            restart_epoch = 0

        else:
            df = pd.read_csv(path_csv)
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
                if transformer_with_decoder:
                    output = model(x_batch, y_batch)
                else:
                    output = model(x_batch)
                output = output.to(device)
                # print("output", output.device)
                # print("y_batch", y_batch.device)
                # print(output.shape)

                # loss = F.mse_loss(output, y_batch)
                # print("losss", loss)
                loss_train_pf = train_losses[args.loss](output, y_batch)
                
                # print("loss_train_pf", loss_train_pf)
                optimizer.zero_grad()
                # loss.backward()
                loss_train_pf.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                with torch.no_grad():
                    loss_train = F.mse_loss(output, y_batch, reduction="none")
                    mse_tensor = torch.mean(loss_train, dim=1)
                    losses_train = mse_tensor if losses_train is None else \
                                   torch.cat((losses_train, mse_tensor), dim=0)

                # Checkpoint
                if not os.path.isdir('model'):
                    os.mkdir("model")
                if not os.path.isdir("model/" + model_name):
                    os.mkdir("model/" + model_name)
                torch.save(model.state_dict(), 
                           "model/{}/{}_{}_{}.model".format(model_name, 
                                                            model_name,
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
                    if transformer_with_decoder:
                        output = model.predict(x_batch)
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
            with open(path_csv, 'a') as f:
                f.write('{},{},{},{}\n'.format(e + 1, mean_loss, 
                                               mean_loss_valid, duration))

            print("Epoch {} MSE Train Loss: {:.4f} +- {:.4f} MSE Valid Loss: \
                {:.4f} +- {:.4f} Duration: {:.2f}".format(e + 1, mean_loss, 
                        std_loss, mean_loss_valid, std_loss_valid, duration))

    ### MODEL EVALUATION ###
    if args.evaluation:
        X_test, y_test = u.get_dataset_rnn(quarter=quarter, farm=farm, 
                                         type_data="test", gap=gap, 
                                         history_size=history_size, 
                                         forecast_horizon=forecast_horizon, 
                                         size=args.dataset_size, tensor=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), 
                                 batch_size=batch_size, shuffle=shuffle, 
                                 drop_last=False)
        # Create tree structure
        if not os.path.isdir("results/figure/"):
            os.mkdir("results/figure/")
        if not os.path.isdir(f"results/figure/{model_name}/"):
                os.mkdir(f"results/figure/{model_name}/")

        indexes = [0, 1, 2, 3, 18]

        df = pd.read_csv(path_csv)
        # print(df)
        u.plot_curve_losses(df, save_path=f"results/figure/{model_name}/{model_name}_curve_loss.pdf")
        best_e = df["epoch"].iloc[df["valid_loss"].argmin()]
        
        print("Loading best model:", df[df["epoch"] == best_e])
        model.load_state_dict(torch.load("model/{}/{}_{}_{}.model".format(model_name, 
                                                            model_name,
                                                            hidden_size, 
                                                            best_e), map_location=torch.device("cpu")), strict=False)
        model = model.to(device)

        rmse = RMSE(reduction="none")
        mae = MAE(reduction="none")
        smape = SMAPE(reduction="none")

        metrics = [rmse, mae, smape]

        Xs = [X_train, X_valid, X_test]
        ys = [y_train, y_valid, y_test]
        loaders = [train_loader, val_loader, test_loader]
        set_types = ["train", "valid", "test"]

        for X, y, loader, set_type in zip(Xs, ys, loaders, set_types):
            plot_bias = (set_type == "valid")

            y_hat = u.predict(model, data_loader=loader, fh=forecast_horizon, 
                              device=device, 
                              transformer_with_decoder=transformer_with_decoder)
            losses = u.apply_metric(metrics, y_hat=y_hat, y_truth=y, set_type=set_type, plot_bias=plot_bias)


            for metric in metrics:
                metric_name = metric.__repr__()[:-2]
                u.plot_best_worst_histo(model, X=X, y=y, 
                                losses=losses[metric_name], 
                                loss_name=metric_name, 
                                set_type=set_type, 
                                model_name=model_name, 
                                transformer_with_decoder=transformer_with_decoder,
                                device=device)

    

