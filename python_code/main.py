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

    # Model args
    parser.add_argument('--hidden_size', help='Size of hidden layers', type=int,
                        default=64)
    parser.add_argument('--num_layers', help='Number of layers in the RNN', 
                        type=int, default=1)
    parser.add_argument('-t','--training', help='Train the model', 
                        action="store_true")
    parser.add_argument('--rnn', help='RNN type: LSTM GRU BRC nBRC attn', type=str, choices=rnns.keys())
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
    data = None

    ######### COMMON PART ##########
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("device", device)
    batch_size = args.batch_size
    checkpoint = 1

    quarter=96
    farm=args.farm
    gap=48
    history_size=96
    forecast_horizon=96
    X_train, y_train = u.get_dataset_rnn(quarter=quarter, farm=farm, type_data="train", gap=gap, 
                               history_size=history_size, forecast_horizon=forecast_horizon, size=args.dataset_size, tensor=True)
    X_valid, y_valid = u.get_dataset_rnn(quarter=quarter, farm=farm, type_data="valid", gap=gap, 
                               history_size=history_size, forecast_horizon=forecast_horizon, size=args.dataset_size, tensor=True)

    train_set_len = X_train.shape[0]
    val_set_len = X_valid.shape[0]
    

    print(f"X_train {X_train.shape}, y_train {y_train.shape}")


    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=True, drop_last=False)

    # Model Parameters
    input_size = X_train.shape[2] # number of expected features for input x
    hidden_size = args.hidden_size # Num of units in the RNN
    num_layers = args.num_layers # Number of recurrent layers
    output_size = y_train.shape[1]
    seq_length = X_train.shape[1]

    rnn = rnns[args.rnn]

    isTransformer = (rnn == rnns["TransformerEncoderDecoder"])

    src_mask = None

    cell_name = None if args.cell == "" else args.cell
    print("cell_name", cell_name)

    model = u.init_model(rnn=rnn, input_size=input_size, hidden_size=hidden_size, seq_length=seq_length, output_size=output_size, 
               gap_length=gap, histo_length=history_size, nhead=num_layers, nlayers=num_layers, device=device, cell_name=cell_name)
    


    path_model = args 
    path_csv = 'results/' + args.rnn + "_" + str(args.hidden_size) + '.csv'

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
                if isTransformer:
                    output = model(x_batch, y_batch)
                else:
                    output = model(x_batch)
                output = output.to(device)
                # print("output", output.device)
                # print("y_batch", y_batch.device)
                # print(output.shape)
                loss = F.mse_loss(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
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
                    if isTransformer:
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
                {:.4f} +- {:.4f} Duration: {:.2f}".format(e + 1, 
                    mean_loss, std_loss, mean_loss_valid, std_loss_valid, duration))

    ### MODEL EVALUATION ###
    if args.evaluation:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=False)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)
        # Create tree structure
        if not os.path.isdir("results/figure/"):
            os.mkdir("results/figure/")
        if not os.path.isdir(f"results/figure/{args.rnn}/"):
                os.mkdir(f"results/figure/{args.rnn}/")

        indexes = [0, 1, 2, 3, 18]

        df = pd.read_csv(path_csv)
        print(df)
        u.plot_curve_losses(df, save_path=f"results/figure/{args.rnn}/{args.rnn}_{args.hidden_size}_curve_loss.pdf")
        best_e = df["epoch"].iloc[df["valid_loss"].argmin()]
        
        print("Loading best model:", df[df["epoch"] == best_e])
        model.load_state_dict(torch.load("model/{}/{}_{}_{}.model".format(args.rnn, 
                                                            args.rnn,
                                                            hidden_size, 
                                                            best_e), map_location=torch.device("cpu")), strict=False)
        model = model.to(device)

        rmse = RMSE(reduction="none")
        mape = MAE(reduction="none")
        smape = SMAPE(reduction="none")

        losses_train = None
        losses_train_mae = None
        losses_train_smape = None
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if isTransformer:
                output = model.predict(x_batch)
            else:
                output = model(x_batch)
            with torch.no_grad():
                loss_train = rmse(output, y_batch)
                rmse_tensor = torch.sqrt(torch.mean(loss_train,axis=1)) #  torch.mean(loss_train, dim=1)
                losses_train = rmse_tensor if losses_train is None else \
                               torch.cat((losses_train, rmse_tensor), dim=0)


                # y_batch = torch.clip(y_batch, min=0.0, max=1.0)
                loss_train_mape = mape(output, y_batch)
                # print("loss_train_mape",loss_train_mape.shape)
                mae_tensor = torch.mean(loss_train_mape, axis=1)
                # print("mae_tensor", mae_tensor.shape, mae_tensor.max(), mae_tensor.min())
                losses_train_mae = mae_tensor if losses_train_mae is None else \
                               torch.cat((losses_train_mae, mae_tensor), dim=0)

                loss_train_smape = smape(output, y_batch)
                # print("loss_train_mape",loss_train_mape.shape)
                smape_tensor = torch.mean(loss_train_smape, axis=1)
                # print("mae_tensor", mae_tensor.shape, mae_tensor.max(), mae_tensor.min())
                losses_train_smape = smape_tensor if losses_train_smape is None else \
                               torch.cat((losses_train_smape, smape_tensor), dim=0)

        print(f"SMAPE TRAIN: {torch.mean(losses_train_smape):.4f} \pm {torch.std(losses_train_smape):.4f}")
        best = torch.argmin(losses_train_smape)
        u.plot_results(model, X_train[best,:,:].to(device), y_train[best,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_best_SMAPE_valid.pdf", src_mask=src_mask)
        worst = torch.argmax(losses_train_smape)
        u.plot_results(model, X_train[worst,:,:].to(device), y_train[worst,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_worst_SMAPE_valid.pdf", src_mask=src_mask)
        print("losses_train_smape", losses_train_smape.shape, losses_train_smape)

        print("losses_train", losses_train.shape)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(losses_train.unsqueeze(0), bins=30)
        plt.xlabel("RMSE")
        plt.ylabel("Nbr of occurences")
        plt.savefig(f"results/figure/{args.rnn}/{args.rnn}_histo_train.pdf")

        print(f"MAE TRAIN: {torch.mean(losses_train_mae):.4f} \pm {torch.std(losses_train_mae):.4f}")
        print(f"RMSE TRAIN: {torch.mean(losses_train):.4f} \pm {torch.std(losses_train):.4f}")
        best = torch.argmin(losses_train)
        # print("Best", best)
        u.plot_results(model, X_train[best,:,:].to(device), y_train[best,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_best_train.pdf", src_mask=src_mask)

        for idx in indexes:
            u.plot_results(model, X_train[idx,:,:].to(device), y_train[idx,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_{idx}_train.pdf", src_mask=src_mask)

        losses_train = None
        losses_train_mae = None
        losses_train_smape = None
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if isTransformer:
                output = model.predict(x_batch)
            else:
                output = model(x_batch)
            with torch.no_grad():
                loss_train = rmse(output, y_batch)
                # print("loss_train", loss_train.shape)
                rmse_tensor = torch.sqrt(torch.mean(loss_train,axis=1))
                # print("rmse_tensor", rmse_tensor.shape)
                losses_train = rmse_tensor if losses_train is None else \
                               torch.cat((losses_train, rmse_tensor), dim=0)

                # y_batch = torch.clip(y_batch, min=0.0, max=1.0)
                loss_train_mape = mape(output, y_batch)
                # print("loss_train_mape",loss_train_mape.shape)
                mae_tensor = torch.mean(loss_train_mape, axis=1)
                # print("mae_tensor", mae_tensor.shape, mae_tensor.max(), mae_tensor.min())
                losses_train_mae = mae_tensor if losses_train_mae is None else \
                               torch.cat((losses_train_mae, mae_tensor), dim=0)

                loss_train_smape = smape(output, y_batch)
                # print("loss_train_mape",loss_train_mape.shape)
                smape_tensor = torch.mean(loss_train_smape, axis=1)
                # print("mae_tensor", mae_tensor.shape, mae_tensor.max(), mae_tensor.min())
                losses_train_smape = smape_tensor if losses_train_smape is None else \
                               torch.cat((losses_train_smape, smape_tensor), dim=0)

        # print("losses_train", losses_train.shape)
        plt.figure()
        plt.hist(losses_train.unsqueeze(0), bins=30)
        plt.xlabel("RMSE")
        plt.ylabel("Nbr of occurences")
        plt.savefig(f"results/figure/{args.rnn}/{args.rnn}_histo_valid.pdf")
        plt.close()
        print(f"MAE VALID: {torch.mean(losses_train_mae):.4f} \pm {torch.std(losses_train_mae):.4f}")

        print(f"SMAPE VALID: {torch.mean(losses_train_smape):.4f} \pm {torch.std(losses_train_smape):.4f}")
        plt.figure()
        plt.hist(losses_train_smape.unsqueeze(0), bins=30)
        plt.xlabel("SMAPE")
        plt.ylabel("Nbr of occurences")
        plt.savefig(f"results/figure/{args.rnn}/{args.rnn}_SMAPE_histo_valid.pdf")

        best = torch.argmin(losses_train_smape)
        u.plot_results(model, X_valid[best,:,:].to(device), y_valid[best,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_best_SMAPE_valid.pdf", src_mask=src_mask)
        worst = torch.argmax(losses_train_smape)
        u.plot_results(model, X_valid[worst,:,:].to(device), y_valid[worst,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_worst_SMAPE_valid.pdf", src_mask=src_mask)
        print("losses_train_smape", losses_train_smape.shape, losses_train_smape)

        best = torch.argmin(losses_train_mae)
        # print("Best", best, losses_train_mae[best])
        u.plot_results(model, X_valid[best,:,:].to(device), y_valid[best,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_best_mae_valid.pdf", src_mask=src_mask)
        worst = torch.argmax(losses_train_mae)
        # print("worst", worst, losses_train_mae[worst])
        u.plot_results(model, X_valid[worst,:,:].to(device), y_valid[worst,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_worst_mae_valid.pdf", src_mask=src_mask)


        print(f"RMSE VALID: {torch.mean(losses_train):.4f} \pm {torch.std(losses_train):.4f}")


        best = torch.argmin(losses_train)
        # print("Best", best, losses_train[best])
        u.plot_results(model, X_valid[best,:,:].to(device), y_valid[best,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_best_valid.pdf", src_mask=src_mask)
        worst = torch.argmax(losses_train)
        # print("worst", worst, losses_train[worst])
        u.plot_results(model, X_valid[worst,:,:].to(device), y_valid[worst,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_worst_valid.pdf", src_mask=src_mask)
        for idx in indexes:
            u.plot_results(model, X_valid[idx,:,:].to(device), y_valid[idx,:].to(device), save_path=f"results/figure/{args.rnn}/{args.rnn}_{idx}_valid.pdf", src_mask=src_mask)
            # print(idx, losses_train[idx] )

    

