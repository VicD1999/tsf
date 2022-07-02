# Author: Victor Dachet

import numpy as np
from scipy.io import netcdf
import pandas as pd
import datetime
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import subprocess
import os

import torch
import torch.nn.functional as F
import torch.nn as nn

from sktime.performance_metrics.forecasting import MeanSquaredError
from pytorch_forecasting.metrics import RMSE, MAE, SMAPE

paths_ores = glob.glob("data/ORES/*.csv")
paths_ores.sort()

paths_mar = ["data/MAR/concat.nc"]

from model import BRC, nBRC, simple_rnn, architecture, architecture_history_forecast, history_forecast, HybridRNN, ahf
from attention import Attention_Net
from transformer import Transformer, TransformerEncoderDecoder

rnns = {"attn":Attention_Net, "simple_rnn":simple_rnn, 
        "architecture":architecture, 
        "architecture_history_forecast":architecture_history_forecast,
        "history_forecast": history_forecast,
        "Transformer": Transformer,
        "TransformerEncoderDecoder":TransformerEncoderDecoder,
        "ahf": ahf}

cells = {"BRC": BRC,
         "nBRC": nBRC,
         "GRU": nn.GRU,
         "LSTM": nn.LSTM,
         "HybridRNN": HybridRNN}

def MSEsMAPE(y_pred, y):
    rmse = RMSE(reduction="mean")
    smape = SMAPE(reduction="mean")
    return rmse(y_pred, y) + smape(y_pred, y)


train_losses = {"MSE":RMSE(reduction="mean"), 
                "MAE":MAE(reduction="mean"),
                "SMAPE":SMAPE(reduction="mean"),
                "MSEsMAPE":MSEsMAPE}

def exist_or_download(path_to_model, model_name):
    """
    args:
        path_to_model: string e.g. "/model/architecture_GRU_MSE_512/architecture_GRU_MSE_512_39.model"
        model_name: string e.g. "architecture_GRU_MSE_512"
    """
    if not os.path.exists(path_to_model):
        if not os.path.isdir(f"model/{model_name}/"):
            os.mkdir(f"model/{model_name}/")
        bashCommand = f"scp -i ~/.ssh/vegamissile victor@vega.mont.priv:/home/victor/tsf/{path_to_model} model/{model_name}/"
        process = subprocess.run(bashCommand.split())
        # print("process", process)



def predict(model, data_loader, fh, device, transformer_with_decoder):
    """
    Warning if the data loader is big this function could be memory 
    intensive
    args:
    -----
        model: torch model
        data_loader: DataLoader
        fh: int forecasting horizon
        device: 'cpu' or 'cuda:0' 
        transformer_with_decoder: boolean

    return:
    -------
        y_hat: Tensor of shape (len(data_loader, fh))
    """
    y_hat = None
    # print("Memory", torch.cuda.memory_summary(device=device, abbreviated=False))
    with torch.no_grad():
        for x_batch, _ in data_loader:
            # print("x_batch", x_batch.shape)
            x_batch = x_batch.to(device)
            # print("x_batch", x_batch.device)
            if transformer_with_decoder:
                output = model.predict(x_batch)
            else:
                output = model(x_batch)

            y_hat = output.to("cpu") if y_hat is None else torch.cat((y_hat, output.to("cpu")), dim=0)
            # print("y_hat", y_hat.device)
            # print("Memory", torch.cuda.memory_summary(device=device, abbreviated=False))
    return y_hat


def apply_metric(metrics, y_hat, y_truth, set_type, plot_bias=False, verbose=True):
    """
    args:
        metrics: list of metrics to apply
        y_hat: Tensor of shape (n_samples, fh) containing the
               predictions of the model
        y_truth: Tensor of shape (n_samples, fh) containing the ground 
                 truth
        set_type: string either train, valid or test
        verbose: boolean to print or not the values of losses
    return:
        losses: a dictionary with as key "metric_name" and value the value of 
                the metric
    """
    assert len(y_hat.shape) ==  len(y_truth.shape)
    assert y_hat.shape[0] == y_truth.shape[0]
    assert y_hat.shape[1] == y_truth.shape[1]

    losses = {}
    for metric in metrics:
        metric_name = metric.__repr__()[:-2] # Remove the () at the end of the method
        non_reduced = metric(y_hat, y_truth)

        if metric_name == "RMSE":
            reduced = torch.sqrt(torch.mean(non_reduced, axis=1))
        else:
            reduced = torch.mean(non_reduced, axis=1)
        
        if verbose:
            print(f"{metric_name} {set_type}: {torch.mean(reduced):.4f} \pm {torch.std(reduced):.4f}")

        losses[metric_name + "_non_reduced"] = non_reduced
        losses[metric_name] = reduced

    return losses

def plot_best_worst_histo(model, X, y, losses, loss_name, set_type, model_name, transformer_with_decoder, device):
    """
    args:
    -----
    model: model to evaluate
    X: Tensor of shape (n_samples, seq_length, nfeatures)
    y: Tensor of shape (n_samples,  fh) containing the ground truth
    losses: Tensor of shape (n_samples)
    loss_name: str e.g.: RMSE, MAE, SMAPE
    set_type: string either train, valid or test
    model_name: string
    transformer_with_decoder: boolean
    device: either 'cpu' or 'cuda:0'
    """
    prefix_path=f"results/figure/{model_name}/{model_name}"
    sufix_path=f"{loss_name}_{set_type}.pdf"

    best = torch.argmin(losses)
    worst = torch.argmax(losses)
    second_worst = torch.argsort(losses)[-2]
    middle = torch.argsort(losses)[losses.shape[0]//2]
    indices = [("_best_", best), ("_worst_", worst), ("_18_", 18), ("_6_", 6), ("_second_worst_", second_worst), ("_middle_", middle)]

    for name, idx in indices:
        y_pred = plot_results(model, X[idx,:,:].to(device), 
                       y[idx,:].to(device), 
                       save_path=prefix_path + name + sufix_path, 
                       transformer_with_decoder=transformer_with_decoder)
        y_pred = y_pred.reshape((1, -1))
        y_truth = y[idx,:].reshape((1, -1))
        # print("y_pred", y_pred.shape)
        # print("y[idx,:]", y[idx,:].shape)
        rmse = train_losses["MSE"](y_pred, y_truth)
        rmse = torch.sqrt(rmse)
        mae = train_losses["MAE"](y_pred, y_truth)
        smape = train_losses["SMAPE"](y_pred, y_truth)
        print(f"{name[1:-1]} on {set_type} set w.r.t. the {loss_name} metric with RMSE ${rmse:.4f}$, MAE ${mae:.4f}$ and SMAPE ${smape:.4f}$")
    
    plt.figure()
    n, bins, _ = plt.hist(losses.unsqueeze(0).detach(), bins=30)
    plt.xlabel(loss_name)
    plt.ylabel("Nbr of occurences")
    # print("n", n, np.max(n))
    # print("bins", bins)
    step = 3 if set_type == "train" else 1
    plt.yticks(range(0, int(np.max(n)), step))
    plt.grid(axis='y')
    plt.savefig(prefix_path + f"_histo_" + sufix_path)
    plt.close()


def init_model(rnn, input_size, hidden_size, seq_length, output_size, 
               gap_length, histo_length, nhead, nlayers, device, cell_name):
    """
    args:
        rnn: model function
        input_size, hidden_size, seq_length: 3 * int 
        output_size, gap_length, histo_length: 3 * int
        nhead, nlayers: 2 * int 
        device: either cpu or gpu
        cell_name: string ["BRC", "nBRC", "GRU", "LSTM", "HybridRNN"]
    """
    print("MODEL init", rnn, "input_size", input_size, "hidden_size", hidden_size, "seq_length", seq_length, 
                         "output_size", output_size, "gap_length",gap_length, 
                         "histo_length", histo_length, "nhead", nhead, 
                         "nlayers", nlayers, "device", device, 
                         "cell_name", cell_name)
    assert type(cell_name) == str or cell_name == None

    if rnn == architecture:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
                     history_size=histo_length, output_size=output_size,
                     gap_length=gap_length)
    elif rnn == architecture_history_forecast:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
            output_size=output_size, histo_length=histo_length, gap_length=gap_length)
    elif rnn == ahf:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
            output_size=output_size, histo_length=histo_length, gap_length=gap_length)
    elif rnn == history_forecast:
        if cell_name == "GRU":
            model = rnn(input_size=input_size, hidden_size=hidden_size, 
                output_size=output_size, histo_length=histo_length, gap_length=gap_length)
        else:
            model = rnn(input_size=input_size, hidden_size=hidden_size, 
                output_size=output_size, histo_length=histo_length, gap_length=gap_length,
                rnn_cell=cells[cell_name])
    elif rnn == Transformer:
        model = rnn(d_model=input_size, nhead=nhead, d_hid=hidden_size, nlayers=nlayers, dropout=0.2, target_length=output_size, device=device)
    elif rnn == TransformerEncoderDecoder:
        model = rnn(d_model=input_size, nlayers=nlayers, d_hid=hidden_size, target_length=output_size, device=device)
    elif rnn == simple_rnn:
        model = rnn(input_size=input_size, hidden_size=hidden_size, output_size=output_size, history_size=histo_length)
    else:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
                     seq_length=seq_length, output_size=output_size)
    return model

def plot_multiple_curve_losses(model_names, save_path=None):
    """
    args:
    -----
        model_names: list of paths to curve losses csv files containing the 
                    fields: [epoch, valid_loss, train_loss]
                    max number of model = 10
        save_path: string path to save the plot
                   if None, no saving

    return:
    -------
        best: dictonnary with the name of the best model as key and the best epoch as value

    """
    assert len(model_names) < 10

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = colors[:len(model_names)]


    best = {}

    linewidth = 2.5
    labelsize = 20

    plt.figure(figsize=(16,12))
    plt.rc('xtick', labelsize=labelsize) 
    plt.rc('ytick', labelsize=labelsize) 
    for color, file in zip(colors, model_names):
        df = pd.read_csv(file)
        ep = df[["epoch"]]
        val = df[["valid_loss"]]
        best_e = df["epoch"].iloc[df["valid_loss"].argmin()]
        train =  df[["train_loss"]]
        model_name = file[8:-4] # remove results/ and .csv
        best[model_name] = best_e
        plt.plot(ep, train, color=color, linestyle="solid", linewidth=linewidth, label=model_name + " Train Loss")
        plt.plot(ep, val, color=color, linestyle="dashed", linewidth=linewidth, label=model_name + " Validation Loss")
        
        # print(df.columns)
        
    plt.xlabel("Epoch", fontsize="xx-large")
    plt.ylabel("MSE", fontsize="xx-large")
    plt.ylim((0.0, 0.16))
    plt.grid()
    plt.legend(prop={'size': 20})    
    if save_path:
        plt.savefig(save_path, dpi=200)

    return best

def split_name_hidden_size(string):
    """
    args:
    -----
        string: formated path to csv following "results/{model}_{cell}_{loss}_{hidden_size}.csv" or "*_{hidden_size}"
    return:
    -------
        model_name: str name corresponding to *
        hidden_size: int corresponding to hiddden_size
    """
    if string[:8] == "results/" and string[-4:] == ".csv":
        string = string[8:-4]
    ls = string.split('_')
    hidden_size = int(ls[-1])
    loss = ls[-2]
    cell = ls[-3]


    ls = ls[:-3]
    model_name = ""
    for i, w in enumerate(ls):
        if i == len(ls)-1:
            model_name += w 
        else:
            model_name += w + "_"
            
    return model_name, cell, loss, hidden_size

def get_wind_time(netCDF_file_path, location):
    """
    args:
    -----
        netCDF_file_path: String
                          netCDF path file containing time series of size (l,)
        location: tuple (k,j,i)
                  indices given in ferret indices (starting from 1)
    return:
    -------
        uuz: float numpy array of size (l,)
             wind speed according to x axis in (m/s)
        vvz: float numpy array (l,)
             wind speed according to y axis in (m/s)
        time: float numpy array (l,)
              minutes elasped from a given date start_time
        start_time: String
                    ISO date time
        
    """
    with netcdf.netcdf_file(netCDF_file_path, 'r') as f:
        # Attention use "- 1" due to the shift between ferret and python
        # k is a special case sometimes 3 levels sometimes 2
        k = f.variables['UUZ'][0,:,0,0].shape[0] - 1 # 80 m 
        j = location[1] - 1 # j wind farm
        i = location[2] - 1 # i wind farm
        
        # get time series in numpy array
        uuz = f.variables['UUZ'][:,k,j,i].copy()
        vvz = f.variables['VVZ'][:,k,j,i].copy()
        
        time = f.variables['TIME'][:].copy()
        
        # Get only the ISO date from "minutes since YYYY-MM-DD HH:mm:ss"
        # + convert from bytes to String
        start_time = f.variables['TIME'].units.decode()[14:]
        
        f.close()
    
    return uuz, vvz, time, start_time


def convert_min_to_date(start_time, elapsed_minutes):
    """
    Method to convert a given number of minutes elapsed from a
    given start_time date into a datetime object
    e.g.: 
    convert_min_to_date(start_time='2016-09-01 00:00:00', elapsed_minutes=4.0)
    datetime.datetime(2016, 9, 1, 0, 4)

    args:
    -----
        start_time: String 
                    ISO format date time e.g. "2016-09-01 00:00:00"
        elapsed_minutes: int or float
                         number of minutes elapsed from start time
    return:
    -------
        now_date: Datetime object
                  corresponding to start_time + elapsed_minutes
    """
    start_time = datetime.datetime.fromisoformat(start_time)
    delta = datetime.timedelta(minutes=int(elapsed_minutes))
    
    now_date = start_time + delta
    
    return now_date

def convert_date_to_min(date):
    """
    Method to convert a given number of minutes elapsed from a
    given start_time date into a datetime object
    e.g.:
    convert_date_to_min(date='2016-09-01 00:00:00')
    1060138080.0
    
    args:
    -----
        date: String 
                    ISO format date time e.g. "2016-09-01 00:00:00"

    return:
    -------
        elapsed_minutes: int
                         number of minutes elapsed from datetime.datetime.min
    """
    min_date = datetime.datetime.min
    elapsed_minutes = (datetime.datetime.fromisoformat(date) \
                      - min_date).total_seconds() / 60.0
        
    return elapsed_minutes

def convert_date_to_hour(date):
    """
    Method to convert a given number of minutes elapsed from a
    given start_time date into a datetime object
    e.g.:
    convert_date_to_min(date='2016-09-01 00:00:00')
    1060138080.0
    
    args:
    -----
        date: String 
                    ISO format date time e.g. "2016-09-01 00:00:00"

    return:
    -------
        elapsed_minutes: int
                         number of hours elapsed from datetime.datetime.min
    """
    min_date = datetime.datetime.min
    elapsed_minutes = (datetime.datetime.fromisoformat(date) \
                      - min_date).total_seconds() / 60.0*60.0
        
    return elapsed_minutes

def convert_hours_to_min(date):
    """
    Method to convert a given number of minutes elapsed from a
    given start_time date into a datetime object
    e.g.:
    convert_date_to_min(date='2016-09-01 00:00:00')
    1060138080.0
    
    args:
    -----
        date: String 
                    ISO format date time e.g. "2016-09-01 00:00:00"

    return:
    -------
        elapsed_minutes: int
                         number of minutes elapsed from 00:00 
    """
    min_date = datetime.datetime.min
    elapsed_minutes = (datetime.datetime.fromisoformat(date) \
                      - min_date).total_seconds() / 60.0
        
    return elapsed_minutes


def create_dataframe(path_ores_data, path_mar_data):
    """
    args:
    -----
        ores_data: DataFrame 
                   with power production from windfarms
        mar_data: String
                  netCDF path file
    return:
    -------
        dataset: DataFrame
                 containing nbr_wind_farms * 4 fields (UUZ, VVZ, KVA, datetime)
    """
    
    # Open csv file from ORES
    paths_corrupted = ["data/ORES/export_eolien_2021-02-01.csv", 
                       "data/ORES/export_eolien_2021-03-01.csv", 
                       "data/ORES/export_eolien_2021-04-01.csv"]

    names = ["coord", "rated installed power (kVA)", 
             "Contractual max power (kVA)", "kW", "Date-Time"]

    ores_df = pd.read_csv(path_ores_data, header=None, names=names)

    # Correction DataFrame for corrupted files
    if path_ores_data in paths_corrupted:
        ores_df = ores_df.set_index(np.arange(0,len(ores_df),1))
        ores_df['coord'][ores_df['coord'] == 6.198469] = "50.386792,6.198469"
        ores_df['coord'][ores_df['coord'] == 3.64753] = "50.58274,3.64753"
        ores_df['coord'][ores_df['coord'] == 4.575570] = "50.53690,4.57557"
    
    prod_per_wind_farm = ores_df.pivot(index="coord", 
                                       columns="Date-Time", 
                                       values="kW")

    power_installed = ores_df.sort_values(by="coord",
                        axis=0)['rated installed power (kVA)'].unique()
    
    locations = [(2,31,61), (2,17,27), (2,23,39)] # (k,j,i)
    
    # Init list to contain all the data
    dataset = []
    
    for i, location in enumerate(locations):
        print("(i, location) = ({}, {})".format(i, location))
        time_serie = prod_per_wind_farm.iloc[i] # Get prod for i th wind farm

        uuz, vvz, time, start_time = get_wind_time(netCDF_file_path=\
                                                                path_mar_data, 
                                                   location=location)

        time = [convert_min_to_date(start_time, elapsed_minutes) 
                for elapsed_minutes in time]

        # DataFrame
        data = {"uuz":uuz, "vvz": vvz, "time":time}
        netcdf_df = pd.DataFrame(data)

        # Interpolation part
        # Wind Speed X
        points = [convert_date_to_min(date=str(date)) \
                    for date in netcdf_df['time'][:]]
        values = netcdf_df['uuz'][:]

        interp_x = [convert_date_to_min(date=date[:-1]) 
                    for date in time_serie.index.values]

        interp_speedX = np.interp(interp_x, points, values)

        # Wind Speed Y
        points = [convert_date_to_min(date=str(date)) \
                    for date in netcdf_df['time'][:]]
        values = netcdf_df['vvz'][:]

        interp_y = [convert_date_to_min(date=date[:-1]) 
                    for date in time_serie.index.values]

        interp_speedY = np.interp(interp_y, points, values)


        # Norm Wind Speed
        norm_ws = np.sqrt(np.square(interp_speedX) + np.square(interp_speedY))

        # Angle Wind Speed
        angle_ws = np.arctan2(interp_speedY, interp_speedX)

        # Dataset Creation
        dataset = dataset + [("prod_wf{}".format(i), 
                              time_serie.values / power_installed[i]), 
                             ("windSpeedNorm{}".format(i), norm_ws),
                             ("windSpeedAngle{}".format(i), angle_ws)]

    dataset = dataset + [("time", time_serie.index.values)]

    # convert list into dictionary to construct the DataFrame   
    dataset = {key:value for key, value in dataset}
        
    df = pd.DataFrame(dataset)

    return df

def create_dataset(vervose=True, normalize=True):
    """
    Routine to create a dataset from netcdf and ORES files

    All the paths must be given in the global variables "path_ores"
    and "path_mar" in the head of this file

    """

    frames = []

    for path_ores, path_mar in zip(paths_ores, paths_mar):
        print(path_ores, path_mar)
        
        print("Length frame: ", len(frames))
        try:
            dataset = create_dataframe(path_ores_data=path_ores, 
                                         path_mar_data=path_mar)

            frames.append(dataset)

            
            for i in range(3):
                correlation = np.corrcoef(dataset['windSpeedNorm{}'.format(i)], 
                                          dataset['prod_wf{}'.format(i)])

                spearman = stats.spearmanr(dataset['windSpeedNorm{}'.format(i)], 
                                           dataset['prod_wf{}'.format(i)])

                if vervose:
                    print("Wind farm ", i)
                    print("Correlation between wind speed and production:\n", 
                          correlation)
                    print("Spearman coefficient between wind speed and \
                          production:\n", 
                          spearman.correlation)
                    print("\n")
            
        except:
            print("Error with ", paths_ores, path_mar)

    if len(frames) > 1:
        dataset = pd.concat(frames)
    else:
        dataset = frames[0]

    dataset = dataset.drop_duplicates(subset = "time")

    # Add sin and cosinus minutes of a day
    vector_seconds = [datetime.timedelta(hours=int(ele[11:-7]), 
                      minutes=int(ele[14:-4]), 
                      seconds=int(ele[17:-1])).total_seconds() \
                      for ele in dataset["time"] ]

    f_sin = lambda ele: np.sin(2*np.pi*(ele)/(24*60*60))
    f_cos = lambda ele: np.cos(2*np.pi*(ele)/(24*60*60))

    dataset["sin_" + "time"] = [f_sin(ele) for ele in vector_seconds]
    dataset["cos_" + "time"] = [f_cos(ele) for ele in vector_seconds]

    # Normalizing Data
    if normalize:
        for i in range(3):
            mean_ws = dataset['windSpeedNorm{}'.format(i)].mean()
            std_ws = dataset['windSpeedNorm{}'.format(i)].std()

            dataset['windSpeedNorm{}'.format(i)] = (dataset['windSpeedNorm{}'.format(i)] - \
                                            mean_ws) / std_ws

                    
            dataset['prod_wf{}'.format(i)] = (dataset['prod_wf{}'.format(i)] - \
                dataset['prod_wf{}'.format(i)].mean()) / dataset['prod_wf{}'.format(i)].std()

        # Save dataset
        dataset.to_csv("data/dataset.csv", index=False)
        print("Dataset saved")
    else:
        # Save dataset
        dataset.to_csv("data/dataset_nn.csv", index=False)
        print("Dataset saved")


def feature_label_split(df, window_size, forecast_size, 
                        features, add_forecast=False, verbose=False):
    """
    
    args:
    -----
        df: DataFrame
        window_size: int
                     Size of the input window
        forecast_size: int
                       Size of the forecasted window
        features: list of Strings 
                  containing the name of the features
        add_forecast: boolean
                      if True add forecast windSpeed in the features
        verbose: bool
                 if True print added
    return:
    -------
        X: features matrix of size (n_samples, window_size, n_features)
        y: target values of size (n_samples, forecast_size)
    
    """
    
    
    tot_window = window_size + forecast_size
    
    assert tot_window < len(df), "Error"
    
    step = 1
    
    n_samples = len(df) // step
    
    X = np.zeros((n_samples, window_size, len(features)+1 if add_forecast 
                                          else len(features)))
    y = np.zeros((n_samples, forecast_size))
    
    for i in range(0, len(df)-tot_window, step):
        if i % 100000 == 0 and verbose:
            print(round(i/(len(df)-tot_window),2))
            
        features_values = []
        
        for feature in features:
            features_values.append(df[feature][i:i+window_size].values)
            
        if add_forecast:
            forecast = np.zeros(window_size)
            forecast[-forecast_size:] = df["windSpeedNorm0"][i+window_size:i+window_size+forecast_size].values
            features_values.append(forecast)
            
        features_values = np.array(features_values).T
            
        X[i // step,:,:] = features_values
        y[i // step,:] = df["prod_wf0"]\
                           [i+window_size:i+window_size+forecast_size].values
        
    
    return X, y


def get_random_split_dataset(df, num_bins=50, percentage=0.2, window_size=120, 
                             forecast_size=60, 
                             features=["prod_wf0", "windSpeedNorm0", 
                                       "sin_time", "cos_time"],
                            add_forecast=False):
    """
    args:
    -----
        df: DataFrame 
            with power production from windfarms
        num_bins: int
                  Number of splits made in the dataset
        percentage: float
                    proportion of bins to use for the validation and test set
        window_size: int
                     size of the input window
        forecast_size: int
                       size of the forecast window
        features: list of Strings 
                  containing the name of the features
        add_forecast: boolean
                      if True add forecast windSpeed in the features
    return:
    -------
        split_data: Dictionary
                    containing the fields X_train, X_valid, X_test which 
                    correspond to 3D numpy array with input windows of size:
                    (n_samples, window_size, n_features)
                    
                    and
                    containing y_train, y_valid, y_test which correspond 
                    to 2D numpy array of the forecasting window of size:
                    (n_samples, forecast_size)

                    
    """
    random.seed(0)

    train_set = []
    valid_set = []
    test_set = []

    for i in range(int(0.5 * percentage * num_bins)):
        tmp = random.randint(0,num_bins-1)
        if tmp not in valid_set:
            valid_set.append(tmp)
    
    i=0
    while i < int(0.5 * percentage * num_bins):
        tmp = random.randint(0,num_bins-1)
        
        if tmp not in valid_set and tmp not in test_set:
            test_set.append(tmp)
            i+=1
    
    test_valid_set = valid_set + test_set
    train_set = [ele for ele in range(num_bins) if ele not in test_valid_set]
    
    # endpoint=False allows to not have the last indices of the dataframe
    indices = np.linspace(0, len(df), num_bins, endpoint=False).astype(int)
        
    train_set = indices[train_set]
    valid_set = indices[valid_set]
    test_set = indices[test_set]
    
    # Second Part to create the window datasets
    n_features = len(features)
    train_size = int((1 - percentage) * len(df))
    
    X_train = None
    y_train = None
    
    X_valid = None
    y_valid = None
    
    X_test = None
    y_test = None
    
    for i in range(num_bins):
        print("Part {} over {}".format(i, num_bins))
        
        if i == num_bins - 1:
            df_to_feature = df.iloc[indices[i]:]
        else:
            df_to_feature = df.iloc[indices[i]:indices[i+1]]
        X, y = feature_label_split(df=df_to_feature, 
                                   window_size=window_size, 
                                   forecast_size=forecast_size, 
                                   features=features,
                                   add_forecast=add_forecast) 

            
        if indices[i] in train_set:
            if X_train is None and y_train is None:
                X_train = X
                y_train = y
            else:
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)
        elif indices[i] in valid_set:
            if X_valid is None and y_valid is None:
                X_valid = X
                y_valid = y
            else:
                X_valid = np.concatenate((X_valid, X), axis=0)
                y_valid = np.concatenate((y_valid, y), axis=0)
        else:
            if X_test is None and y_test is None:
                X_test = X
                y_test = y
            else:
                X_test = np.concatenate((X_test, X), axis=0)
                y_test = np.concatenate((y_test, y), axis=0)
    
    # Problem with the last 180 items I do not catch the error yet
    error = window_size + forecast_size
    split_data = {"X_train":X_train[:-error,:,:], "y_train":y_train[:-error], 
                  "X_valid":X_valid[:-error,:,:], "y_valid":y_valid[:-error],
                  "X_test":X_test[:-error,:,:], "y_test":y_test[:-error]}
    
    return split_data


def load_split_dataset(path="data/data.pkl"):
    """ Load the split dataset """
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data

def write_split_dataset(data, path="data/data.pkl"):
    """ Save the split dataset """
    with open(path, "wb") as f:
        pickle.dump(data, f)

def plot_curve_losses(df, save_path=None):
    """ 
    Plot the curve losses 

    args:
    -----
        - df: DataFrame
              Containing the fields: "train_loss" and "valid_loss"
        - save_path: String
                     Path where to save the figure

    """

    plt.figure(figsize=(7,4))
    n_epoch = len(df)
    plt.xticks(range(0, n_epoch+1, n_epoch//10))
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["valid_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()
    plt.savefig(save_path, dpi=200)
    # plt.show()

def plot_results(model, X, y, save_path=None, sklearn=False, show=False, transformer_with_decoder=False):
    """ 
    Plot predicted results

    args:
    -----
        - model : torch model
        - X: input model
        - y: true forecast
        - save_path: String
                     Path where to save the figure
        - sklearn: boolean
                    whether it is a sklearn model or not 
        - show: boolean 
        - transformer_with_decoder: boolean
    Returns:
    --------
        - y_pred: the forecasted values

    """

    if sklearn:
        y_pred = model.predict(X)

    else:

        with torch.no_grad():
            X = X.unsqueeze(0)
            if transformer_with_decoder:
                y_pred = model.predict(X).squeeze(0)
            else:
                y_pred = model(X).squeeze(0)

            y, y_pred = y.cpu(), y_pred.cpu()
            
            # print(F.mse_loss(y_pred, y))
        
    plt.figure(figsize=(6,4))
    plt.plot(y, label="True values" )
    plt.plot(y_pred, label="Predicted values" )
    plt.grid()
    plt.xlabel("timestamps")
    plt.ylabel("Production Power Normalized")
    plt.ylim((0,1))
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()

    plt.close()

    return y_pred

def pinball_loss(d, f, alpha):
    return max(alpha*(d-f), (1-alpha)*(f-d))

# Define RMSE metric
def rmse(y, yhat):
    l = MeanSquaredError(square_root=True)
    return l(y, yhat)

def big_dataset(new_df, type_data, gap=0, farm=0):
    """
    Creates a big dataset at each 96 time steps with a forecasting horizon of 
    96 steps.

    args:
    -----
        new_df: Pandas DataFrame with 
                'histoWindSpeedNorm0_80', 'histoWindSpeedAngle0_80',
                'histoTemperature0_80', 'histoWindSpeedNorm0_100',
                'histoWindSpeedAngle0_100', 'histoTemperature0_100'
                'windSpeedNorm0_80', 'windSpeedAngle0_80', 'temperature0_80',
                'windSpeedNorm0_100', 'windSpeedAngle0_100', 'temperature0_100'
                'prod_wf0'
                as columns
        type_data: string either "train", "valid" or "test"
        farm: integer in {0,1,2}

    return:
    -------
        X_histo: numpy array of size (num_samples, history_size, num_histo_features)
        X_forecast: numpy array of size (num_samples, forecast_horizon, num_forecast_features)
        y: numpy array of size (num_samples, forecast_horizon)
    """    
    verbose = False

    histo_features = ['histoWindSpeedNorm0_80', 'histoWindSpeedAngle0_80',
                      'histoTemperature0_80', 'histoWindSpeedNorm0_100',
                      'histoWindSpeedAngle0_100', 'histoTemperature0_100',
                      f'prod_wf0']
    history_size = 96
    num_histo_features = len(histo_features)

    forecast_features = ['windSpeedNorm0_80', 'windSpeedAngle0_80', 'temperature0_80',
                         'windSpeedNorm0_100', 'windSpeedAngle0_100', 'temperature0_100']
    num_forecast_features = len(forecast_features)
    forecast_horizon = 96

    num_samples= len(new_df) - history_size - forecast_horizon - gap

    X_histo = np.empty((num_samples, history_size, num_histo_features))
    X_forecast = np.empty((num_samples, forecast_horizon+gap, num_forecast_features))

    y = np.empty((num_samples, forecast_horizon))

    for t in range(num_samples):
        if not t % 1000:
            print(f"{t} over {num_samples}")
        histo = new_df[histo_features].iloc[t:history_size+t]# .values.reshape((96*6))
        X_histo[t,:,:] = histo
        forecast = new_df[forecast_features].iloc[history_size+t:history_size+gap+t+forecast_horizon].values # .reshape((6*(i+1)))
        X_forecast[t,:,:] = forecast

        y[t,:] = new_df["prod_wf0"].iloc[history_size+gap+t:history_size+gap+t+forecast_horizon]

        if verbose:
            print("X histo", new_df.iloc[t:history_size+t])
            print("X forecast", new_df.iloc[history_size+t:history_size+gap+t+forecast_horizon])
            print("y", new_df.iloc[history_size+gap+t:history_size+gap+t+forecast_horizon])
            # print("histo", histo.shape, type(histo))
            # print("forecast", forecast.shape, type(forecast))
    
    np.save(f"data/output15/X{farm}_big_{type_data}_histo_{history_size}_gap_{gap}.npy" , X_histo)
    np.save(f"data/output15/X{farm}_big_{type_data}_forecast_{forecast_horizon}_gap_{gap}.npy" , X_forecast)
    np.save(f"data/output15/y{farm}_big_{type_data}_{forecast_horizon}_gap_{gap}.npy", y)
    

    if verbose:
        print("X_histo", X_histo.shape)
        print("X_forecast", X_forecast.shape)
        print("y", y.shape)


    return X_histo, X_forecast, y



def small_dataset(new_df, type_data, gap=0, farm=0, save=True, gefcom=False):
    """
    Creates a small dataset at each 96 time steps with a forecasting horizon of 
    96 steps.

    args:
    -----
        new_df: Pandas DataFrame with 
                'histoWindSpeedNorm0_80', 'histoWindSpeedAngle0_80',
                'histoTemperature0_80', 'histoWindSpeedNorm0_100',
                'histoWindSpeedAngle0_100', 'histoTemperature0_100'
                'windSpeedNorm0_80', 'windSpeedAngle0_80', 'temperature0_80',
                'windSpeedNorm0_100', 'windSpeedAngle0_100', 'temperature0_100'
                'prod_wf0'
                as columns
        type_data: string either "train", "valid" or "test"
        farm: integer in {0,1,2}

    return:
    -------
        X_histo: numpy array of size (num_samples, history_size, num_histo_features)
        X_forecast: numpy array of size (num_samples, forecast_horizon+gap, num_forecast_features)
        y: numpy array of size (num_samples, forecast_horizon)
    """    
    
    
    if gefcom:
        new_df = new_df[new_df["ZONEID"] == farm]
        new_df = create_HYD(new_df)
        history_size = 24
        forecast_horizon = 24
        histo_features = ['U10', 'V10',
                      'U100', 'V100',
                      f'TARGETVAR']
        forecast_features = ['U10', 'V10','U100', 'V100',]
    else:
        history_size = 96
        forecast_horizon = 96
        histo_features = ['histoWindSpeedNorm0_80', 'histoWindSpeedAngle0_80',
                          'histoTemperature0_80', 'histoWindSpeedNorm0_100',
                          'histoWindSpeedAngle0_100', 'histoTemperature0_100',
                          f'prod_wf0']
        forecast_features = ['windSpeedNorm0_80', 'windSpeedAngle0_80', 'temperature0_80',
                         'windSpeedNorm0_100', 'windSpeedAngle0_100', 'temperature0_100']
        
    new_df.reset_index(inplace=True)
    skip_half_day = new_df[new_df["HOUR"] == 12].index[0]
    # print(new_df[new_df["HOUR"] == 12].index)
    # print("skip_half_day", skip_half_day)
    
    
    num_histo_features = len(histo_features)
    
    num_forecast_features = len(forecast_features)
    
    num_samples= (len(new_df)//(history_size)) - 2 
    # print(num_samples)
    get_date = ['YEAR', 'DAYOFYEAR', 'HOUR']

    # for regressor t+1 only 6 forecast
    # for regressor t+n 6*n forecasts
    
    target_name = "TARGETVAR" if gefcom else "prod_wf0"

    X_histo = np.empty((num_samples, history_size, num_histo_features))
    X_forecast = np.empty((num_samples, forecast_horizon+gap, num_forecast_features))

    y = np.empty((num_samples, forecast_horizon))

    for t in range(num_samples):
        histo = new_df[histo_features].iloc[skip_half_day+t*history_size:skip_half_day+(t+1)*history_size]
        # print("histo", new_df[get_date].iloc[skip_half_day+t*history_size:skip_half_day+(t+1)*history_size])
        try:
            X_histo[t,:,:] = histo
        
            forecast = new_df[forecast_features].iloc[skip_half_day+(t+1)*history_size:skip_half_day+(t+1)*history_size+(gap+forecast_horizon)]
            # print("forecast", new_df[get_date].iloc[skip_half_day+(t+1)*history_size:skip_half_day+(t+1)*history_size+(gap+forecast_horizon)])
            X_forecast[t,:,:] = forecast
            # print("y", new_df[get_date].iloc[skip_half_day+(t+1)*history_size+gap:skip_half_day+(t+1)*history_size+gap+forecast_horizon])
            y[t] = new_df[target_name].iloc[skip_half_day+(t+1)*history_size+gap:skip_half_day+(t+1)*history_size+gap+forecast_horizon]

        except:
            print(t, num_samples)
    
    if save:
        prefix = "data/gefcom/" if gefcom else "data/output15/" 
        np.save(prefix + f"X{farm}_small_{type_data}_histo_{history_size}_gap_{gap}.npy" , X_histo)
        np.save(prefix + f"X{farm}_small_{type_data}_forecast_{forecast_horizon}_gap_{gap}.npy" , X_forecast)
        np.save(prefix + f"y{farm}_small_{type_data}_{forecast_horizon}_gap_{gap}.npy", y)

    return X_histo, X_forecast, y


def get_dataset_sklearn(quarter, farm=0, type_data="train", gap=0, history_size=96, forecast_horizon=96, size="big", gefcom=False):
    """
    args:
        quarter: integer between 0 and 95

    returns:
    --------
        X: features matrix with history and forecast variable
        y: target vector for quarter quarter
    """
    prefix = "data/gefcom/" if gefcom else "data/output15/" 
    
    histo = np.load(prefix + f"X{farm}_{size}_{type_data}_histo_{history_size}_gap_{gap}.npy")
    forecast = np.load(prefix + f"X{farm}_{size}_{type_data}_forecast_{forecast_horizon}_gap_{gap}.npy")

    histo = histo.reshape((histo.shape[0], histo.shape[1]*histo.shape[2]))
    forecast =  forecast[:,:gap+quarter,:].reshape(forecast.shape[0], (gap+quarter)*forecast.shape[2])

    X = np.concatenate([histo, forecast], axis=1)

    y = np.load(prefix + f"y{farm}_{size}_{type_data}_{forecast_horizon}_gap_{gap}.npy")
    print("y", y.shape)
    y = y[:,quarter]

    return X, y


def get_dataset_rnn(quarter, farm=0, type_data="train", gap=0, history_size=96, forecast_horizon=96, size="big", tensor=False, gefcom=False):
    """
    
    """
    prefix = "data/gefcom/" if gefcom else "data/output15/" 
    histo = np.load(prefix + f"X{farm}_{size}_{type_data}_histo_{history_size}_gap_{gap}.npy")
    forecast = np.load(prefix + f"X{farm}_{size}_{type_data}_forecast_{forecast_horizon}_gap_{gap}.npy")
    y = np.load(prefix + f"y{farm}_{size}_{type_data}_{forecast_horizon}_gap_{gap}.npy")
    
    forecast = forecast[:,:gap+quarter,:]
    if forecast.shape[2] != histo.shape[2]:
        forecast = np.concatenate([forecast, np.zeros((forecast.shape[0], 
                                                       forecast.shape[1], 1))], 
                                  axis=2)
    # print("histo", histo.shape)
    # print("forecast", forecast.shape)
    X = np.concatenate([histo, forecast], axis=1)
    
    if tensor:
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()
    else:
        return X, y


def split_df(df, split=0.9):
    """
    Split into training and test set 

    args:
    -----
        df: Pandas DataFrame with 
        split: float between 0.0 and 1.0
               Portion to allocate to the training set
    return:
    -------
        df_train: 
        df_valid:
        df_test: 
    """
    split_index0 = int(len(df) * split)

    split_index1 = split_index0 + (len(df) - split_index0) // 2

    df_train = df.iloc[:split_index0]
    df_valid = df.iloc[split_index0:split_index1]
    df_test = df.iloc[split_index1:]

    return df_train, df_valid, df_test


def load_sklearn_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        model = pickle.load(f)

    return model

def simple_plot(forecast, truth, periods=24, save=None):
    x = np.arange(0,periods)
    plt.figure(figsize=(6,4))
    plt.plot(truth, label="True values" )
    plt.plot(forecast, label="Predicted values" )
    plt.grid()
    plt.xlabel("Timestamps")
    plt.ylabel("Production Power Normalized")
    plt.ylim((0,1))
    plt.legend()
    if save:
        plt.savefig(save, dpi=200)

    plt.close()
    # plt.show()

def to_bin(ele, nbins=15):
    """
    args:
    -----
        ele: float in [0, 1]
    output:
    -------
        return: the corresponding bin by dividing the interval [0, 1]
                in nbins
        

    """
    assert ele >= 0 and ele <= 1.0
    for i in range(1,nbins+1):
        if ele < i*(1/nbins):
            return i

def create_HYD(df):
    df["ISO"] = df["TIMESTAMP"].apply(f)
    df["ISO"] = df["ISO"].apply(datetime.datetime.fromisoformat)
    df["YEAR"]      = df["ISO"].map(lambda x: x.year)
    df["DAYOFYEAR"] = df["ISO"].map(lambda x: x.timetuple().tm_yday)
    df["HOUR"]      = df["ISO"].map(lambda x: x.hour)
    return df

def f(date):
    ymd = date[:4] + "-" + date[4:6] + "-" + date[6:8] + " "
    if len(date) == 13:
        h = "0" + date[-4:]
        return ymd + h
    else:
        return ymd + date[-5:]

