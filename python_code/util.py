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

import torch
import torch.nn.functional as F

from sktime.performance_metrics.forecasting import MeanSquaredError

paths_ores = glob.glob("data/ORES/*.csv")
paths_ores.sort()

paths_mar = ["data/MAR/concat.nc"]

from model import LSTM, GRU, BRC, nBRC, simple_rnn, architecture, architecture_history_forecast, history_forecast, HybridRNN
from attention import Attention_Net
from transformer import Transformer, TransformerEncoderDecoder

rnns = {"attn":Attention_Net, "simple_rnn":simple_rnn, 
        "architecture":architecture, 
        "architecture_history_forecast":architecture_history_forecast,
        "history_forecast": history_forecast,
        "Transformer": Transformer,
        "TransformerEncoderDecoder":TransformerEncoderDecoder}

cells = {"BRC": BRC,
         "nBRC": nBRC,
         "GRU": torch.nn.GRU,
         "LSTM": torch.nn.LSTM,
         "HybridRNN": HybridRNN}

def init_model(rnn, input_size, hidden_size, seq_length, output_size, 
               gap_length, histo_length, nhead, nlayers, device, cell_name):
    
    if rnn == architecture:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
                     seq_length=seq_length, output_size=output_size,
                     gap_length=gap_length)
    elif rnn == architecture_history_forecast:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
            output_size=output_size, histo_length=histo_length, gap_length=gap)
    elif rnn == history_forecast:
        model = rnn(input_size=input_size, hidden_size=hidden_size, 
            output_size=output_size, histo_length=histo_length, gap_length=gap_length,
            rnn_cell=cells[cell_name])
    elif rnn == Transformer:
        model = rnn(d_model=input_size, nhead=nhead, d_hid=hidden_size, nlayers=num_layers, dropout=0.2, device=device)
    elif rnn == TransformerEncoderDecoder:
        model = rnn(d_model=input_size, nlayers=num_layers, d_hid=hidden_size, device=device)
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

    plt.figure(figsize=(8,5))
    for color, file in zip(colors, model_names):
        df = pd.read_csv(file)
        ep = df[["epoch"]]
        val = df[["valid_loss"]]
        best_e = df["epoch"].iloc[df["valid_loss"].argmin()]
        train =  df[["train_loss"]]
        model_name = file[8:-4] # remove results/ and .csv
        best[model_name] = best_e
        plt.plot(ep, train, color=color, linestyle="solid", label=model_name + " Train Loss")
        plt.plot(ep, val, color=color, linestyle="dashed", label=model_name + " Validation Loss")
        
        # print(df.columns)
        
    # plt.ylim((0.02, 0.04))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()    
    if save_path:
        plt.savefig(save_path, dpi=200)

    return best

def split_name_hidden_size(string):
    """
    args:
    -----
        string: formated path to csv following "results/*_{hidden_size}.csv" or "*_{hidden_size}"
    return:
    -------
        model_name: str name corresponding to *
        hidden_size: int corresponding to hiddden_size
    """
    if string[:8] == "results/" and string[-4:] == ".csv":
        string = string[8:-4]
    ls = string.split('_')
    hidden_size = int(ls[-1])

    ls = ls[:-1]
    model_name = ""
    for i, w in enumerate(ls):
        if i == len(ls)-1:
            model_name += w 
        else:
            model_name += w + "_"
            
    return model_name, hidden_size

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

def plot_results(model, X, y, save_path=None, sklearn=False, show=False, src_mask=None):
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
    Returns:
    --------
        - y_pred: the forecasted values

    """

    if sklearn:
        y_pred = model.predict(X)

    else:

        with torch.no_grad():
            X = X.unsqueeze(0)
            if src_mask is not None:
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

def small_dataset(new_df, type_data, gap=0, farm=0):
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
    new_df.reset_index(inplace=True)
    skip_half_day = new_df[new_df["HOUR"] == 12].index[0]
    # print(new_df[new_df["HOUR"] == 12].index)
    # print("skip_half_day", skip_half_day)
    history_size = 96
    forecast_horizon = 96
    histo_features = ['histoWindSpeedNorm0_80', 'histoWindSpeedAngle0_80',
                      'histoTemperature0_80', 'histoWindSpeedNorm0_100',
                      'histoWindSpeedAngle0_100', 'histoTemperature0_100',
                      f'prod_wf0']
    num_histo_features = len(histo_features)
    forecast_features = ['windSpeedNorm0_80', 'windSpeedAngle0_80', 'temperature0_80',
                         'windSpeedNorm0_100', 'windSpeedAngle0_100', 'temperature0_100']
    num_forecast_features = len(forecast_features)
    
    num_samples= (len(new_df)//(history_size)) - 2 
    # print(num_samples)
    get_date = ['YEAR', 'DAYOFYEAR', 'HOUR', 'MIN']

    # for regressor t+1 only 6 forecast
    # for regressor t+n 6*n forecasts

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
            y[t] = new_df["prod_wf0"].iloc[skip_half_day+(t+1)*history_size+gap:skip_half_day+(t+1)*history_size+gap+forecast_horizon]

        except:
            print(t, num_samples)

    np.save(f"data/output15/X{farm}_small_{type_data}_histo_{history_size}_gap_{gap}.npy" , X_histo)
    np.save(f"data/output15/X{farm}_small_{type_data}_forecast_{forecast_horizon}_gap_{gap}.npy" , X_forecast)
    np.save(f"data/output15/y{farm}_small_{type_data}_{forecast_horizon}_gap_{gap}.npy", y)
    

    return X_histo, X_forecast, y


def get_dataset_sklearn(quarter, farm=0, type_data="train", gap=0, history_size=96, forecast_horizon=96, size="big"):
    """
    args:
        quarter: integer between 0 and 95

    returns:
    --------
        X: features matrix with history and forecast variable
        y: target vector for quarter quarter
    """
    histo = np.load(f"data/output15/X{farm}_{size}_{type_data}_histo_{history_size}_gap_{gap}.npy")
    forecast = np.load(f"data/output15/X{farm}_{size}_{type_data}_forecast_{forecast_horizon}_gap_{gap}.npy")

    histo = histo.reshape((histo.shape[0], histo.shape[1]*histo.shape[2]))
    forecast =  forecast[:,:gap+quarter,:].reshape(forecast.shape[0], (gap+quarter)*forecast.shape[2])

    X = np.concatenate([histo, forecast], axis=1)

    y = np.load(f"data/output15/y{farm}_{size}_{type_data}_{forecast_horizon}_gap_{gap}.npy")

    y = y[:,quarter]

    return X, y


def get_dataset_rnn(quarter, farm=0, type_data="train", gap=0, history_size=96, forecast_horizon=96, size="big", tensor=False):
    """
    
    """
    histo = np.load(f"data/output15/X{farm}_{size}_{type_data}_histo_{history_size}_gap_{gap}.npy")
    forecast = np.load(f"data/output15/X{farm}_{size}_{type_data}_forecast_{forecast_horizon}_gap_{gap}.npy")
    y = np.load(f"data/output15/y{farm}_{size}_{type_data}_{forecast_horizon}_gap_{gap}.npy")
    
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

