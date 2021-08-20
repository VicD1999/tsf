# Author: Victor Dachet

import numpy as np
from scipy.io import netcdf
import pandas as pd
import datetime
import random
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

paths_ores = ["data/ORES/export_eolien_2021-02-01.csv",
              "data/ORES/export_eolien_2021-03-01.csv",
              "data/ORES/export_eolien_2021-04-01.csv",
              "data/ORES/export_eolien_2021-05-01.csv",
              "data/ORES/export_eolien_2021-06-01.csv",
              "data/ORES/export_eolien_2021-07-01.csv"]

paths_mar = ["data/MAR/concat_20210118_20210131.nc", # 02
             "data/MAR/concat_20210201_20210228.nc", # 03
             "data/MAR/concat_20210228_20210331.nc", # 04
             "data/MAR/concat_20210331_20210430.nc", # 05
             "data/MAR/concat_20210430_20210531.nc", # 06
             "data/MAR/concat_20210531_20210630.nc"] # 07

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

def create_dataset(vervose=True):
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


    dataset = pd.concat(frames)

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
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["valid_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()
    plt.savefig(save_path, dpi=200)
    plt.show()

def plot_results(model, X, y, save_path=None):
    """ 
    Plot predicted results

    args:
    -----
        - model : torch model
        - X: input model
        - y: true forecast
        - save_path: String
                     Path where to save the figure

    """
    with torch.no_grad():
        X = X.unsqueeze(0)

        y_pred = model(X).squeeze(0)
        
        print(F.mse_loss(y_pred, y))
        
    plt.figure(figsize=(6,4))
    plt.plot(y, label="True values" )
    plt.plot(y_pred, label="Predicted values" )
    plt.grid()
    plt.xlabel("timestamps")
    plt.ylabel("Production Power Normalized")
    plt.legend()
    plt.savefig(save_path, dpi=200)
    plt.show()


