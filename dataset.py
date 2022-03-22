# Script to create wind time series from netcdf files for each wind farms
# Author: Victor Dachet

# from scipy.io import netcdf
import os
import glob
from sklearn.model_selection import TimeSeriesSplit
from util import *

wind_time_serie = False
power_time_serie = False
interpolation = False
interp_histo = False
dataset = False
output15 = False
create_sklearn_datasets = True

locations = [(2,31,61), (2,17,27), (2,23,39)] # (k,j,i)
 
################################################################
#                      Wind time series                        #
################################################################
if wind_time_serie:

    ################################################################
    #                      Wind time series FORECAST               #
    ################################################################
    files_list = glob.glob("data/MAR/concat*.nc")
    files_list = sorted(files_list)

    ls = []
    time_size = 0
    locations = [(2,31,61), (2,17,27), (2,23,39)] # (k,j,i)



    for num_wf, location in enumerate(locations):
        time = np.empty((0))
        
        uuz80 = np.empty((0))
        vvz80 = np.empty((0))
        ttz80 = np.empty((0))

        uuz100 = np.empty((0))
        vvz100 = np.empty((0))
        ttz100 = np.empty((0))
        
        for netCDF_file_path in files_list:
            new_netcdf = {}
            with netcdf.netcdf_file(netCDF_file_path, 'r') as f:
                k = f.variables['UUZ'][0,:,0,0].shape[0] - 1 # 80 m 
                j = location[1] - 1 # j wind farm
                i = location[2] - 1 # i wind farm

                for name, var in f.variables.items():
                    # new_netcdf[name] = var[:].copy()
                    # print("name", name, var.shape)
                    if name[:4] == "TIME" and name[-4:]!="bnds":
                        key = name
                        print("var", var.shape)
                        time = np.concatenate((time, var[:].copy() ))
                        time_size += time.shape[0]
                        # print(var[0], time_size, netCDF_file_path)
                    
                    if name == "UUZ":
                        # print(var[:,k,j,i].shape)
                        uuz80 = np.concatenate((uuz80, var[:,-2,j,i].copy()))
                        uuz100 = np.concatenate((uuz100, var[:,-1,j,i].copy()))
                    elif name == "VVZ": 
                        vvz80 = np.concatenate((vvz80, var[:,-2,j,i].copy()))
                        vvz100 = np.concatenate((vvz100, var[:,-1,j,i].copy()))
                    elif name == "TTZ":
                        ttz80 = np.concatenate((ttz80, var[:,-2,j,i].copy()))
                        ttz100 = np.concatenate((ttz100, var[:,-1,j,i].copy()))
                

                f.close()
                
        dico = {"TIME":time, "UUZ80":uuz80, "VVZ80":vvz80, "TTZ80": ttz80,"UUZ100":uuz100, "VVZ100":vvz100, "TTZ100": ttz100}
        df = pd.DataFrame(dico)
        df = df.drop_duplicates(subset=["TIME"],  ignore_index=True)
        df.to_csv(f"data/windFarm{num_wf}.csv", index=False)


    files_list = glob.glob("data/MAR/wind_farm*.nc")
    files_list = sorted(files_list)

    ls = []
    time_size = 0
    locations = [(2,31,61) , (2,17,27), (2,23,39)] # (k,j,i)

    ################################################################
    #                      Wind time series HISTORIC               #
    ################################################################

    for num_wf, location in enumerate(locations):
        time = np.empty((0))
        
        uuz80 = np.empty((0))
        vvz80 = np.empty((0))
        ttz80 = np.empty((0))

        uuz100 = np.empty((0))
        vvz100 = np.empty((0))
        ttz100 = np.empty((0))
        
        for netCDF_file_path in files_list:
            new_netcdf = {}
            with netcdf.netcdf_file(netCDF_file_path, 'r') as f:
                k = f.variables['UUZ'][0,:,0,0].shape[0] - 1 # 80 m 
                j = location[1] - 1 # j wind farm
                i = location[2] - 1 # i wind farm

                for name, var in f.variables.items():
                    # new_netcdf[name] = var[:].copy()
                    # print("name", name, var.shape)
                    if name[:4] == "TIME" and name[-4:]!="bnds":
                        key = name
                        print("var", var.shape)
                        time = np.concatenate((time, var[:].copy() ))
                        time_size += time.shape[0]
                        # print(var[0], time_size, netCDF_file_path)
                    
                    if name == "UUZ":
                        print(var.shape)
                        uuz80 = np.concatenate((uuz80, var[:,-2, 0, 0].copy()))
                        uuz100 = np.concatenate((uuz100, var[:,-1, 0, 0].copy()))
                    elif name == "VVZ": 
                        vvz80 = np.concatenate((vvz80, var[:,-2, 0, 0].copy()))
                        vvz100 = np.concatenate((vvz100, var[:,-1, 0, 0].copy()))
                    elif name == "TTZ":
                        ttz80 = np.concatenate((ttz80, var[:,-2, 0, 0].copy()))
                        ttz100 = np.concatenate((ttz100, var[:,-1, 0, 0].copy()))
                

                f.close()
                
        dico = {"TIME":time, "UUZ80":uuz80, "VVZ80":vvz80, "TTZ80": ttz80,"UUZ100":uuz100, "VVZ100":vvz100, "TTZ100": ttz100}
        df = pd.DataFrame(dico)

        df = df.drop_duplicates(subset=["TIME"],  ignore_index=True)
        df.to_csv(f"data/histoWindFarm{num_wf}.csv", index=False)


################################################################
#                      Wind POWER time series                  #
################################################################

if power_time_serie:

    files_list = glob.glob("data/ORES/export*.csv")
    files_list = sorted(files_list)

    paths_corrupted = ["data/ORES/export_eolien_2021-02-01.csv", 
                       "data/ORES/export_eolien_2021-03-01.csv", 
                       "data/ORES/export_eolien_2021-04-01.csv"]

    names = ["coord", "rated installed power (kVA)", 
             "Contractual max power (kVA)", "kW", "Date-Time"]

    ores_concat = None

    for path_ores_data in files_list:

        with open(path_ores_data, 'r') as f:
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
            
            ores_concat = pd.concat([ores_concat, prod_per_wind_farm], axis=1)
            
            print(prod_per_wind_farm.shape[1])

            power_installed = ores_df.sort_values(by="coord",
                                axis=0)['rated installed power (kVA)'].unique()
        ores_concat.interpolate(method='linear', axis=1, inplace=True)
        prod_per_wind_farm = ores_concat


################################################################
#                      INTERPOLATION                           #
################################################################

if interpolation:
    start_time="2016-09-01 00:00:00"

    for i, location in enumerate(locations):
        print("(i, location) = ({}, {})".format(i, location))

        dico = {}

        df_wind = pd.read_csv(f"data/windFarm{i}.csv")
        time_serie = prod_per_wind_farm.iloc[i] # Get prod for i th wind farm

        points = df_wind['TIME'][:]

        time = [convert_min_to_date(start_time, elapsed_minutes) 
                        for elapsed_minutes in df_wind["TIME"]]

        for altitude in ["80", "100"]:
            # INTERP X
            points = [convert_date_to_min(date=str(date)) for date in time[:]]

            values = df_wind['UUZ100'][:]

            interp_x = [convert_date_to_min(date=date[:-1]) 
                        for date in time_serie.index.values]

            interp_speedX = np.interp(interp_x, points, values)

            # INTERP Y
            values = df_wind['VVZ' + altitude][:]
            interp_y = [convert_date_to_min(date=date[:-1]) 
                        for date in time_serie.index.values]

            interp_speedY = np.interp(interp_y, points, values)

            # INTERP Temperature
            values = df_wind['TTZ' + altitude][:]
            interp_temp = [convert_date_to_min(date=date[:-1]) 
                           for date in time_serie.index.values]

            interp_temp = np.interp(interp_temp, points, values)

            norm_ws = np.sqrt(np.square(interp_speedX) + np.square(interp_speedY))

            # Angle Wind Speed
            angle_ws = np.arctan2(interp_speedY, interp_speedX)

            dico["windSpeedNorm0_" + altitude] = norm_ws
            dico["windSpeedAngle0_" + altitude] = angle_ws
            dico["temperature0_" + altitude] = interp_temp
        
        dico["time"] = interp_x
        dico["prod_wf0"] = time_serie.values
        df = pd.DataFrame(dico)
        df.to_csv(f"data/farm{i}.csv", index=False)
        
        print("Correlation with wind production")
        print("Temperature:")
        print(np.corrcoef(df["temperature0_80"], df["prod_wf0"]))
        print("Wind at 80 meters:")
        print(np.corrcoef(df["windSpeedNorm0_80"], df["prod_wf0"]))
        print("Wind at 100 meters:")
        print(np.corrcoef(df["windSpeedNorm0_100"], df["prod_wf0"]))

if interp_histo:
    start_time="2018-09-01 00:00:00"

    for i, location in enumerate(locations):
        print("(i, location) = ({}, {})".format(i, location))

        dico = {}

        df_wind = pd.read_csv(f"data/histoWindFarm{i}.csv")
        # print(df_wind)

        time_serie = prod_per_wind_farm.iloc[i] # Get prod for i th wind farm
        # print("time_serie", time_serie)
        points = df_wind['TIME'][:]

        time = [convert_min_to_date(start_time, elapsed_minutes) 
                        for elapsed_minutes in df_wind["TIME"]]

        for altitude in ["80", "100"]:
            # INTERP X
            points = [convert_date_to_min(date=str(date)) for date in time[:]]
            # print("points", len(points))

            values = df_wind['UUZ' + altitude][:]
            # print("values", values.shape)

            interp_x = [convert_date_to_min(date=date[:-1]) 
                        for date in time_serie.index.values]

            interp_speedX = np.interp(interp_x, points, values)


            # INTERP Y
            values = df_wind['VVZ' + altitude][:]
            interp_y = [convert_date_to_min(date=date[:-1]) 
                        for date in time_serie.index.values]

            interp_speedY = np.interp(interp_y, points, values)

            # INTERP Temperature
            values = df_wind['TTZ' + altitude][:]
            interp_temp = [convert_date_to_min(date=date[:-1]) 
                           for date in time_serie.index.values]

            interp_temp = np.interp(interp_temp, points, values)

            norm_ws = np.sqrt(np.square(interp_speedX) + np.square(interp_speedY))

            # Angle Wind Speed
            angle_ws = np.arctan2(interp_speedY, interp_speedX)

            dico["histoWindSpeedNorm0_" + altitude] = norm_ws
            dico["histoWindSpeedAngle0_" + altitude] = angle_ws
            dico["histoTemperature0_" + altitude] = interp_temp

        dico["time"] = interp_x
        dico["prod_wf0"] = time_serie.values
        df = pd.DataFrame(dico)
        df.to_csv(f"data/histo_farm_{i}.csv", index=False)

        print("Correlation with wind production")
        print("Temperature:")
        print(np.corrcoef(df["histoTemperature0_80"], df["prod_wf0"]))
        print("Wind at 80 meters:")
        print(np.corrcoef(df["histoWindSpeedNorm0_80"], df["prod_wf0"]))
        print("Wind at 100 meters:")
        print(np.corrcoef(df["histoWindSpeedNorm0_100"], df["prod_wf0"]))
        print()

if dataset:
    for i in range(3):
        df1 = pd.read_csv(f"data/histo_farm_{i}.csv")
        df2 = pd.read_csv(f"data/farm{i}.csv")
        df1 = df1.drop(["time"], axis=1) # df_marks.drop(['chemistry'], axis=1)
        df1 = df1.drop(["prod_wf0"], axis=1)
        df = pd.concat([df1, df2], axis=1)

        def f1(date):
            return convert_min_to_date(start_time=datetime.datetime.min.isoformat(), elapsed_minutes=date).isoformat()

        df["ISO"] = df["time"].apply(f1)
        df["ISO"] = df["ISO"].apply(datetime.datetime.fromisoformat)
        df["YEAR"]      = df["ISO"].map(lambda x: x.year)
        df["DAYOFYEAR"] = df["ISO"].map(lambda x: x.timetuple().tm_yday)
        df["HOUR"]      = df["ISO"].map(lambda x: x.hour)
        df["MIN"]       = df["ISO"].map(lambda x: x.minute)
        df.to_csv(f"data/dataset{i}.csv", index=False)
    
if output15:
    for i in range(3):
        df = pd.read_csv(f"data/dataset{i}.csv")
        print(len(df))
        resolution = 15
        # df.columns
        test = df[['histoWindSpeedNorm0_80', 'histoWindSpeedAngle0_80',
              'histoTemperature0_80', 'histoWindSpeedNorm0_100',
              'histoWindSpeedAngle0_100', 'histoTemperature0_100',
              'windSpeedNorm0_80', 'windSpeedAngle0_80', 'temperature0_80',
              'windSpeedNorm0_100', 'windSpeedAngle0_100', 'temperature0_100', 'prod_wf0']].rolling(window=resolution, min_periods=None, center=False, 
               win_type=None, on=None, axis=0, 
               closed=None, method='single').mean()
        # ls is a mask for 15min lines
        ls = [ bool(df["MIN"].iloc[i]%resolution == 0) for i in range(len(test)) ]
        ls[0] = False
        new_df = pd.concat([test.iloc[ls], df[["time", "YEAR", "DAYOFYEAR", "HOUR", "MIN"]].iloc[ls]], axis=1).reset_index()

        new_df.to_csv(f"data/output15/dataset{i}_15.csv", index=False)


def small_dataset(new_df, path_save_X=None, path_save_y=None):
    farm = 0 # AFTER: for farm in range(3):
    num_samples= len(new_df)//(96)

    i = 0 # for regressor t+1 only 6 forecast
          # for regressor t+n 6*n forecasts

    X = np.empty((num_samples, 96*6+6))

    y = np.empty((num_samples))

    for t in range(num_samples):

        histo = new_df[['histoWindSpeedNorm0_80', 'histoWindSpeedAngle0_80',
                          'histoTemperature0_80', 'histoWindSpeedNorm0_100',
                          'histoWindSpeedAngle0_100', 'histoTemperature0_100',]].iloc[t*96:(t+1)*96].values.reshape((96*6))
        forecast = new_df[['windSpeedNorm0_80', 'windSpeedAngle0_80', 'temperature0_80',
                           'windSpeedNorm0_100', 'windSpeedAngle0_100', 'temperature0_100']].iloc[(t+1)*96:96*(t+1)+1].values.reshape((6))
        
        concat = np.concatenate([histo,forecast], axis=0)
        # print("concat", concat.shape)
        X[t,:] = concat
        y[t] = new_df["prod_wf0"].iloc[(t+1)*96:96*(t+1)+1]

        # print("histo", histo.shape, type(histo))
        # print("forecast", forecast.shape, type(forecast))

    print("X", X.shape)
    print("y", y.shape)

    if path_save_X:
        np.save(path_save_X, X)
    if path_save_y:
        np.save(path_save_y, y)

    return X, y


if create_sklearn_datasets:
    big = True
    small = False

    farm = 0
    

    if small:
        # HERE It is a small dataset of only 385 days
        small_dataset(new_df=df_train, 
                    path_save_X=f"data/output15/X{farm}_small_train.npy",
                    path_save_y=f"data/output15/y{farm}_small_train.npy")

        small_dataset(new_df=df_valid, 
                    path_save_X=f"data/output15/X{farm}_small_valid.npy",
                    path_save_y=f"data/output15/y{farm}_small_valid.npy")

        small_dataset(new_df=df_test, 
                    path_save_X=f"data/output15/X{farm}_small_test.npy",
                    path_save_y=f"data/output15/y{farm}_small_test.npy")

    if big:
        for farm in range(1):
            print("Create dataset of farm", farm)
            new_df = pd.read_csv(f"data/output15/dataset{farm}_15.csv")
            df_train, df_valid, df_test = split_df(new_df, split=0.8)

            big_dataset(df_train, type_data="train", gap=48, farm=farm)
            big_dataset(df_valid, type_data="valid", gap=48, farm=farm)
            big_dataset(df_test, type_data="test", gap=48, farm=farm)

    """
    X, y = get_dataset_sklearn(day=7, farm=0, type_data="train", gap=48, history_size=96, forecast_horizon=96)

    print("X", X.shape)
    print("y", y.shape)

    X, y = get_dataset_rnn(day=7, farm=0, type_data="train", gap=48, history_size=96, forecast_horizon=96)

    print("X", X.shape)
    print("y", y.shape)
    """
