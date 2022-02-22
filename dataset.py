# Script to create wind time series from netcdf files for each wind farms
# Author: Victor Dachet

from scipy.io import netcdf
import os
import glob
from util import *

wind_time_serie = False
power_time_serie = True
interpolation = True

locations = [(2,31,61), (2,17,27), (2,23,39)] # (k,j,i)
 
################################################################
#                      Wind time series                        #
################################################################
if wind_time_serie:

    files_list = glob.glob("data/MAR/concat*.nc")
    files_list = sorted(files_list)

    ls = []
    time_size = 0

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
                    # print("name", name)
                    if name[:4] == "TIME":
                        key = name
                        time = np.concatenate((time, var[:].copy() ))
                        time_size += time.shape[0]
                        print(var[0], time_size, netCDF_file_path)
                    
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
        df.to_csv(f"data/windFarm{num_wf}.csv", index=False)


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

            power_installed = ores_df.sort_values(by="coord",
                                axis=0)['rated installed power (kVA)'].unique()



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

