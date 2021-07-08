# Author: Victor Dachet

import numpy as np
from scipy.io import netcdf
import pandas as pd
import datetime

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
        k = location[0] - 1 # 80 m
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
    names = ["coord", "rated installed power (kVA)", 
             "Contractual max power (kVA)", "kW", "Date-Time"]

    ores_df = pd.read_csv(path_ores_data, header=None, names=names)
    
    prod_per_wind_farm = ores_df.pivot(index="coord", 
                                       columns="Date-Time", 
                                       values="kW")
    
    locations = [(2,31,61), (2,17,27), (2,23,39)] # (k,j,i)
    
    dataset = []
    
    for i, location in enumerate(locations):
        print(f"(i, location) = ({i}, {location})")
        time_serie = prod_per_wind_farm.iloc[i] # Get prod from the 1st wind farm

        uuz, vvz, time, start_time = get_wind_time(netCDF_file_path=path_mar_data, 
                                                   location=location)

        time = [convert_min_to_date(start_time, elapsed_minutes) 
                for elapsed_minutes in time]

        # DataFrame
        data = {"uuz":uuz, "vvz": vvz, "time":time}
        df = pd.DataFrame(data)

        # Interpolation part
        # Wind Speed X
        points = [convert_date_to_min(date=str(date)) for date in df['time'][:]]
        values = df['uuz'][:]

        interp_x = [convert_date_to_min(date=date[:-1]) 
                    for date in time_serie.index.values]

        interp_speedX = np.interp(interp_x, points, values)

        # Wind Speed Y
        points = [convert_date_to_min(date=str(date)) for date in df['time'][:]]
        values = df['vvz'][:]

        interp_y = [convert_date_to_min(date=date[:-1]) 
                    for date in time_serie.index.values]

        interp_speedY = np.interp(interp_y, points, values)


        # Norm Wind Speed
        norm_ws = np.sqrt(np.square(interp_speedX) + np.square(interp_speedY))

        # Dataset Creation
        dataset = dataset + [(f"prod_wf{i}",time_serie.values), 
                             (f"windSpeedNorm{i}", norm_ws),
                             (f"time{i}", time_serie.index.values)]
        
    dataset = {key:value for key, value in dataset}
        
    df = pd.DataFrame(dataset)
    
    return df

