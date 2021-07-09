# Author: Victor Dachet
from scipy import stats
import numpy as np
import pandas as pd

import util as u


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

frames = []

for path_ores, path_mar in zip(paths_ores, paths_mar):
    print(path_ores, path_mar)
    
    print("Length frame: ", len(frames))
    try:
        dataset = u.create_dataframe(path_ores_data=path_ores, 
                                     path_mar_data=path_mar)

        frames.append(dataset)

        
        for i in range(3):
            print("Wind farm ", i)

            correlation = np.corrcoef(dataset[f'windSpeedNorm{i}'], 
                                      dataset[f'prod_wf{i}'])

            print("Correlation between wind speed and production:\n", correlation)

            spearman = stats.spearmanr(dataset[f'windSpeedNorm{i}'], 
                                       dataset[f'prod_wf{i}'])

            print("Spearman coefficient between wind speed and production:\n", 
                  spearman.correlation)

            print("\n")
        
    except:
        print("Error with ", paths_ores, path_mar)


dataset = pd.concat(frames)
dataset = dataset.drop_duplicates(subset = "time0")
dataset = dataset.drop_duplicates(subset = "time1")
dataset = dataset.drop_duplicates(subset = "time2")

# Save dataset
dataset.to_csv("data/dataset.csv")

    
