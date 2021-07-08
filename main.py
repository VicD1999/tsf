# Author: Victor Dachet
from scipy import stats
import numpy as np

import util as u

path_ores_data = "data/ORES/export_eolien_2021-06-01.csv"
path_mar_data="data/MAR/concat_20210430_20210531.nc"

dataset = u.create_dataframe(path_ores_data=path_ores_data, 
                             path_mar_data=path_mar_data)

for i in range(3):
    print("Wind farm ", i)

    correlation = np.corrcoef(dataset[f'windSpeedNorm{i}'], dataset[f'prod_wf{i}'])

    print("Correlation between wind speed and production:\n", correlation)

    spearman = stats.spearmanr(dataset[f'windSpeedNorm{i}'], dataset[f'prod_wf{i}'])

    print("Spearman coefficient between wind speed and production:\n", 
          spearman.correlation)

    print("\n")
