# Author: Victor Dachet
from scipy import stats
import numpy as np
import pandas as pd

import util as u

if __name__ == '__main__':
    # u.create_dataset(vervose=False)

    df = pd.read_csv("data/dataset.csv")

    data = u.get_random_split_dataset(df)

    u.write_split_dataset(data)
    
