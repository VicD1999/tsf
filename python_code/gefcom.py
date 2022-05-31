import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import util as u


def create_gefcom_dataset():
    dico = {}

    for zone in range(1,11):
        df = pd.DataFrame()
        for i in range(1,16):
            # print(i)
            df = pd.concat([df, pd.read_csv(f"GEFCom2014Data/Wind/Task {i}/Task{i}_W_Zone1_10/Task{i}_W_Zone{zone}.csv")], ignore_index=True)
            # print(df.iloc[-1])
            # print()

        df = df.drop_duplicates(subset=["TIMESTAMP"],  ignore_index=True)
            
        dico[zone] = df.copy()

    # COncatenation of the zones
    df = pd.concat([dico[i] for i in range(1,11)])
    df.fillna(method="ffill", inplace=True)

    df.to_csv("GEFCom2014Data/Wind/concat.csv")

def open_gefcom():
    df = pd.read_csv("GEFCom2014Data/Wind/concat.csv")
    df = df.rename(columns={"Unnamed: 0":"t"})

    return df

def get_split_dataset(df):
    """
    args: df: DataFrame 
    -----

    """

    split_train = "20130801 00:00"
    split_valid = "20131001 00:00"

    train_set = df[df["TIMESTAMP"] < split_train]
    condition = df["TIMESTAMP"] >=  split_train
    condition2 = df["TIMESTAMP"] < split_valid
    valid_set = df.loc[(condition & condition2)]
    test_set = df[df["TIMESTAMP"] >= split_valid]


    return train_set, valid_set, test_set

def preprocess(df):
    df["H100"] = np.sqrt(df["V100"]**2 + df["U100"]**2)
    df["H10"] = np.sqrt(df["V10"]**2 + df["U10"]**2)

    df["A10"] = np.rad2deg(np.arctan2(-df["U10"], -df["V10"]))
    df["A100"] = np.rad2deg(np.arctan2(-df["U100"], -df["V100"]))

    return df



if __name__ == '__main__':
    # create_gefcom_dataset()

    df = open_gefcom()

    print("df", df)

    print(df['TARGETVAR'].isnull().sum().sum())

    df_train, df_valid, df_test = get_split_dataset(df)

    # for farm in range():
    farm = 1
    print("Create dataset of farm", farm)
    
    
    # HERE It is a small dataset of only 385 days
    X_histo, X_forecast, y = u.small_dataset(df_train, type_data="train", gap=12, farm=1, save=True, gefcom=True)
    print("X_histo", X_histo.shape)
    print("X_forecast", X_forecast.shape)
    print("y", y.shape)

    X_histo, X_forecast, y = u.small_dataset(df_valid, type_data="valid", gap=12, farm=1, save=True, gefcom=True)
    print("X_histo", X_histo.shape)
    print("X_forecast", X_forecast.shape)
    print("y", y.shape)
    # print("y", y)


    X_histo, X_forecast, y = u.small_dataset(df_test, type_data="test", gap=12, farm=1, save=True, gefcom=True)
    print("X_histo", X_histo.shape)
    print("X_forecast", X_forecast.shape)
    print("y", y.shape)

