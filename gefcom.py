import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


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
    df.rename(columns={"Unnamed: 0":"t"})

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



if __name__ == '__main__':
    create_gefcom_dataset()

    df = open_gefcom()

    print(df['TARGETVAR'].isnull().sum().sum())

    train_set, valid_set, test_set = get_split_dataset(df=df)

    print(train_set)
    print()

    print(valid_set)
    print()

    print(test_set)
    print()


