
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from prophet import Prophet

from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.forecasting.naive import NaiveForecaster

from util import get_dataset_rnn, split_df, rmse, simple_plot
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument('-g','--gefcom', help='Use gefcom dataset if not specified MAR dataset', 
                        action="store_true")
    # Model Evaluation args


    args = parser.parse_args()

    if args.gefcom:

        df = open_gefcom()

        train_set, valid_set, test_set = get_split_dataset(df=df)

        tmp = train_set.loc[train_set["ZONEID"] == 1]

        # create timeseries readable by fbprophet
        ts = pd.DataFrame({'ds':tmp.TIMESTAMP,'y':tmp.TARGETVAR})
        ts.head()

        rmse = MeanSquaredError(square_root=True)


        # Prophet

        priors = [0.5]
        periods = len(valid_set) # 24

        for prior in priors:

            model = Prophet(weekly_seasonality=True, changepoint_range=1,
                           changepoint_prior_scale=prior, yearly_seasonality=False)
            model.fit(ts)
            periods = len(valid_set) # 24
            future = model.make_future_dataframe(periods=periods, freq='H')
            forecast = model.predict(future)
            fig = model.plot(forecast)
            
            pred = forecast["yhat"][-periods:].values
            truth = valid_set["TARGETVAR"][:periods].values
            l = rmse(pred, truth)
            
            l_24 = rmse(pred[:24], truth[:24])
            
            simple_plot(forecast["yhat"][-periods:], valid_set["TARGETVAR"][:periods])

            print(f"Prior {prior} & {l} & {l_24}")

    else: # MAR DATASET
        # Climatology Forecaster
        farm = 0
        new_df = pd.read_csv(f"data/output15/dataset{farm}_15.csv")
        df_train, df_valid, df_test = split_df(new_df, split=0.8)

        fh = 96
        gap=48
        day = 95

        X, y = get_dataset_rnn(day, farm=0, type_data="train", gap=48, history_size=96, forecast_horizon=96)

        
        climatology_forecaster = NaiveForecaster(strategy="mean")
        climatology_forecaster.fit(df_train["prod_wf0"].values)

        y_pred = climatology_forecaster.predict(fh=[i for i in range(1,fh+1)]).ravel()

        losses = [rmse(y[i,:], y_pred) for i in range(y.shape[0])]

        print("Train set:")
        print(f"rmse: {np.mean(losses):.2f} \pm {np.std(losses):.2f} \nrmse normalized {np.mean(losses)/30_000:.2f} \pm {np.std(losses)/30_000:.2f}")
        
        persistance_forecaster = NaiveForecaster(strategy='last')
        persistance_forecaster.fit(y)
    
        pred = persistance_forecaster.predict(fh=[i for i in range(1,fh+1)])

        # for sample in range(X.shape[0]):
        y_pred = X[:,95,-1].reshape((-1,1)) * np.ones(shape=(X.shape[0], fh))
        losses_train = np.sqrt(np.mean(np.square(y_pred - y), axis=1))
        print("Train set persistance:")
        print(f"RMSE {np.mean(losses_train)/30_000:.2f} \pm {np.std(losses_train)/30_000:.2f}\n")

        X, y = get_dataset_rnn(day, farm=0, type_data="valid", gap=48, history_size=96, forecast_horizon=96)

        y_pred = climatology_forecaster.predict(fh=[i for i in range(1,fh+1)]).ravel()
        
        losses = [rmse(y[i,:], y_pred) for i in range(y.shape[0])]
        print("Valid set:")
        print(f"rmse: {np.mean(losses):.2f} \pm {np.std(losses):.2f} \nrmse normalized {np.mean(losses)/30_000:.2f} \pm {np.std(losses)/30_000:.2f}")

        best = np.argmin(losses)
        print(f"Best rmse: {losses[best]}")
        simple_plot(truth=y[best] ,forecast=y_pred, periods=96, save="Images/naive_best.png")

        worst = np.argmax(losses)
        print(f"Worse rmse: {losses[worst]}")
        simple_plot(truth=y[worst] ,forecast=y_pred, periods=96, save="Images/naive_worst.png")
        

        y_pred = X[:,95,-1].reshape((-1,1)) * np.ones(shape=(X.shape[0], fh))
        losses_valid = np.sqrt(np.mean(np.square(y_pred - y), axis=1))
        print("Valid set persistance:")
        print(f"RMSE {np.mean(losses_valid)/30_000:.2f} \pm {np.std(losses_valid)/30_000:.2f}\n")



