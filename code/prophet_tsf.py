import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gefcom import open_gefcom, get_split_dataset
from prophet import Prophet
from sktime.performance_metrics.forecasting import MeanSquaredError

df = open_gefcom()
train_set, valid_set, test_set = get_split_dataset(df=df)
# Get Data for the 1st zone
df = train_set.loc[train_set["ZONEID"] == 1]

rmse = MeanSquaredError(square_root=True)

priors = [0.001, 0.01, 0.1, 0.5]
periods = len(valid_set) # 24

ts = pd.DataFrame({'ds':df.TIMESTAMP,'y':df.TARGETVAR})
ts.head()

for prior in priors:

    model = Prophet(weekly_seasonality=True, changepoint_range=1,changepoint_prior_scale=prior, yearly_seasonality=False)
    model.fit(ts)
    periods = len(valid_set) # 24
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    fig = model.plot(forecast)
    
    pred = forecast["yhat"][-periods:].values
    truth = valid_set["TARGETVAR"][:periods].values
    l = rmse(pred, truth)
    
    x = np.arange(0,periods)
    plt.figure()
    plt.plot(x, forecast["yhat"][-periods:], label="forecast")
    plt.plot(x, valid_set["TARGETVAR"][:periods], label="Truth")
    plt.legend()
    plt.show()

    print(f"Prior {prior} & {l}")