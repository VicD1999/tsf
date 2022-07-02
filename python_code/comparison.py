
import pandas as pd
import argparse
import time
import os

import util as u
from model import LSTM, GRU, BRC, nBRC, simple_rnn, architecture, architecture_history_forecast, history_forecast, HybridRNN
from attention import Attention_Net
from transformer import Transformer, TransformerEncoderDecoder

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch

from pytorch_forecasting.metrics import RMSE, MAE, SMAPE

rnns = u.rnns


# Data Loading
quarter = 96
history_size=96 
forecast_horizon=96
size="small"
gap=48
farm=0
device = torch.device('cpu')
batch_size = 16
# X_train, y_train = u.get_dataset_rnn(day=day, farm=farm, type_data="train", gap=gap, 
#                            history_size=history_size, forecast_horizon=forecast_horizon, size="small")
X_valid, y_valid = u.get_dataset_rnn(quarter=quarter, farm=farm, 
                                     type_data="valid", gap=gap, 
                                     history_size=history_size, 
                                     forecast_horizon=forecast_horizon, 
                                     size=size, tensor=True)
X_test, y_test = u.get_dataset_rnn(quarter=quarter, farm=farm, 
                                     type_data="test", gap=gap, 
                                     history_size=history_size, 
                                     forecast_horizon=forecast_horizon, 
                                     size=size, tensor=True)


val_set_len = X_valid.shape[0]
seq_length = X_valid.shape[1]
input_size = X_valid.shape[2]

val_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

model_names = ['results/history_forecast_510.csv', 
               # 'results/architecture_history_forecast_256.csv',
               'results/simple_rnn_512.csv',
               'results/architecture_512.csv']


best = u.plot_multiple_curve_losses(model_names, save_path="results/figure/rnn_curve_losses.pdf")

model_names = list(map(u.split_name_hidden_size, model_names)) 

print(best)
print(model_names)

rnn, hidden_size = model_names[0]

model = u.init_model(rnns[rnn], input_size, hidden_size, seq_length, output_size=forecast_horizon, 
                    gap_length=gap, histo_length=history_size, nhead=input_size, nlayers=num_layers, device=device, cell_name="BRC")



# for model_name in model_names:



# model = rnns[rnn](input_size=input_size, hidden_size=args.hidden_size, 
#                        seq_length=seq_length, output_size=args.forecast_size)
# model.load_state_dict(torch.load(f"model/{model_name}/{model_name}.model", map_location=device), strict=False)

# mean_loss_valid = 0
# # Validation Loss
# losses_valid = None
# with torch.no_grad():
#     for x_batch, y_batch in val_loader:
#         x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#         output = model(x_batch)
#         # loss_valid = (output_i - y_batch_i)^2
#         # is a 2D Tensor
#         loss_valid = F.mse_loss(output, y_batch, reduction="none")
#         # mse_tensor is one vector containing the mse of
#         # each sample in the batch
#         mse_tensor = torch.mean(loss_valid, dim=1)

#         # losses_valid contains the mse of each batch already
#         # done
#         losses_valid = mse_tensor if losses_valid is None else \
#                        torch.cat((losses_valid, mse_tensor), dim=0)

# # Get the mean of all the MSE
# mean_loss_valid = torch.mean(losses_valid, dim=0)

# # Get the standard deviation of all the MSE
# std_loss_valid = torch.std(losses_valid, dim=0)

# print(model_name)
# print(f"mean +- std: {mean_loss_valid} +- {std_loss_valid}")
        
"""
plt.figure(figsize=(7,5))
plt.bar(model_names, mean_loss_valid, yerr = std_loss_valid)
"""