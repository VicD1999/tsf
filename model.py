# Author: Victor Dachet inspired by Gaspard Lambrechts

import torch
import torch.nn as nn
import torch.nn.init as init

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, seq_length, output_size, batch_first=True):
        super(LSTM, self).__init__()
        
        batch_first = True
        input_size = input_size
        hidden_size = hidden_size
        num_layers = num_layers
        seq_length = seq_length
        output_size = output_size
        
        # output size = (N, L, D * H_out)
        # N = batch size
        # L = sequence length
        # D = 2 if bidirectional=True otherwise 1
        # H_out = hidden_size
        
        # output shape = (batch_size, input_size, num_features)
        # input_size = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=batch_first)
        
        self.fc = nn.Linear(hidden_size * seq_length, output_size)
        
        
    def forward(self, x):
        
        # With no starting parameter h0, c0 are init to zero
        output, (hn, cn) = self.lstm(x)
        
        # To go from (N, L, D * H_out) to (N, L * D * H_out) 
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        
        
        return output