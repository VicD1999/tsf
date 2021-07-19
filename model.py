# Author: Victor Dachet inspired by Gaspard Lambrechts

import torch
import torch.nn as nn
import torch.nn.init as init

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, seq_length, output_size, num_layers=1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.output_size = output_size
        
        # output size = (N, L, D * H_out)
        # N = batch size
        # L = sequence length
        # D = 2 if bidirectional=True otherwise 1
        # H_out = hidden_size
        
        # output shape = (batch_size, input_size, num_features)
        # input_size = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(seq_length * hidden_size, output_size)
        
        
    def forward(self, x):
        # With no starting parameter h0, c0 are init to zero
        output, (hn, cn) = self.lstm(x)
                
        # To go from (N, L, D * H_out) to (N, L * D * H_out) 
        output = torch.flatten(output, start_dim=1)
        
        output = self.fc(output)
        
        return output


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, seq_length, output_size, num_layers=1):
        super(GRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.output_size = output_size
        
        # output size = (N, L, D * H_out)
        # N = batch size
        # L = sequence length
        # D = 2 if bidirectional=True otherwise 1
        # H_out = hidden_size
        
        # output shape = (batch_size, input_size, num_features)
        # input_size = seq_length
        self.lstm = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(seq_length * hidden_size, output_size)
        
        
    def forward(self, x):
        # With no starting parameter h0, c0 are init to zero
        output, (hn, cn) = self.lstm(x)
                
        # To go from (N, L, D * H_out) to (N, L * D * H_out) 
        output = torch.flatten(output, start_dim=1)
        
        output = self.fc(output)
        
        return output

# Gaspard Lambrechts' implementation
class BRCLayer(nn.Module):
    """
    Recurrent Neural Network (single layer) using the Bistable Recurrent Cell
    (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        self.w_c = nn.Parameter(init.normal_(torch.empty(hidden_size)))
        self.b_c = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        self.w_a = nn.Parameter(init.normal_(torch.empty(hidden_size)))
        self.b_a = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.normal_(torch.empty(hidden_size)))

    def forward(self, x_seq, h):
        """
        Compute the forward pass for the whole sequence.

        Arguments
        ---------
        - x_seq: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h: tensor of shape (batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        assert h.size(0) == x_seq.size(1)
        assert h.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                device=x_seq.device)

        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(torch.mm(x, self.U_c.T) + self.w_c * h +
                    self.b_c)
            a = 1. + torch.tanh(torch.mm(x, self.U_a.T) + self.w_a * h +
                    self.b_a)
            h = c * h + (1. - c) * torch.tanh(torch.mm(x, self.U_h.T) + a * h +
                    self.b_h)
            y_seq[t, ...] = h

        return y_seq, h

# Gaspard Lambrechts' implementation
class nBRCLayer(nn.Module):
    """
    Recurrent Neural Network (single layer) using the Recurrently
    Neuromodulated Bistable Recurrent Cell (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        W_c = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_c = nn.Parameter(W_c)
        self.b_c = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        W_a = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_a = nn.Parameter(W_a)
        self.b_a = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.normal_(torch.empty(hidden_size)))

    def forward(self, x_seq, h):
        """
        Compute the forward pass for the whole sequence.

        Arguments
        ---------
        - x_seq: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h: tensor of shape (batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        assert h.size(0) == x_seq.size(1)
        assert h.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                device=x_seq.device)

        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(torch.mm(x, self.U_c.T) +
                    torch.mm(h, self.W_c.T) + self.b_c)
            a = 1. + torch.tanh(torch.mm(x, self.U_a.T) +
                    torch.mm(h, self.W_a.T) + self.b_a)
            h = c * h + (1. - c) * torch.tanh(torch.mm(x, self.U_h.T) +
                    a * h + self.b_h)
            y_seq[t, ...] = h

        return y_seq, h

class BRC(nn.Module):

    def __init__(self, input_size, hidden_size, seq_length, output_size, neuromodulated=False):
        super(BRC, self).__init__()
        
        self.batch_first = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.neuromodulated = neuromodulated
        
        self.brc = nBRCLayer(input_size=input_size, hidden_size=hidden_size) \
                    if self.neuromodulated else \
                    BRCLayer(input_size=input_size, hidden_size=hidden_size)
        
        self.fc = nn.Linear(hidden_size * seq_length, output_size)
        
        
    def forward(self, x):
        h0 = torch.zeros(x.shape[0], self.hidden_size)
        x = x.transpose(0,1)

        output, hn = self.brc(x, h0)
        # To revert (seq_len, batch_size, num_features) into
        # (batch_size, seq_len, num_features)
        output = output.transpose(0,1)        

        # To go from (N, L, D * H_out) to (N, L * D * H_out) 
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)        
        
        return output

class nBRC(BRC):
    def __init__(self, input_size, hidden_size, seq_length, output_size):
        super().__init__(input_size, hidden_size, seq_length, output_size, neuromodulated=True)
        