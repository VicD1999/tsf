# Author: Victor Dachet inspired by Gaspard Lambrechts

import torch
import torch.nn as nn
import torch.nn.init as init


class simple_rnn(nn.Module):
    """
    input x = (B,L,P)
    output y = (B,m)
    """
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(simple_rnn, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = nn.GRU(input_size-1, hidden_size, num_layers=1, batch_first=True)
        torch.nn.init.orthogonal_(self.encoder.weight_hh_l0)
        torch.nn.init.orthogonal_(self.encoder.weight_ih_l0)
        
        # self.f = nn.Tanh()
        self.f = nn.Sigmoid()

        self.decoder = nn.Sequential(nn.Linear(hidden_size+output_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, output_size))

        

                  
    def forward(self, x):
        y, h = self.encoder(x[:,:,:-1])

        # y[:,-1,:] = self.f(y[:,-1,:])
        # print("y", y[:,-1,:].shape)
        # print("x", x[:,:96,-1].shape)
        concat = torch.cat([y[:,-1,:], x[:,:96,-1]], axis=1)
        # print("concat", concat.shape)
        y = self.decoder(concat)

        y = self.f(y)
            
        return y

class architecture(nn.Module):
    """
    input x = (B,L,P)
    output y = (B,m)
    """
    def __init__(self, input_size, hidden_size, output_size, seq_length, gap_length):
        super(architecture, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gap_length = gap_length

        self.hidden_size = hidden_size

        self.encoder = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        torch.nn.init.orthogonal_(self.encoder.weight_hh_l0)
        torch.nn.init.orthogonal_(self.encoder.weight_ih_l0)
        
        self.decoder = nn.Sequential(nn.Linear(hidden_size+input_size-1, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, 1),
                                      nn.Sigmoid())

        

                  
    def forward(self, x):
        y, h = self.encoder(x[:,:96,:])

        y_hat = torch.empty((x.shape[0], self.gap_length+self.output_size))
        # print("y_hat", y_hat.shape)

        for i in range(self.gap_length+self.output_size):
            # print("y[:,-1,:]", y[:,-1,:].shape)
            # print("x[:,96+i,:-1]", x[:,96+i,:-1].shape)
            concat = torch.cat([y[:,-1,:], x[:,95+i,:-1]], axis=1)
            # print("concat", concat.shape)
            # print()
            y = self.decoder(concat)
            # print("y", y.shape)
            y_hat[:,i] = y[:, 0]
            
            # print("x[:,96+i,:-1]", x[:,96+i,:-1].shape)
            # print("y", y.shape)
            z_hat_y = torch.cat([x[:,95+i,:-1], y], axis=1).unsqueeze(1)
            # print("z_hat_y", z_hat_y.shape)
            y, h = self.encoder(z_hat_y, h)
            
            
        return y_hat[:,-self.output_size:]


class architecture_history_forecast(nn.Module):
    """
    input x = (B,L,P)
    output y = (B,m)
    """
    def __init__(self, input_size, hidden_size, output_size, histo_length, gap_length):
        super(architecture_history_forecast, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gap_length = gap_length
        self.histo_length = histo_length

        self.hidden_size = hidden_size

        self.encoder_histo = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        torch.nn.init.orthogonal_(self.encoder_histo.weight_hh_l0)
        torch.nn.init.orthogonal_(self.encoder_histo.weight_ih_l0)
        
        self.encoder_gap = nn.GRU(input_size-1, hidden_size, num_layers=1, batch_first=True)
        torch.nn.init.orthogonal_(self.encoder_gap.weight_hh_l0)
        torch.nn.init.orthogonal_(self.encoder_gap.weight_ih_l0)
        
        
        self.decoder_forecast = nn.GRU(input_size-1, hidden_size*2, num_layers=1, batch_first=True)
        
        self.decoder = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, 1),
                                      nn.Sigmoid())

        

                  
    def forward(self, x):
        y_histo, h_histo = self.encoder_histo(x[:,:self.histo_length,:])
        gap = x[:,self.histo_length:self.histo_length+self.gap_length,:-1]
        # print("gap", gap.shape)
        y_gap, h_gap = self.encoder_gap(gap)
        
        # print("h_histo", h_histo.shape)
        h = torch.cat([h_histo, h_gap], axis=2)
        # print("h", h.shape)
        
        y_hat = torch.empty((x.shape[0], self.output_size))
        # print("y_hat", y_hat.shape)

        for i in range(self.output_size):
            # print("x[:,self.histo_length+self.gap_length+i,:-1]", x[:,self.histo_length+self.gap_length+i,:-1].shape)
            z_hat = x[:,self.histo_length+self.gap_length+i-1,:-1].unsqueeze(1)
            # print("h", h.shape)
            y, h = self.decoder_forecast(z_hat, h)
            y_ = self.decoder(y[:,-1,:])
            # print("y_", y_.shape)
            # print("y_hat", y_hat.shape)
            y_hat[:,i] = self.decoder(y[:,-1,:]).squeeze(1)
            
            
        return y_hat

class history_forecast(nn.Module):
    """
    input x = (B,L,P)
    output y = (B,m)
    """
    def __init__(self, input_size, hidden_size, output_size, histo_length, gap_length, rnn_cell=nn.GRU):
        super(history_forecast, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gap_length = gap_length
        self.histo_length = histo_length

        self.hidden_size = hidden_size

        self.encoder_histo = rnn_cell(input_size, hidden_size, batch_first=True)
        
        self.encoder_gap = rnn_cell(input_size-1, hidden_size, batch_first=True)
        
        
        self.decoder_forecast = rnn_cell(input_size-1, hidden_size*2, batch_first=True)
        
        self.decoder = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, 1),
                                      nn.Sigmoid())

        

                  
    def forward(self, x):
        
        y_histo, h_histo = self.encoder_histo(x[:,:self.histo_length,:])
        gap = x[:,self.histo_length:self.histo_length+self.gap_length,:-1]
        # print("gap", gap.shape)
        y_gap, h_gap = self.encoder_gap(gap)
        
        # print("h_histo", h_histo.shape)
        # print("h_gap", h_gap.shape)
        h = torch.cat([h_histo, h_gap], axis=-1)
        # print("h", h.shape)
        
        y_hat = torch.empty((x.shape[0], self.output_size))
        # print("y_hat", y_hat.shape)

        for i in range(self.output_size):
            # print("x[:,self.histo_length+self.gap_length+i,:-1]", x[:,self.histo_length+self.gap_length+i,:-1].shape)
            z_hat = x[:,self.histo_length+self.gap_length+i-1,:-1].unsqueeze(1)
            # print("h", h.shape)
            y, h = self.decoder_forecast(z_hat)
            y_ = self.decoder(y[:,-1,:])
            # print("y_", y_.shape)
            # print("y_hat", y_hat.shape)
            y_hat[:,i] = self.decoder(y[:,-1,:]).squeeze(1)
            
            
        return y_hat


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
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(seq_length * hidden_size, output_size)
        
        
    def forward(self, x):
        # With no starting parameter h0, c0 are init to zero
        output, hn = self.gru(x)
                
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

    def __init__(self, input_size, hidden_size, batch_first=True, neuromodulated=False):
        super(BRC, self).__init__()
        
        self.batch_first = batch_first
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.neuromodulated = neuromodulated
        
        self.brc = nBRCLayer(input_size=input_size, hidden_size=hidden_size) \
                    if self.neuromodulated else \
                    BRCLayer(input_size=input_size, hidden_size=hidden_size)
                
        
    def forward(self, x):
        if self.batch_first:
            h0 = torch.zeros(x.shape[0], self.hidden_size)
            x = x.transpose(0,1)
        else:
            h0 = torch.zeros(x.shape[1], self.hidden_size)
            
        output, hn = self.brc(x, h0)
        
        # To revert (seq_len, batch_size, num_features) into
        # (batch_size, seq_len, num_features)
        if self.batch_first:
            output = output.transpose(0,1)

        return output, hn

class nBRC(BRC):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__(input_size, hidden_size, batch_first=batch_first, neuromodulated=True)

class HybridRNN(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first=True, num_layer=1):
        super(HybridRNN, self).__init__()
        hidden_size_actual = int(hidden_size / 2)

        self.memory_cell = nBRC(input_size, hidden_size_actual, batch_first=batch_first)
        self.transient_cell = nn.GRU(input_size, hidden_size_actual, num_layer, batch_first=batch_first)

    def forward(self, x, h0=None):

        if h0 is not None:
            raise NotImplementedError

        xm, hm = self.memory_cell(x)
        xt, yt = self.transient_cell(x)

        x = torch.cat((xm, xt), dim=-1)
        # print("yt", yt.shape)
        # print("hm", hm.shape)
        hn = torch.cat((hm, yt.squeeze(0)), dim=-1)

        return x, hn
        