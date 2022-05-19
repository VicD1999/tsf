from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    input x (B,L,P)
    output h (B,L,hidden_size), last_hidden_sate ()
    """
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        output, hidden = self.gru(inputs, hidden)
        
        return output, hidden

    def init_hidden(self, x):
        """
        x tensor of shape (B,L,num_features)
        """
        return torch.zeros(1, x.shape[0], self.hidden_size)


class Decoder(nn.Module):
    """
    input (x (B,L,P) , s_{i-1} (1,B,hidden_size) , out_enc (B,L,hidden_size) )
    output s_i (B, m or hidden_size)
    """
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, s_i_1):
        """
        x (B, hidden_size)
        s_i_1 = (1, 32, hidden_size)
        encoder_outputs = h
        """
        
        y_i, s_i = self.gru(x, s_i_1)

        y_i = self.out(y_i)
                
        return y_i, s_i
    
    def init_hidden(self, x):
        """
        x tensor of shape (B,L,num_features)
        """
        return torch.zeros(1, x.shape, self.hidden_size)


class Attention_Net(nn.Module):
    """
    input x = (B,L,P)
    output y = (B,m)
    """
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(Attention_Net, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, 1)
        
        self.attn = nn.Linear(hidden_size * 2, seq_length)
        
        self.out = nn.Linear(seq_length, output_size)
                  
    def forward(self, x):
        seq_length = x.shape[1]
        batch_size = x.shape[0]
        h, last_hidden = self.encoder(x, self.encoder.init_hidden(x))
        # h (B,L,hidden_size)
        # last_hidden (1,B,hidden_size)
        
        y = torch.empty((batch_size, seq_length))
        
        s_i_1 = last_hidden # s_0 is the hidden state of the last hidden encoder
        for i in range(seq_length):
            
            # Attention weights
            e_i = torch.empty((batch_size, seq_length)) # (B, L)
            for j in range(seq_length):
                h_j = h[:,j,:] # (B, hidden_size)
                # s_i_1 (1, B, hidden_size)
                # h_j (B, hidden_size)
                s_i_1_h_j = torch.cat((s_i_1.squeeze(0), h_j), dim=1) # (B, hidden_size * 2)
                
                e_ij = self.attn(s_i_1_h_j) # (B, 1)
                
                # print("e_ij", e_ij.shape)
                # print("e_i", e_i.shape)
                
                e_i[:,j] = e_ij[:,0] # e_i (B, L)
                
            alpha_i = F.softmax(e_i, dim=1) # (B, L)
            
            # context vector
            # h (B,L,hidden_size)
            # alpha_i (B, L)
            c_i = torch.bmm( alpha_i.unsqueeze(1), h ) # (B, hidden_size)

            y_i, s_i = self.decoder(c_i, s_i_1)
            # print("y_i", y_i.shape)
            y[:,i] = y_i[:,0,0]
            
            # Update s_i_1
            s_i_1 = s_i

        y = self.out(y)
            
            
        return y

if __name__ == '__main__':
    X_train = torch.rand((32, 120, 5))
    y_train = torch.rand((32, 20))
    train_set_len = X_train.shape[0]
    train = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    model = Attention_Net(input_size=5, hidden_size=64, output_size=20, seq_length=120)
    for x_batch, y_batch in train_loader:
        output = model(x_batch)
        print("output", output.shape)
        break