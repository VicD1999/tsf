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

        self.out = nn.Sequential(nn.Linear(hidden_size, output_size),
                                 nn.Sigmoid())
        
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


# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)

#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))

#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)

#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)

#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)



if __name__ == '__main__':
    # batch_size = 2
    # X_train, y_train = torch.rand((32, 120, 5))
    #  = torch.rand((32, 20))
    # train_set_len = X_train.shape[0]
    # train = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    
    # for x_batch, y_batch in train_loader:
    #     output = model(x_batch)
    #     print("output", output.shape)
    #     break


    import util as u
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(0)

    quarter=96
    farm=0
    gap=48
    history_size=96
    forecast_horizon=96
    X_train, y_train = u.get_dataset_rnn(quarter=quarter, farm=farm, type_data="train", gap=gap, 
                               history_size=history_size, forecast_horizon=forecast_horizon, size="small")
    # X_valid, y_valid = u.get_dataset_rnn(quarter=quarter, farm=farm, type_data="valid", gap=gap, 
    #                            history_size=history_size, forecast_horizon=forecast_horizon, size="small")

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    train = TensorDataset(X_train[:2,:,:], y_train[:2,:])

    train_loader = DataLoader(train, batch_size=2, shuffle=True, drop_last=False)


    # model = Transformer(d_model=7, nhead=7, d_hid=2048, nlayers=10, dropout=0.5, target_length=96, device=device)
    # model = Transformer_enc_dec(d_model=7, nhead=9, d_hid=2048,
    #                             nlayers=10, dropout=0.2, device=device)
    model = Attention_Net(input_size=7, hidden_size=64, output_size=forecast_horizon, seq_length=history_size+gap+forecast_horizon)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for e in range(0, 500):
        mean_loss = 0.

        losses_train = None
        for x_batch, y_batch in train_loader:
            # print(x_batch.shape)
            # print(y_batch.shape)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            output = output.to(device)
            loss = F.mse_loss(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train = F.mse_loss(output, y_batch, reduction="none")
            mse_tensor = torch.mean(loss_train, dim=1)
            losses_train = mse_tensor if losses_train is None else \
                                   torch.cat((losses_train, mse_tensor), dim=0)

        mean_loss = torch.mean(losses_train)

        print(f"epoch {e} loss {mean_loss}")

    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            output = model(x_batch)
            loss = F.mse_loss(output, y_batch)
        print("loss", loss.item())
    # print("output", output.shape)

    torch.save(model.state_dict(), 
               "model/test.model")