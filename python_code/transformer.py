import torch
import torch.nn as nn

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset

import matplotlib.pyplot as plt


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    """
    Positional encoding class add sin and cos features to the matrix
    """

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 300, device: str = None):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.Tensor([-math.log(10_000.0) / d_model]))

        self.sin = torch.sin(position * div_term).to(device)
        self.cos = torch.cos(position * div_term).to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, features_dim]

        Output:
            x: Tensor, shape [batch_size, seq_len, features_dim+2]
        """

        cat1 = self.sin[:x.shape[1]].unsqueeze(0).expand(x.shape[0], x.shape[1], 1)
        cat2 = self.cos[:x.shape[1]].unsqueeze(0).expand(x.shape[0], x.shape[1], 1)

        x = torch.cat((x, cat1, cat2), dim=2)


        return x

    def __len__(self):
        """
        Output the number of features added
        """
        return 2

class Transformer(nn.Module):
    """
    TransformerEncoder Class
    Composed of an encoding part as in https://arxiv.org/abs/1706.03762
    Then the encoded sequence is given to a little MLP

    """

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 target_length: int=96,
                 device: str=None):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model, device=device)
        self.d_model = d_model + len(self.pos_encoder)

        # Encoder Trasnformer
        encoder_layers = TransformerEncoderLayer(self.d_model, self.d_model, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)      

        # MLP as Decoder  
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
                                      nn.ReLU(),
                                      nn.Linear(self.d_model*16, 1),
                                      nn.Sigmoid())

        self.target_length = target_length


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, num_fatures]

        Returns:
            output Tensor of shape [batch_size, output_len]
        """
        # print("src before sqrt", src.shape)
        # src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src) # [batch_size, seq_len, num_fatures+2]
        # print("AFTER pos_encoder", src.shape)
        # print("src", src.shape)
        # print("src_mask", src_mask.shape)
        output = self.transformer_encoder(src) # [batch_size, seq_len, num_fatures+2]
        output = self.decoder(output) # [batch_size, seq_len, 1]
        return output[:,-self.target_length:,0] # [batch_size, target_length]


class TransformerEncodeDecoder(nn.Module):
    """
    Class from: https://medium.com/analytics-vidhya/implementing-transformer-from-scratch-in-pytorch-8c872a3044c9
    Classic Transformer that both encodes and decodes.
    
    Prediction-time inference is done greedily.
    NOTE: start token is hard-coded to be 0, end token to be 1. If changing, update predict() accordingly.
    """

    def __init__(self, nlayers: int, d_hid: int, 
                       max_output_length: int=300, 
                       d_model: int = 7, 
                       target_length: int=96, 
                       device: str=None):
        super().__init__()

        # Parameters
        
        self.d_hid = d_hid
        self.max_output_length = max_output_length
        self.device = device
        self.target_length = target_length
        
        self.pos_encoder = PositionalEncoding(d_model=d_model, device=device)
        self.d_model = d_model + len(self.pos_encoder)
        nhead = self.d_model


        # Encoder part
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=self.d_hid, batch_first=True),
            num_layers=nlayers
        )

        # Decoder part
        self.y_mask = generate_square_subsequent_mask(self.max_output_length).to(device)
        # print("mask", self.y_mask)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=self.d_hid, batch_first=True),
            num_layers=nlayers
        )
        self.fc = nn.Linear(self.d_model, 1)

        # It is empirically important to initialize weights properly
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def view(self, x):
        x, first_token, forecast = x[:,:-self.target_length-1,:], x[:,-self.target_length-1,:], x[:,-self.target_length:,:]
        return x, first_token, forecast

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx, E)
            y: (B, Sy)
        Output
            (B, Sy)
        """
        x, first_token, forecast = self.view(x)
        # print("forecast before", forecast)
        forecast[:,:,-1] = y 
        forecast = torch.cat((first_token.unsqueeze(1), forecast), dim=1)
        # print("forecast", forecast.shape)
        encoded_x = self.encode(x)  # (B, Sx, E)
        print("encoded_x should be (B, Sx, E)", encoded_x.shape)
        output = self.decode(forecast, encoded_x)  # (B, Sy+1)
        print("output", output.shape)
        return output[:,-self.target_length:] # (B, C, Sy)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (Sx, B, E) embedding
        """
        x = self.pos_encoder(x)  # (B, Sx, E)
        # print("pos_encoder must be (B, Sx, E)", x.shape)
        x = self.transformer_encoder(x)  # (B, Sx, E)
        # print("Transformer encoder (B, Sx, E)", x.shape)
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """
        Input
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (Sy, B, C) logits
        """
        y = self.pos_encoder(y)  # (B, Sy, E)
        # print("y (B, Sy, E)", y.shape)
        Sy = y.shape[1]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (B, Sy, E)
        output = self.fc(output)  # (B, Sy, C)
        return output.squeeze(2)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.
        Input
            x: (B, Sx, E)
        Output
            (B, Sy)
        """

        x, first_token, forecast = self.view(x)
        # print("forecast before", forecast)
        
        # print("forecast", forecast.shape)
        encoded_x = self.encode(x)  # (Sx, B, E)
        # print("encoded_x should be (B, Sx, E)", encoded_x.shape)
        # forecast[:,:,-1] = y 
        for Sy in range(self.target_length):
            y = torch.cat((first_token.unsqueeze(1), forecast[:,:Sy,:]), dim=1)
            # print("y", y.shape)
            output = self.decode(y, encoded_x)  # (Sy, B, C)
            # print("output", output.shape)
            # print("forecast[:,Sy,-1]", forecast[:,Sy,-1].shape)
            forecast[:,Sy,-1] = output[:,-1]
        # print("output", output.shape)
        return output # (B, C, Sy)


if __name__ == "__main__":
    print(torch.__version__)

    import util as u
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(0)

    day=96
    farm=0
    gap=48
    history_size=96
    forecast_horizon=96
    X_train, y_train = u.get_dataset_rnn(day=day, farm=farm, type_data="train", gap=gap, 
                               history_size=history_size, forecast_horizon=forecast_horizon, size="small")
    X_valid, y_valid = u.get_dataset_rnn(day=day, farm=farm, type_data="valid", gap=gap, 
                               history_size=history_size, forecast_horizon=forecast_horizon, size="small")

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    train = TensorDataset(X_train[:2,:,:], y_train[:2,:])

    train_loader = DataLoader(train, batch_size=2, shuffle=True, drop_last=False)


    # model = Transformer(d_model=7, nhead=7, d_hid=2048, nlayers=10, dropout=0.5, target_length=96, device=device)
    # model = Transformer_enc_dec(d_model=7, nhead=9, d_hid=2048,
    #                             nlayers=10, dropout=0.2, device=device)
    model = TransformerEncodeDecoder(d_model=7, nlayers=4, d_hid=128, max_output_length=300, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for e in range(0, 500):
        mean_loss = 0.

        losses_train = None
        for x_batch, y_batch in train_loader:
            # print(x_batch.shape)
            # print(y_batch.shape)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch, y_batch)
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
            output = model.predict(x_batch)
            loss = F.mse_loss(output, y_batch)
        print("loss", loss.item())
    # print("output", output.shape)

    torch.save(model.state_dict(), 
               "model/test.model")
            



    # forecasting = True
    # if forecasting:
    #     # ENNCODER DECODER PARAMETERS
    #     # seq_length = 240
    #     # batch_size = 8
        
    #     # head_dim = 32
    #     # d_model = nhead = 7 # nhead * head_dim
    #     # d_hid= 2048 * 2
    #     # nlayers=8
    #     # dropout=0.5
    #     # model = Transformer_enc_dec(d_model=d_model, nhead=nhead, d_hid=d_hid,
    #     #                     nlayers=nlayers, dropout=dropout)

    #     # TRANSFORMER PARAMETERS
    #     # seq_length = 240
    #     # batch_size = 8
        
    #     # head_dim = 32
    #     # d_model = nhead = 7 # nhead * head_dim
    #     # d_hid= 4096 + 512
    #     # nlayers= 10
    #     # dropout=0.5

    #     # # model = TransformerModel(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=0.2)
    #     # model = TransformerModelWithoutMask(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=0.2)
                
    #     seq_length = 240
    #     batch_size = 2
        
    #     head_dim = 32
    #     d_model = nhead = 7 # nhead * head_dim
    #     d_hid= 4096 + 512
    #     nlayers= 6
    #     dropout=0.5

    #     model = PytorchTransf(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=0.2)

    #     x = torch.rand((batch_size, 96+48, d_model))
    #     y = torch.rand((batch_size, 96, d_model))
    #     print("x", x.shape)
    #     # Uncomment for TransformerModel and TransformerEncDec
    #     x = torch.rand((batch_size, seq_length, d_model))
    #     src_mask = generate_square_subsequent_mask(seq_length) # .to(device)
    #     print("src_mask", src_mask.shape)
    #     output = model(x)
    #     print("output", output.shape)

    #     # output = model(x, y)
    #     # print("output", output.shape)
    #     torch.save(model.state_dict(), "model/transformer.model")
    #     # print("model:", model)
        
    # else:
    #     seq_length = 240
    #     batch_size = 8
        
    #     d_model = nhead = 7 # nhead * head_dim
        
    #     model = TemporalFusionTransformer()
    #     x = torch.rand((batch_size, seq_length, d_model))
    #     output = model(x)
