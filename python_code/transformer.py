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

# Inspired by the pytorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 target_length: int=96,
                 device: str=None):
        super().__init__()
        self.d_model = d_model * 2
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
                                      nn.ReLU(),
                                      nn.Linear(self.d_model*16, 1),
                                      nn.Sigmoid())

        self.target_length = target_length


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # print("src before sqrt", src.shape)
        src = src * math.sqrt(self.d_model)
        # print("src AFter sqrt", src.shape)
        # print("BEFORE pos_encoder", src)
        src = self.pos_encoder(src)
        # print("AFTER pos_encoder", src.shape)
        # print("src", src.shape)
        # print("src_mask", src_mask.shape)
        output = self.transformer_encoder(src, src_mask)
        # print("AFTER Transformer", output.shape)
        output = self.decoder(output)
        return output[:,:self.target_length,0]

class TransformerModelWithoutMask(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 target_length: int=96,
                 device: str=None):
        super().__init__()
        self.d_model = d_model * 2
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
                                      nn.ReLU(),
                                      nn.Linear(self.d_model*16, 1),
                                      nn.Sigmoid())

        self.target_length = target_length


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # print("src before sqrt", src.shape)
        src = src * math.sqrt(self.d_model)
        # print("src AFter sqrt", src.shape)
        # print("BEFORE pos_encoder", src)
        src = self.pos_encoder(src)
        # print("AFTER pos_encoder", src.shape)
        # print("src", src.shape)
        # print("src_mask", src_mask.shape)
        output = self.transformer_encoder(src)
        # print("AFTER Transformer", output.shape)
        output = self.decoder(output)
        return output[:,:self.target_length,0]

class Transformer_enc_dec(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 target_length: int=96,
                 device: str=None):
        super().__init__()
        self.device = device
        self.d_model = d_model * 2
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)


        self.y_mask = generate_square_subsequent_mask(100).to(self.device)
        decoder_layers = TransformerDecoderLayer(self.d_model, nhead, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        # self.encoder = nn.Embedding(ntoken, d_model)
        
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
                                      nn.ReLU(),
                                      nn.Linear(self.d_model*16, 1),
                                      nn.Sigmoid())

        self.target_length = target_length


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # print("src before sqrt", src.shape)
        src = src * math.sqrt(self.d_model)
        # print("src AFter sqrt", src.shape)
        # print("BEFORE pos_encoder", src)
        src = self.pos_encoder(src)
        # print("AFTER pos_encoder", src.shape)
        # print("src", src.shape)
        # print("src_mask", src_mask.shape)
        encoded_x = self.transformer_encoder(src, src_mask)
        # print("AFTER Transformer", encoded_x.shape)


        Sy = self.target_length # Should be 96
        
        y_i = -torch.ones((src.shape[0],14)).to(self.device)
        # print("y_i", y_i.device)
        y_i = y_i.unsqueeze(1)
        y_hat = torch.empty((src.shape[0], Sy))
        for i in range(Sy):
        # y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)
            y_mask = self.y_mask[:i+1, :i+1]
            # print("i", i)
            # print("y_i", y_i.shape)
            # print("y_mask", y_mask.shape)
            # print("encoded_x", encoded_x.device)
            # print("y_mask", y_mask.device)
            # print("y_i", y_i.device)
            y_tmp = self.transformer_decoder(y_i, encoded_x, y_mask)
            # print("y_tmp", y_tmp.shape)
            y_i = torch.cat([y_i, y_tmp[:,-1,:].unsqueeze(1)], axis=1)

            # print("y_i", y_i.shape)
            y = self.decoder(y_i)
            # print("y", y.shape)
            y_hat[:,i] = y[:,0,0]

            
        return y_hat


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 300):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # print("position", position.shape)
        div_term = torch.exp(torch.arange(0, d_model*2, 2) * (-math.log(10_000.0) / d_model))
        # print("div_term", div_term.shape)
        pe = torch.zeros(1, max_len, d_model*2)
        # print("pe", pe.shape)
        pe[0, :, :d_model] = torch.sin(position * div_term)
        pe[0, :, d_model:] = torch.cos(position * div_term)

        # TO SEE POSITIONAL ENCODING
        # plt.figure(figsize=(10,10))
        # plt.imshow(pe[0,:,:])
        # plt.show()
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        
        x = torch.cat((x, x), dim=2)
        # print("PositionalEncoding x", x.shape)
        x = x + self.pe[0,:x.size(1)]


        return self.dropout(x)


class PytorchTransf(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 target_length: int=96,
                 device: str=None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.model_type = 'Transformer'
        

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers, 
                            num_decoder_layers=nlayers, dim_feedforward=2048, dropout=dropout, 
                            activation=nn.ReLU(), custom_encoder=None, 
                            custom_decoder=None, layer_norm_eps=1e-05, 
                            batch_first=True, norm_first=False, device=device, dtype=None)

        
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
                                      nn.ReLU(),
                                      nn.Linear(self.d_model*16, 1),
                                      nn.Sigmoid())



    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, d_model]
            tgt: Tensor, shape [batch_size, output_seq_len, d_model]

        Returns:
            output Tensor of shape [batch_size, outpu_seq_len]
        """
        # print("src before sqrt", src.shape)
        output = torch.empty((tgt.shape[0], tgt.shape[1]))
        for i in range(tgt.shape[1]):
            tgt = self.transformer(src, tgt[:i])
            y_hat = self.decoder(y_hat)

            
        return y_hat


if __name__ == "__main__":
    print(torch.__version__)
    forecasting = False
    if forecasting:
        # ENNCODER DECODER PARAMETERS
        # seq_length = 240
        # batch_size = 8
        
        # head_dim = 32
        # d_model = nhead = 7 # nhead * head_dim
        # d_hid= 2048 * 2
        # nlayers=8
        # dropout=0.5
        # model = Transformer_enc_dec(d_model=d_model, nhead=nhead, d_hid=d_hid,
        #                     nlayers=nlayers, dropout=dropout)

        # TRANSFORMER PARAMETERS
        seq_length = 240
        batch_size = 8
        
        head_dim = 32
        d_model = nhead = 7 # nhead * head_dim
        d_hid= 4096 + 512
        nlayers= 10
        dropout=0.5

        # model = TransformerModel(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=0.2)
        model = TransformerModelWithoutMask(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=0.2)
                
        # seq_length = 240
        # batch_size = 8
        
        # head_dim = 32
        # d_model = nhead = 7 # nhead * head_dim
        # d_hid= 4096 + 512
        # nlayers= 6
        # dropout=0.5

        # model = PytorchTransf(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=0.2)

        # x = torch.rand((batch_size, 96+48, d_model))
        # y = torch.rand((batch_size, 96, d_model))
        # print("x", x.shape)
        # Uncomment for TransformerModel and TransformerEncDec
        x = torch.rand((batch_size, seq_length, d_model))
        src_mask = generate_square_subsequent_mask(seq_length) # .to(device)
        print("src_mask", src_mask.shape)
        output = model(x, src_mask)
        print("output", output.shape)

        # output = model(x, y)
        # print("output", output.shape)
        torch.save(model.state_dict(), "model/transformer.model")
        # print("model:", model)
        
    else:
        seq_length = 240
        batch_size = 8
        
        d_model = nhead = 7 # nhead * head_dim
        
        model = TemporalFusionTransformer()
        x = torch.rand((batch_size, seq_length, d_model))
        output = model(x)
