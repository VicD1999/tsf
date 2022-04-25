import torch
import torch.nn as nn

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

# import matplotlib.pyplot as plt

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model * 2
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
                                      nn.ReLU(),
                                      nn.Linear(self.d_model*16, 96),
                                      nn.Sigmoid())


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        print("src before sqrt", src.shape)
        src = src * math.sqrt(self.d_model)
        print("src AFter sqrt", src.shape)
        # print("BEFORE pos_encoder", src)
        src = self.pos_encoder(src)
        print("AFTER pos_encoder", src.shape)
        # print("src", src.shape)
        # print("src_mask", src_mask.shape)
        output = self.transformer_encoder(src, src_mask)
        print("AFTER Transformer", output.shape)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        print("position", position.shape)
        div_term = torch.exp(torch.arange(0, d_model*2, 2) * (-math.log(10_000.0) / d_model))
        print("div_term", div_term.shape)
        pe = torch.zeros(1, max_len, d_model*2)
        print("pe", pe.shape)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # TO SEE POSITIONAL ENCODING
        # plt.figure(figsize=(10,10))
        # plt.imshow(pe[0,:,:])
        # plt.show()
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        
        x = torch.cat((x, x), dim=2)
        print("PositionalEncoding x", x.shape)
        x = x + self.pe[0,:x.size(1)]


        return self.dropout(x)


if __name__ == "__main__":
    print(torch.__version__)
    seq_length = 240
    batch_size = 32
    
    head_dim = 32
    d_model = nhead = 7 # nhead * head_dim
    d_hid=2048
    nlayers=3
    dropout=0.5
    model = TransformerModel(d_model=d_model, nhead=nhead, d_hid=d_hid,
                        nlayers=nlayers, dropout=dropout)
    x = torch.rand((batch_size, seq_length, d_model))
    print("x", x.shape)
    src_mask = generate_square_subsequent_mask(seq_length) # .to(device)
    print("src_mask", src_mask)
    output = model(x, src_mask)
    print("output", output.shape)
    # print("model:", model)
    
