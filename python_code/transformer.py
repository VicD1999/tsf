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


class Transformer(nn.Module):
    """
    Class from: https://medium.com/analytics-vidhya/implementing-transformer-from-scratch-in-pytorch-8c872a3044c9
    Classic Transformer that both encodes and decodes.
    
    Prediction-time inference is done greedily.
    NOTE: start token is hard-coded to be 0, end token to be 1. If changing, update predict() accordingly.
    """

    def __init__(self, nlayers: int, max_output_length: int=300, d_model: int = 7, device: str=None):
        super().__init__()

        # Parameters
        self.d_model = d_model + 2
        self.max_output_length = max_output_length
        self.device = device
        nhead = self.d_model
        dim_feedforward = dim

        # Encoder part
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=nlayers
        )

        # Decoder part
        self.y_mask = generate_square_subsequent_mask(self.max_output_length)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=nlayers
        )
        self.fc = nn.Linear(self.d_model, 1)

        # It is empirically important to initialize weights properly
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
      
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (Sx, B, E) embedding
        """
        x = self.pos_encoder(x)  # (Sx, B, E)
        x = self.transformer_encoder(x)  # (Sx, B, E)
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """
        Input
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (Sy, B, C) logits
        """
        y = self.pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)
        
        output_tokens = (torch.ones((x.shape[0], self.max_output_length))).type_as(x).long() # (B, max_length)
        output_tokens[:, 0] = 0  # Set start token
        for Sy in range(1, self.max_output_length):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(y, encoded_x)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[-1:]  # Set the last output token
        return output_tokens



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

# class TransformerModelWithoutMask(nn.Module):

#     def __init__(self, d_model: int, nhead: int, d_hid: int,
#                  nlayers: int, dropout: float = 0.5,
#                  target_length: int=96,
#                  device: str=None):
#         super().__init__()
#         self.d_model = d_model * 2
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(d_model, dropout)

#         encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         # self.encoder = nn.Embedding(ntoken, d_model)
        
#         self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
#                                       nn.ReLU(),
#                                       nn.Linear(self.d_model*16, 1),
#                                       nn.Sigmoid())

#         self.target_length = target_length


#     def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
#         """
#         Args:
#             src: Tensor, shape [seq_len, batch_size]
#             src_mask: Tensor, shape [seq_len, seq_len]

#         Returns:
#             output Tensor of shape [seq_len, batch_size, ntoken]
#         """
#         # print("src before sqrt", src.shape)
#         src = src * math.sqrt(self.d_model)
#         # print("src AFter sqrt", src.shape)
#         # print("BEFORE pos_encoder", src)
#         src = self.pos_encoder(src)
#         # print("AFTER pos_encoder", src.shape)
#         # print("src", src.shape)
#         # print("src_mask", src_mask.shape)
#         output = self.transformer_encoder(src)
#         # print("AFTER Transformer", output.shape)
#         output = self.decoder(output)
#         return output[:,:self.target_length,0]

class Transformer_enc_dec(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 target_length: int=96,
                 device: str=None):
        super().__init__()
        self.device = device
        self.d_model = d_model + 2
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, device=device)
        
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


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # print("src before sqrt", src.shape)
        # src = src * math.sqrt(self.d_model)
        # print("src AFter sqrt", src.shape)
        # print("BEFORE pos_encoder", src)

        src = self.pos_encoder(src)
        src, forecast = src[:144], src[144:]
        # print("AFTER pos_encoder", src.shape)
        # print("src", src.shape)
        # print("src_mask", src_mask.shape)
        encoded_x = self.transformer_encoder(src)
        print("AFTER Transformer", encoded_x.shape)


        Sy = self.target_length # Should be 96
        
        y_i = -torch.ones((src.shape[0],self.d_model)).to(self.device)
        print("y_i", y_i.device)
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


# class PytorchTransf(nn.Module):

#     def __init__(self, d_model: int, nhead: int, d_hid: int,
#                  nlayers: int, dropout: float = 0.5,
#                  target_length: int=96,
#                  device: str=None):
#         super().__init__()
#         self.device = device
#         self.d_model = d_model
#         self.model_type = 'Transformer'
#         self.target_length = target_length

#         self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers, 
#                             num_decoder_layers=nlayers, dim_feedforward=2048, dropout=dropout, 
#                             activation=nn.ReLU(), custom_encoder=None, 
#                             custom_decoder=None, layer_norm_eps=1e-05, 
#                             batch_first=True, norm_first=False, device=device, dtype=None)

        
#         self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
#                                       nn.ReLU(),
#                                       nn.Linear(self.d_model*16, 1),
#                                       nn.Sigmoid())



#     def forward(self, src: Tensor) -> Tensor:
#         """
#         Args:
#             src: Tensor, shape [batch_size, seq_len, d_model]
#             tgt: Tensor, shape [batch_size, output_seq_len, d_model]

#         Returns:
#             output Tensor of shape [batch_size, outpu_seq_len]
#         """
#         # print("src before sqrt", src.shape)
#         # output = torch.empty((tgt.shape[0], tgt.shape[1]))
#         batch_size = src.shape[0]
#         d_model = src.shape[2]
#         tgt_i_1 = torch.zeros((batch_size, 1, d_model))
#         for i in range(self.target_length):
#             # print("src", src.shape)
#             # print("tgt", tgt[:,:i+1].shape)
#             tgt_i_1 = self.transformer(src, tgt_i_1)
#             print("tgt_i_1", tgt_i_1.shape)
#         # print("y_hat after transformer", y_hat.shape)
#         y_hat = self.decoder(tgt_i_1[:,:-self.target_length,:])
#         # print("y_hat", y_hat.shape)
#         # output[:,i] = y_hat

#         return y_hat.squeeze(2)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 300, device: str = None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # print("position", position.shape)
        div_term = torch.exp(torch.Tensor([-math.log(10_000.0) / d_model]))
        # print("div_term", div_term.shape)
        # pe = torch.zeros(1, max_len, d_model*2)
        # print("pe", pe.shape)

        self.sin = torch.sin(position * div_term).to(device)
        self.cos = torch.cos(position * div_term).to(device)


        # TO SEE POSITIONAL ENCODING
        # plt.figure(figsize=(10,10))
        # plt.imshow(pe[0,:,:])
        # plt.show()
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        # self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        
        # x = torch.cat((x, x), dim=2)
        # print("PositionalEncoding x", x.shape)

        # print("semf.sin", self.sin.shape)

        cat1 = self.sin[:x.shape[1]].unsqueeze(0).expand(x.shape[0], x.shape[1], 1)
        # print("cat1", cat1.shape)
        cat2 = self.cos[:x.shape[1]].unsqueeze(0).expand(x.shape[0], x.shape[1], 1)
        # print("cat2", cat2.shape)
        x = torch.cat((x, cat1, cat2), dim=2)


        return x

class TestTransf(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 target_length: int=96,
                 device: str=None):
        super().__init__()
        self.d_model = d_model + 2
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, device=device)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.d_model, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.d_model*16),
                                      nn.ReLU(),
                                      nn.Linear(self.d_model*16, 1),
                                      nn.Sigmoid())

        self.target_length = target_length


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # print("src before sqrt", src.shape)
        # src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # print("AFTER pos_encoder", src.shape)
        # print("src", src.shape)
        # print("src_mask", src_mask.shape)
        output = self.transformer_encoder(src)
        # print("AFTER Transformer", output.shape)
        output = self.decoder(output)
        return output[:,:self.target_length,0]



if __name__ == "__main__":
    print(torch.__version__)

    import util as u
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



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


    # model = TestTransf(d_model=7, nhead=7, d_hid=2048, nlayers=10, dropout=0.5, target_length=96, device=device)
    model = Transformer_enc_dec(d_model=7, nhead=9, d_hid=2048,
                                nlayers=10, dropout=0.2, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    for e in range(0, 200):
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
