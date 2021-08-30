import torch
import torch.nn as nn


class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self,feature_size=5,num_layers=3,dropout=0):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # print("sz", sz.shape)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        print("mask", mask.shape)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        print("mask", mask.shape)
        return mask

    def forward(self, src):
        
        mask = self._generate_square_subsequent_mask(src.shape[1]) # .to(device)
        print("mask", mask.shape)
        print("src", src.shape)
        output = self.transformer_encoder(src, mask)
        print("output", output.shape)
        output = self.decoder(output)
        return output

if __name__ == "__main__":
	model = Transformer()
	x = torch.rand((32, 120, 5))
	print("x", x.shape)
	output = model(x)
	print("output", output.shape)
