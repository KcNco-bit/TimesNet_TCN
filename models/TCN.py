import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import TCN_Module


class TCNBlock(nn.Module):
    def __init__(self, configs):
        super(TCNBlock, self).__init__()
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.tcn_conv = nn.Sequential(
            TCN_Module(self.d_model, self.d_ff, kernel_size=3, dilation_rates=[1, 2, 4]),
            nn.GELU(),
            TCN_Module(self.d_ff, self.d_model, kernel_size=3, dilation_rates=[1, 2, 4])
        )

    def forward(self, x):
        B, T, N = x.size()  
        out = x.permute(0, 2, 1).unsqueeze(2)  # [B, N, 1, T]
        tcn_out = self.tcn_conv(out)
        out = tcn_out.squeeze(2).permute(0, 2, 1)
        res = out + x
        return res


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        self.model = nn.ModuleList([TCNBlock(configs)
                                    for _ in range(configs.e_layers)])
        
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
