import math
import numpy as np
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from my_BiLSTM import my_bidirectional_lstm, my_lstm

#HYPERPARAMETERS
# Nz = 128
# M = 20
# enc_hidden_size = 256
# dec_hidden_size = 512
# dropout = 0.9
# batch_size = 100
# eta_min = 0.01
# R = 0.99995
# KL_min = 0.2
# wKL = 0.5
# lr = 0.001
# lr_decay = 0.9999
# min_lr = 0.00001
# grad_clip = 1.
# temperature = 0.4

class encoder(nn.Module):

    def __init__(self, input_size, h_size, z_size):
        super(encoder, self).__init__()
        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size

        #dont care about outputs of encoder
        self.bi_lstm = my_bidirectional_lstm(input_size, h_size, None)

        self.fc_mu = nn.Linear(2*h_size, z_size)
        self.fc_sigma_hat = nn.Linear(2*h_size, z_size)

    def forward(self, x):
        h_forward, h_backward = self.bi_lstm(x)
        h_concat = torch.cat([h_forward, h_backward], dim=1)

        mu = self.fc_mu(h_concat)
        sig_hat = self.fc_sigma_hat(h_concat)

        sigma = torch.exp(sig_hat / 2.0)

        epsilon = torch.randn_like(mu)
        z = mu + sigma * epsilon

        return z, mu, sig_hat


class decoder(nn.Module):

    def __init__(self, input_size, h_size, z_size, out_size):
        super(decoder, self).__init__()

        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size
        self.out_size = out_size

        self.fc = nn.Linear(z_size, 2 * h_size)
        self.lstm = my_lstm(input_size + z_size, h_size, out_size)
    
    def forward(self, x, z):
        
        h_c = self.fc(z)
        h_c = torch.tanh(h_c)
        h0, c0 = h_c[:, :self.h_size], h_c[:, self.h_size:]

        z_expanded = z.unsqueeze(1).expand(x.size(0), -1, -1)
        x_z = torch.cat([x, z_expanded], dim=2)

        outs, _ = self.lstm(x_z, h0, c0)

        return outs

class sketch_rnn(nn.Module):

    def __init__(self, input_size, enc_hsize, dec_hsize, z_size, dec_out_size):
        super(sketch_rnn, self).__init__()

        self.input_size = input_size
        self.enc_hsize = enc_hsize
        self.dec_hsize = dec_hsize
        self.z_size = z_size
        self.dec_out_size = dec_out_size

        self.encoder = encoder(input_size, enc_hsize, z_size)
        self.decoder = decoder(input_size, dec_hsize, z_size, dec_out_size)

    def forward(self, x):
        
        z, mu, sig_hat = self.encoder(x)
        y = self.decoder(x, z)

        return y, mu, sig_hat

#y --> [batch_size, 6*M + 3]
# the last 3 values of the yi tensor represents the predicted q1, q2, q3
# and the rest of the values are broken into pairs of 6, where the first element
# is the probablity of the jth bivar-Normal distribtuin and the 5 values along side is the parameters of the bivar-normal distribuion

#give the 6*M part to Ls loss, last 3 part to Lp loss
class SKETCH_RNN_LOSS(nn.Module):

    def __init__(self, Nmax, M, Nz):
        self.Nmax = Nmax
        self.M = M
        self.Nz = Nz
    
    def bivariate_normal(self, dx, dy, mux, muy, sx, sy, rho):
        normx = (dx - mux) / sx
        normy = (dy - muy) / sy
        z = normx**2 + normy**2 - 2 * rho * normx * normy
        denom = 2 * math.pi * sx * sy * torch.sqrt(1 - rho**2 + 1e-6)
        exponent = -z / (2 * (1 - rho**2 + 1e-6))
        return torch.exp(exponent) / (denom + 1e-6)

    def LsLoss(self, y, x):
        # y: [seq, batch, 6*M + 3]
        # x: [seq, batch, 5]
        seq_len, batch_size, _ = x.shape
        dx = x[..., 0]
        dy = x[..., 1]

        # Reshape first 6*M part into [seq, batch, M, 6]
        y_mixtures = y[:, :, :6*self.M].reshape(seq_len, batch_size, self.M, 6)

        pi = y_mixtures[..., 0]  # [seq, batch, M]
        mux = y_mixtures[..., 1]
        muy = y_mixtures[..., 2]
        sx = torch.clamp(y_mixtures[..., 3], min=1e-6)
        sy = torch.clamp(y_mixtures[..., 4], min=1e-6)
        rho = torch.clamp(y_mixtures[..., 5], min=-0.999, max=0.999)

        # Expand dx, dy to match [seq, batch, M]
        dx_exp = dx.unsqueeze(-1).expand(-1, -1, self.M)
        dy_exp = dy.unsqueeze(-1).expand(-1, -1, self.M)

        v = self.bivariate_normal(dx_exp, dy_exp, mux, muy, sx, sy, rho)  # [seq, batch, M]
        weighted_v = pi * v  # [seq, batch, M]

        sum_over_m = weighted_v.sum(dim=-1)  # [seq, batch]
        log_sum = torch.log(sum_over_m + 1e-9)  # avoid log(0)

        total_log = log_sum.sum()  # sum over all timesteps and batch
        loss = -total_log / self.Nmax

        return loss
    
    def LpLoss(self, y, x):

        p1 = x[..., 2]
        p2 = x[..., 3]
        p3 = x[..., 4]

        y_qs = y[:, :, 6*self.M:]
        q1 = y_qs[..., 0]
        q2 = y_qs[..., 1]
        q3 = y_qs[..., 2]

        pq1 = p1 * torch.log(q1 + 1e-9)
        pq2 = p2 * torch.log(q2 + 1e-9)
        pq3 = p3 * torch.log(q3 + 1e-9)

        sum_all = pq1 + pq2 + pq3

        return -sum_all / self.Nmax
    
    def lk_loss(self, mu, sig_hat):

        return -(1 + sig_hat - mu** - torch.exp(sig_hat)) / 2 * self.Nz

    def forward(self, y, x, mu, sig_hat, R, eta_min, wkl, step, kl_min=0.01):
        # Reconstruction losses
        lr = self.LsLoss(y, x) + self.LpLoss(y, x)

        # KL loss
        lkl = self.lk_loss(mu, sig_hat)

        # KL annealing factor η_step
        eta_step = 1.0 - (1.0 - eta_min) * R ** step

        # KL term with annealing and minimum KL threshold
        annealed_kl = wkl * eta_step * torch.maximum(lkl, torch.tensor(kl_min, device=lkl.device))

        return lr + annealed_kl

#drop out
#layer norm
#temprature
#softmax logits
#training

