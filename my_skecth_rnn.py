import numpy as np
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from my_BiLSTM import my_bidirectional_lstm, my_lstm


#HYPERPARAMETERS
Nz = 128
M = 20
enc_hidden_size = 256
dec_hidden_size = 512
dropout = 0.9
batch_size = 100
eta_min = 0.01
R = 0.99995
KL_min = 0.2
wKL = 0.5
lr = 0.001
lr_decay = 0.9999
min_lr = 0.00001
grad_clip = 1.
temperature = 0.4

"""
class encoder(nn.Module):

    def __init__(self, input_size, h_size, z_size):
        super(encoder, self).__init__()
        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size

        #dont care about outputs of encoder
        self.bi_lstm = my_BiLSTM(input_size, h_size, None)

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

        return z, mu, sigma


class decoder(nn.Module):

    def __init__(self, input_size, h_size, z_size, out_size):
        super(decoder, self).__init__()

        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size
        self.out_size = out_size

        self.fc = nn.Linear(z_size, 2 * h_size)
        self.lstm = my_lstm(input_size + z_size, h_size, out_size)
    
    def forward(x, z):
        
        h_c = self.fc(z)
        h0, c0 = h_c[:, :self.h_size], h_c[:, self.h_size:]
        
        z_expanded = z.unsqueeze(1).expand(-1, x.size(1), -1)
        x_z = torch.cat([x, z_expanded], dim=2)

        outs, final_states = self.lstm(x_z, h0, c0)

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

    def forward(x):
        
        z, mu, sig = self.encoder(x)
        y = self.decoder(x, z)

        return y

        

        










"""
