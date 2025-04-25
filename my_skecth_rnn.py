import math
import numpy as np
import torch 
import torch.nn as nn
class encoder(nn.Module):

    def __init__(self, input_size, h_size, z_size, rec_dropout=0., layer_norm=False):
        super(encoder, self).__init__()
        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size

        #dont care about outputs of encoder
        self.bi_lstm = nn.LSTM(input_size, h_size, bidirectional=True, 
                                   dropout=rec_dropout, batch_first=False) #,layer_norm=layer_norm)

        self.fc_mu = nn.Linear(2*h_size, z_size)
        self.fc_sigma_hat = nn.Linear(2*h_size, z_size)

    def forward(self, x):
        _, (hidden_states, _) = self.bi_lstm(x)
        h_forward, h_backward = hidden_states
        h_concat = torch.cat([h_forward, h_backward], dim=1)

        mu = self.fc_mu(h_concat)
        sig_hat = self.fc_sigma_hat(h_concat)

        sigma = torch.exp(sig_hat / 2.0)

        epsilon = torch.randn_like(mu)
        z = mu + sigma * epsilon

        return z, mu, sig_hat


class decoder(nn.Module):

    def __init__(self, input_size, h_size, z_size, out_size, rec_dropout=0.):#, layer_norm=False):
        super(decoder, self).__init__()

        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size
        self.out_size = out_size

        self.fc = nn.Linear(z_size, 2 * h_size)
        self.lstm = nn.LSTM(input_size + z_size, h_size, dropout=rec_dropout, batch_first=False)
        self.fc_out = nn.Linear(h_size, out_size)
    
    def forward(self, x, z):
        
        h_c = self.fc(z)
        h_c = torch.tanh(h_c)
        h0, c0 = h_c[:, :self.h_size], h_c[:, self.h_size:]

        #add sos in x
        sos = torch.stack([torch.Tensor([0,0,1,0,0])]*x.size(1)).to(x.device).unsqueeze(0) # [seq_len, batch_size, input_size]
        x_sos = torch.cat([sos, x], dim=0) #[seq_len + 1, batch_size, input_size]
        z_expanded = z.unsqueeze(0).expand(x_sos.size(0), -1, -1)
        x_z = torch.cat([x_sos, z_expanded], dim=2) #[seq_len + 1, batch_size, input_size + z_size]
        h0_c0 = (h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous())

        outs, _ = self.lstm(x_z, h0_c0)
        y = self.fc_out(outs)

        return y #[seq_len + 1, batch_size, out_size]

class sketch_rnn(nn.Module):

    def __init__(self, input_size, enc_hsize, dec_hsize, z_size, dec_out_size, rec_dropout=0.):#, layer_norm=False):
        super(sketch_rnn, self).__init__()

        self.input_size = input_size
        self.enc_hsize = enc_hsize
        self.dec_hsize = dec_hsize
        self.z_size = z_size
        self.dec_out_size = dec_out_size

        self.encoder = encoder(input_size, enc_hsize, z_size, rec_dropout)
        self.decoder = decoder(input_size, dec_hsize, z_size, dec_out_size, rec_dropout)

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
        super(SKETCH_RNN_LOSS, self).__init__()
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
    
    def process_logits(self, y):

        y_decr = y[:-1, :, :] # Remove the last element of the sequence

        feat_size = y_decr.size(-1)
        # Apply Softmax to the last 3 features
        softmax_last3 = torch.nn.functional.softmax(y_decr[:, :, -3:], dim=-1)

        # Apply Softmax to every 1st feature in 6-feature groups (ignoring last 3)
        idx_first = torch.arange(0, feat_size, 6)  # Indices of the 1st features
        softmax_first = torch.nn.functional.softmax(y_decr[:, :, idx_first], dim=-1)

        # Apply Tanh to every 6th feature in 6-feature groups (ignoring last 3)
        idx_sixth = torch.arange(5, feat_size, 6)  # Indices of the 6th features
        tanh_sixth = torch.tanh(y_decr[:, :, idx_sixth])

        # Create a transformed tensor
        y_transformed = y_decr.clone()
        y_transformed[:, :, -3:] = softmax_last3  # Update last 3 features
        y_transformed[:, :, idx_first] = softmax_first  # Update 1st feature in each group
        y_transformed[:, :, idx_sixth] = tanh_sixth  # Update 6th feature in each group
        
        return y_transformed

    def LsLoss(self, y, x):
        # y: [seq, batch, 6*M + 3]
        # x: [seq, batch, 5]
        seq_len, batch_size, _ = x.shape
        dx = x[..., 0] # [seq, batch]
        dy = x[..., 1] # [seq, batch]

        # Reshape first 6*M part into [seq, batch, M, 6]
        y_mixtures = y[:, :, :6*self.M].reshape(seq_len, batch_size, self.M, 6)

        pi = y_mixtures[..., 0]  # [seq, batch, M]
        mux = y_mixtures[..., 1]
        muy = y_mixtures[..., 2]
        sx = torch.exp(y_mixtures[..., 3])  # [seq, batch, M]
        sy = torch.exp(y_mixtures[..., 4])
        rho = y_mixtures[..., 5]

        # Expand dx, dy to match [seq, batch, M]
        dx_exp = dx.unsqueeze(-1).expand(-1, -1, self.M)
        dy_exp = dy.unsqueeze(-1).expand(-1, -1, self.M)

        v = self.bivariate_normal(dx_exp, dy_exp, mux, muy, sx, sy, rho)  # [seq, batch, M]
        weighted_v = pi * v  # [seq, batch, M]

        sum_over_m = weighted_v.sum(dim=-1)  # [seq, batch]
        log_sum = torch.log(sum_over_m + 1e-9)  # [seq, batch]
        sum_over_seq = log_sum.sum(dim=0)  # [batch]
        total_log = sum_over_seq.sum()  # scalar

        loss = -total_log / float(self.Nmax * batch_size)

        return loss
    
    def LpLoss(self, y, x):
        _, batch_size, _ = x.shape
        
        p1 = x[..., 2]  # [seq, batch]
        p2 = x[..., 3]
        p3 = x[..., 4]
        
        y_qs = y[:, :, 6*self.M:]
        q1 = y_qs[..., 0]  # [seq, batch]
        q2 = y_qs[..., 1]
        q3 = y_qs[..., 2]
        
        # Calculate individual cross-entropy terms
        pq1 = p1 * torch.log(q1 + 1e-9)
        pq2 = p2 * torch.log(q2 + 1e-9)
        pq3 = p3 * torch.log(q3 + 1e-9)
        
        # Sum over k dimension (k=1,2,3)
        sum_k = pq1 + pq2 + pq3  # [seq, batch]
        
        # Sum over all timesteps (Nmax in the formula)
        sum_i = sum_k.sum(dim=0)  # [batch]
        
        # Take mean over batch
        loss = -sum_i.sum() / float(self.Nmax * batch_size)
    
        return loss
    
    def lk_loss(self, mu, sig_hat):

        batch_size = mu.shape[0]
        nz_sum = torch.sum(1 + sig_hat - mu**2 - torch.exp(sig_hat), dim=1)
        batch_sum = torch.sum(nz_sum, dim=0)
        return -batch_sum / float(2 * self.Nz * batch_size)

    def forward(self, y, x, mu, sig_hat, R, eta_min, wkl, step, kl_min=0.01):

        # Process logits
        y = self.process_logits(y) # [seq, batch, 6*M + 3]
        # Reconstruction losses
        lr = self.LsLoss(y, x) + self.LpLoss(y, x)

        # KL loss
        lkl = self.lk_loss(mu, sig_hat)

        # KL annealing factor η_step
        eta_step = 1.0 - (1.0 - eta_min) * R ** step

        # KL term with annealing and minimum KL threshold
        annealed_kl = wkl * eta_step * torch.maximum(lkl, torch.tensor(kl_min, device=lkl.device))

        return lr + annealed_kl



#temprature applied during inferance


