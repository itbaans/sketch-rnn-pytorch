import math
import PIL
from matplotlib import pyplot as plt
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
    
    def forward(self, x, z, hidden_cell=None, device = 'cuda'):
        
        if self.training:
            h_c = self.fc(z)
            h_c = torch.tanh(h_c)
            h0, c0 = h_c[:, :self.h_size], h_c[:, self.h_size:]

            #add sos in x
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*x.size(1)).to(x.device).unsqueeze(0) # [seq_len, batch_size, input_size]
            x_sos = torch.cat([sos, x], dim=0) #[seq_len + 1, batch_size, input_size]
            z_expanded = z.unsqueeze(0).expand(x_sos.size(0), -1, -1)
            x_z = torch.cat([x_sos, z_expanded], dim=2) #[seq_len + 1, batch_size, input_size + z_size]
            h0_c0 = (h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous())

            outs, (hidden, _) = self.lstm(x_z, h0_c0)
            y = self.fc_out(outs)

        else:
            if hidden_cell is not None:
                h_c = self.fc(z)
                h_c = torch.tanh(h_c)
                h0, c0 = h_c[:, :self.h_size], h_c[:, self.h_size:]            
                hidden_cell = (h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous())

            if x is not None:
                z_expanded = z.unsqueeze(0).expand(x.size(0), -1, -1)
                print(x.shape, z_expanded.shape)
                x_z = torch.cat([x, z_expanded], dim=2) #[seq_len + 1, batch_size, input_size + z_size]
                outs, (hidden, cell) = self.lstm(x_z, hidden_cell)
            else:
                # Initialize the decoder with the SOS token and the latent variable z
                sos = torch.stack([torch.Tensor([0,0,1,0,0])]*1).to(device).unsqueeze(0) # [seq_len, batch_size, input_size]
                x_sos = sos
                z_expanded = z.unsqueeze(0).expand(x_sos.size(0), -1, -1)
                x_z = torch.cat([x_sos, z_expanded], dim=2)
                outs, (hidden, cell) = self.lstm(x_z, hidden_cell)
            y = self.fc_out(hidden)
            return y, (hidden, cell)

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

    def get_sampling_params(self, decoder_out, temp):
        # Extract the last 3 values from the decoder output
        feat_size = decoder_out.size(-1)
        # Apply Softmax to the last 3 features
        softmax_last3 = torch.nn.functional.softmax(decoder_out[:, :, -3:], dim=-1)
        softmax_last3 = softmax_last3 * temp

        # Apply Softmax to every 1st feature in 6-feature groups (ignoring last 3)
        idx_first = torch.arange(0, feat_size, 6)  # Indices of the 1st features
        softmax_first = torch.nn.functional.softmax(decoder_out[:, :, idx_first], dim=-1)
        softmax_first = softmax_first * temp

        # Apply Tanh to every 6th feature in 6-feature groups (ignoring last 3)
        idx_sixth = torch.arange(5, feat_size, 6)  # Indices of the 6th features
        tanh_sixth = torch.tanh(decoder_out[:, :, idx_sixth])

        # Apply exp to every 4th and 5th feature in 6-feature groups (ignoring last 3)
        idx_fourth = torch.arange(3, feat_size, 6)  # Indices of the 4th features
        idx_fifth = torch.arange(4, feat_size, 6)  # Indices of the 5th features
        exp_fourth = torch.exp(decoder_out[:, :, idx_fourth])
        exp_fourth = exp_fourth * temp

        exp_fifth = torch.exp(decoder_out[:, :, idx_fifth])
        exp_fifth = exp_fifth * temp

        # Create a transformed tensor
        y_transformed = decoder_out.clone()
        y_transformed[:, :, -3:] = softmax_last3  # Update last 3 features
        y_transformed[:, :, idx_first] = softmax_first  # Update 1st feature in each group
        y_transformed[:, :, idx_sixth] = tanh_sixth  # Update 6th feature in each group

        y_transformed[:, :, idx_fourth] = exp_fourth  # Update 4th feature in each group
        y_transformed[:, :, idx_fifth] = exp_fifth  # Update 5th feature in each group
        
        return y_transformed
    
    def sample_bivariate_normal(self, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
        # Check for near-zero sigma values which can cause issues
        sigma_x = torch.max(sigma_x, torch.tensor(1e-6, device=sigma_x.device))
        sigma_y = torch.max(sigma_y, torch.tensor(1e-6, device=sigma_y.device))
        rho_xy = torch.clamp(rho_xy, -0.9999, 0.9999) # Clamp correlation

        mean = torch.tensor([mu_x, mu_y], device=mu_x.device)

        # Construct covariance matrix
        cov = torch.zeros((2, 2), device=mean.device)
        cov[0, 0] = sigma_x * sigma_x
        cov[1, 1] = sigma_y * sigma_y
        cov[0, 1] = cov[1, 0] = rho_xy * sigma_x * sigma_y

        # Use PyTorch's multivariate normal sampler
        # Need to handle potential non-positive definite matrices if parameters are extreme
        try:
            # Add small jitter for numerical stability if needed
            # cov += torch.eye(2, device=mean.device) * 1e-6
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            sample = dist.sample() # Shape (2,)
            return sample[0], sample[1] # Return dx, dy
        except ValueError as e:
            print(f"Warning: Covariance matrix issue - {e}. Returning mean.")
            # Fallback if covariance is not valid (shouldn't happen with clamps/epsilons often)
            return mu_x, mu_y

    def sample_next_stroke(self, processed_params, M):

        if len(processed_params.shape) != 2 or processed_params.shape[1] != 6 * M + 3:
            print(f"Error: Input params shape is {processed_params.shape}, expected (batch_size, {6*M+3}).")
            return None

        batch_size = processed_params.shape[0]
        device = processed_params.device
        output_strokes = np.zeros((batch_size, 5), dtype=np.float32)

        # Extract pen state probabilities (q1, q2, q3) - last 3 elements
        q_probs = processed_params[:, -3:] # Shape (batch_size, 3)

        # Extract GMM parameters
        # Shape (batch_size, 6 * M) -> (batch_size, M, 6)
        gmm_params = processed_params[:, :6 * M].view(batch_size, M, 6)

        pi_probs = gmm_params[:, :, 0] # Mixture weights (already probabilities) - Shape (batch_size, M)
        mu_x = gmm_params[:, :, 1]     # Shape (batch_size, M)
        mu_y = gmm_params[:, :, 2]     # Shape (batch_size, M)
        sigma_x = gmm_params[:, :, 3]  # Std dev (already positive from exp) - Shape (batch_size, M)
        sigma_y = gmm_params[:, :, 4]  # Std dev (already positive from exp) - Shape (batch_size, M)
        rho_xy = gmm_params[:, :, 5]   # Correlation (already in [-1, 1] from tanh) - Shape (batch_size, M)

        # --- Sampling Loop (for each item in the batch) ---
        dx_sampled, dy_sampled, p_sampled = None, None, None
        for i in range(batch_size): #batch_size probably should be 1
            # 1. Sample Pen State (p1, p2, p3)
            # Use multinomial distribution to sample one index based on probabilities q_probs
            # Ensure probabilities sum to 1 (softmax should handle this, but clamp for safety)
            q_probs_i = q_probs[i] / torch.sum(q_probs[i] + 1e-9) # Normalize just in case
            pen_state_idx = torch.multinomial(q_probs_i, num_samples=1).item() # Get the index (0, 1, or 2)
            # Create one-hot vector
            p_sampled = [0., 0., 0.]
            p_sampled[pen_state_idx] = 1.

            # 2. Sample Offset (dx, dy) from GMM
            # Only sample dx, dy if drawing hasn't ended (p3 != 1)
            if p_sampled[2] != 1: # If p3 is not 1
                # Sample mixture component index based on pi_probs
                pi_probs_i = pi_probs[i] / torch.sum(pi_probs[i] + 1e-9) # Normalize just in case
                mixture_idx = torch.multinomial(pi_probs_i, num_samples=1).item() # Index of chosen Gaussian

                # Get parameters for the chosen mixture component
                mux_s = mu_x[i, mixture_idx]
                muy_s = mu_y[i, mixture_idx]
                sx_s = sigma_x[i, mixture_idx]
                sy_s = sigma_y[i, mixture_idx]
                rho_s = rho_xy[i, mixture_idx]

                # Sample (dx, dy) from the selected bivariate normal distribution
                dx_sampled, dy_sampled = self.sample_bivariate_normal(mux_s, muy_s, sx_s, sy_s, rho_s)
            else:
                # If p3 == 1 (end of drawing), set offset to (0, 0)
                dx_sampled = torch.tensor(0.0, device=device)
                dy_sampled = torch.tensor(0.0, device=device)

            # 3. Store the results for this batch item
            output_strokes[i, 0] = dx_sampled.item()
            output_strokes[i, 1] = dy_sampled.item()
            output_strokes[i, 2:] = p_sampled

        return output_strokes, dx_sampled, dy_sampled, p_sampled
    
    def make_image(self, sequence, epoch, name='_output_'):
        """Plot drawing with separated strokes"""
        # Split sequence based on strokes (where the third column > 0)
        strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)

        # Create a figure and axis for plotting
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # Plot each stroke (inverting the y-axis for better visualization)
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])

        # Get canvas and save the image
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()

        # Convert the plot to a PIL image using the print_to_buffer method
        width, height = canvas.get_width_height()
        buf, (width, height) = canvas.print_to_buffer()
        image_array = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))

        # Convert the ARGB image to RGB by removing the alpha channel
        pil_image = PIL.Image.fromarray(image_array[:, :, :3], 'RGB')

        # Save the image with a dynamic name based on epoch
        name = str(epoch) + name + '.jpg'
        pil_image.save(name, "JPEG")

        # Close the plot to free up resources
        plt.close("all")

    def forward(self, x, nmax=None, tenperature=1.0, M=20):
        
        if self.training:
            z, mu, sig_hat = self.encoder(x)
            y = self.decoder(x, z)         
            return y, mu, sig_hat, z
        
        else:   
            self.encoder.train(False)
            self.decoder.train(False)
            z, _, _ = self.encoder(x)
            seq_x = []
            seq_y = []
            seq_z = []
            s=None
            hidden_cell = None
            for i in range(nmax):
                if i == 0:
                    y, hidden_cell = self.decoder(None, z, hidden_cell)
                else:
                    y, hidden_cell = self.decoder(s, z, hidden_cell)
                
                y = self.get_sampling_params(y, tenperature)
                y = y.squeeze(0)
                s, dx, dy, p = self.sample_next_stroke(y, M)
                #convert s to tensor
                s = torch.tensor(s, device=x.device)
                s = s.unsqueeze(0)
                #detach dxm dy
                dx = dx.detach().cpu().numpy()
                dy = dy.detach().cpu().numpy()
            
                seq_x.append(dx)
                seq_y.append(dy)
                seq_z.append(p[1]==1)

                if p[2] == 1:
                    print(i)
                    break
            x_sample = np.cumsum(seq_x, 0)
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
            sequence = np.stack([x_sample,y_sample,z_sample]).T
            print(sequence)
            self.make_image(sequence, 6969, name='_output_')
                

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


