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

    def forward(self, x, hidden_cell=None):
        if hidden_cell is None:
            h0 = torch.zeros(2, x.size(1), self.h_size).to(x.device)
            c0 = torch.zeros(2, x.size(1), self.h_size).to(x.device)
            hidden_cell = (h0, c0)

        _, (hidden_states, _) = self.bi_lstm(x.float(), hidden_cell)
        h_forward, h_backward = hidden_states
        #print(h_forward.shape, h_backward.shape)
        h_concat = torch.cat([h_forward, h_backward], dim=1)
        #print(h_concat.shape)

        mu = self.fc_mu(h_concat)
        sig_hat = self.fc_sigma_hat(h_concat)

        sigma = torch.exp(sig_hat / 2.0)
        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda()
        z = mu + sigma * N

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
            #print(outs.shape)
            y = self.fc_out(outs)
            print(y.shape)

        else:
            if hidden_cell is None:
                h_c = self.fc(z)
                h_c = torch.tanh(h_c)
                h0, c0 = h_c[:, :self.h_size], h_c[:, self.h_size:]            
                hidden_cell = (h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous())

            if x is not None:
                z_expanded = z.unsqueeze(0).expand(x.size(0), -1, -1)
                #print(x.shape, z_expanded.shape)
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
        softmax_last3 = decoder_out[:, :, -3:] / temp
        softmax_last3 = torch.nn.functional.softmax(softmax_last3, dim=-1)

        # Apply Softmax to every 1st feature in 6-feature groups (ignoring last 3)
        idx_first = torch.arange(0, feat_size, 6)  # Indices of the 1st features
        softmax_first = decoder_out[:, :, idx_first] / temp
        softmax_first = torch.nn.functional.softmax(softmax_first, dim=-1)

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
        # Normalized distance
        dx_norm = (dx - mux) / sx
        dy_norm = (dy - muy) / sy
        
        # Calculate z term correctly according to the paper
        z = (dx_norm**2 + dy_norm**2 - 2 * rho * dx_norm * dy_norm) / (1 - rho**2 + 1e-8)
        
        # Calculate denominator
        denom = 2 * math.pi * sx * sy * torch.sqrt(1 - rho**2 + 1e-8)
        
        # Calculate exponent
        exponent = -z / 2
        
        return torch.exp(exponent) / denom
    
    def process_logits(self, y):
        """Process the raw output logits from the decoder network."""
        y_decr = y
        seq_len, batch_size, feat_size = y_decr.shape
        
        # Reshape to separate the mixture components and pen states
        mixture_params = y_decr[:, :, :6*self.M].reshape(seq_len, batch_size, self.M, 6)
        pen_params = y_decr[:, :, 6*self.M:]
        
        # Process mixture weights - apply softmax to pi values
        pi = torch.nn.functional.softmax(mixture_params[:, :, :, 0], dim=2)
        
        # Keep mu_x and mu_y as they are
        mu_x = mixture_params[:, :, :, 1]
        mu_y = mixture_params[:, :, :, 2]
        
        # Apply exp to sigma to ensure positivity
        sigma_x = torch.exp(mixture_params[:, :, :, 3])
        sigma_y = torch.exp(mixture_params[:, :, :, 4])
        
        # Apply tanh to rho to ensure it's between -1 and 1
        rho = torch.tanh(mixture_params[:, :, :, 5])
        
        # Apply softmax to pen states
        pen_logits = torch.nn.functional.softmax(pen_params, dim=2)
        
        # Reconstruct the output tensor
        processed_mixture = torch.stack([pi, mu_x, mu_y, sigma_x, sigma_y, rho], dim=3)
        processed_mixture = processed_mixture.reshape(seq_len, batch_size, 6*self.M)
        
        # Concatenate with processed pen states
        y_processed = torch.cat([processed_mixture, pen_logits], dim=2)
        
        return y_processed # Exclude the last time step for consistency with input x

    def LsLoss(self, y, x, mask=None):
        """Calculate the mixture density loss for the stroke parameters, with optional masking."""
        seq_len, batch_size, _ = x.shape
        dx = x[..., 0]
        dy = x[..., 1]
        print(seq_len)
        y_mixtures = y[:, :, :6*self.M].reshape(seq_len, batch_size, self.M, 6)

        pi = y_mixtures[..., 0]
        mux = y_mixtures[..., 1]
        muy = y_mixtures[..., 2]
        sx = y_mixtures[..., 3]
        sy = y_mixtures[..., 4]
        rho = y_mixtures[..., 5]

        dx_exp = dx.unsqueeze(-1).expand(-1, -1, self.M)
        dy_exp = dy.unsqueeze(-1).expand(-1, -1, self.M)

        v = self.bivariate_normal(dx_exp, dy_exp, mux, muy, sx, sy, rho)
        weighted_v = pi * v
        sum_over_m = weighted_v.sum(dim=-1)  # [seq, batch]

        log_sum = torch.log(sum_over_m + 1e-8)  # [seq, batch]

        if mask is not None:
            log_sum = log_sum * mask  # apply mask before summing

        seq_sum = log_sum.sum(dim=0)  # [batch]
        loss = -seq_sum.mean() / float(self.Nmax)

        return loss

    def LpLoss(self, y, x, mask=None):
        """Calculate the categorical cross-entropy loss for the pen states, with optional masking."""
        seq_len, batch_size, _ = x.shape

        p1 = x[..., 2]
        p2 = x[..., 3]
        p3 = x[..., 4]

        y_qs = y[:, :, 6*self.M:]
        q1 = y_qs[..., 0]
        q2 = y_qs[..., 1]
        q3 = y_qs[..., 2]

        pq1 = p1 * torch.log(q1 + 1e-8)
        pq2 = p2 * torch.log(q2 + 1e-8)
        pq3 = p3 * torch.log(q3 + 1e-8)

        sum_k = pq1 + pq2 + pq3  # [seq, batch]

        if mask is not None:
            sum_k = sum_k * mask  # apply mask before summing

        sum_seq = sum_k.sum(dim=0)  # [batch]
        loss = -sum_seq.mean() / float(self.Nmax)

        return loss

    def lk_loss(self, mu, sig_hat):
        """Calculate the KL divergence loss."""
        # Sum over latent dimensions first
        nz_sum = torch.sum(1 + sig_hat - mu**2 - torch.exp(sig_hat), dim=1)
        
        # Take mean over batch
        loss = -nz_sum.mean() / (2 * self.Nz)
        
        return loss

    def forward(self, y, x, mu, sig_hat, R, eta_min, wkl, lenghts, step, kl_min=0.01):
        """Calculate the complete loss with KL annealing."""
        # Process logits to get proper distributions
        y_processed = self.process_logits(y)

        eos = torch.stack([torch.Tensor([0,0,0,0,1])]*x.size(1)).to(x.device).unsqueeze(0) # [seq_len, batch_size, input_size]
        x = torch.cat([x, eos], dim=0) #[seq_len + 1, batch_size, input_size]

        mask = torch.zeros(y.size(0), y.size(1), device=y.device)
        for indice,length in enumerate(lenghts):
            mask[:length,indice] = 1
        #count the 1s in the mask in the first dimension for 1st batch
        lr = self.LsLoss(y_processed, x, mask=mask) + self.LpLoss(y_processed, x, mask=mask)
        
        # Calculate KL divergence loss
        lkl = self.lk_loss(mu, sig_hat)
        
        # Calculate annealing factor
        eta_step = 1.0 - (1.0 - eta_min) * (R ** step)
        
        # Apply KL annealing with minimum threshold
        kl_loss = torch.max(lkl, torch.tensor(kl_min, device=lkl.device))
        annealed_kl = wkl * eta_step * kl_loss
        
        # Total loss
        total_loss = lr + annealed_kl
        
        return total_loss, lr, lkl



#temprature applied during inferance


