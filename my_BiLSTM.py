#NOT USED
import numpy as np
import os
import torch 
import torch.nn as nn
import torch.optim as optim

class my_lstm_cell(nn.Module):
    def __init__(self, inp_size, h_size):
        super(my_lstm_cell, self).__init__()
        # initializations
        self.inp_size = inp_size
        self.h_size = h_size
        # Input weights
        self.x_w_i = nn.Linear(inp_size, h_size)  # input gate
        self.x_w_f = nn.Linear(inp_size, h_size)  # forget gate
        self.x_w_o = nn.Linear(inp_size, h_size)  # output gate
        self.x_w_c = nn.Linear(inp_size, h_size)  # cell gate
        
        # Hidden weights
        self.h_w_i = nn.Linear(h_size, h_size)
        self.h_w_f = nn.Linear(h_size, h_size)
        self.h_w_o = nn.Linear(h_size, h_size)
        self.h_w_c = nn.Linear(h_size, h_size)  # cell gate

    def forward(self, x, c, h):
        # Gates
        i = torch.sigmoid(self.x_w_i(x) + self.h_w_i(h))  # input gate
        f = torch.sigmoid(self.x_w_f(x) + self.h_w_f(h))  # forget gate
        o = torch.sigmoid(self.x_w_o(x) + self.h_w_o(h))  # output gate
        
        # New cell candidate (missing in original)
        c_tilde = torch.tanh(self.x_w_c(x) + self.h_w_c(h))
        
        # Update cell state
        new_c = (f * c) + (i * c_tilde) 
        
        # Generate new hidden state
        new_h = o * torch.tanh(new_c)
        
        return new_c, new_h
    

class my_lstm(nn.Module):
    def __init__(self, inp_size, h_size, out_size):
        super(my_lstm, self).__init__()
        # initializations
        self.inp_size = inp_size
        self.h_size = h_size
        
        self.lstm_cell = my_lstm_cell(inp_size, h_size)
        self.fc = nn.Linear(h_size, out_size)  # fixed typo: Linaer -> Linear

    def forward(self, x, h=None, c=None):  # x is seq of inputs with shape [seq_len, batch_size, inp_size]
        batch_size = x.size(1)
        
        # Initialize hidden and cell states
        if(h == None and c == None):
            h = torch.zeros(batch_size, self.h_size, device=x.device)
            c = torch.zeros(batch_size, self.h_size, device=x.device)
        
        outputs = []
        
        for t in range(len(x)):
            c, h = self.lstm_cell(x[t], c, h)  # was using x instead of x[t]
            outputs.append(self.fc(h))  # should use h not o for output projection
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=0)
            
        return outputs, (h, c)  # Return outputs and final states (h, c)

# generated using claude, to add bidirectional, recurrent dropout, layer norm
class UnifiedLSTM(nn.Module):
    def __init__(self, inp_size, h_size, out_size, bidirectional=False, merge_mode='concat',
                 dropout=0.0, recurrent_dropout=0.0, layer_norm=False):
        super(UnifiedLSTM, self).__init__()
        # initializations
        self.inp_size = inp_size
        self.h_size = h_size
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode if bidirectional else None
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.layer_norm = layer_norm
        
        # Layer normalization layers
        if layer_norm:
            self.ln_forward_h = nn.LayerNorm(h_size)
            self.ln_forward_c = nn.LayerNorm(h_size)
            if bidirectional:
                self.ln_backward_h = nn.LayerNorm(h_size)
                self.ln_backward_c = nn.LayerNorm(h_size)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.recurrent_dropout_mask_forward = None
        self.recurrent_dropout_mask_backward = None
        
        # Forward LSTM cell (used in both unidirectional and bidirectional)
        self.forward_lstm_cell = my_lstm_cell(inp_size, h_size)
        
        # Backward LSTM cell (only used in bidirectional)
        if bidirectional:
            self.backward_lstm_cell = my_lstm_cell(inp_size, h_size)
            
            # Output projection depends on merge mode for bidirectional
            if merge_mode == 'concat':
                self.fc = nn.Linear(h_size * 2, out_size)
            else:  # sum, mul, avg
                self.fc = nn.Linear(h_size, out_size)
        else:
            # Simple output projection for unidirectional
            self.fc = nn.Linear(h_size, out_size)

    def _create_recurrent_dropout_mask(self, batch_size, device):
        """Create recurrent dropout masks for this batch if needed"""
        if self.recurrent_dropout > 0 and self.training:
            self.recurrent_dropout_mask_forward = torch.bernoulli(
                torch.ones(batch_size, self.h_size, device=device) * (1 - self.recurrent_dropout)
            ) / (1 - self.recurrent_dropout)
            
            if self.bidirectional:
                self.recurrent_dropout_mask_backward = torch.bernoulli(
                    torch.ones(batch_size, self.h_size, device=device) * (1 - self.recurrent_dropout)
                ) / (1 - self.recurrent_dropout)
    
    def forward(self, x, h=None, c=None):  # x is seq of inputs with shape [seq_len, batch_size, inp_size]
        batch_size = x.size(1)
        seq_len = len(x)
        
        # Apply input dropout if specified
        if self.dropout_layer is not None and self.training:
            x = torch.stack([self.dropout_layer(x_t) for x_t in x])
        
        # Create recurrent dropout masks for this sequence
        self._create_recurrent_dropout_mask(batch_size, x.device)
        
        if self.bidirectional:
            # Initialize hidden and cell states for both directions
            h_forward = torch.zeros(batch_size, self.h_size, device=x.device)
            c_forward = torch.zeros(batch_size, self.h_size, device=x.device)
            h_backward = torch.zeros(batch_size, self.h_size, device=x.device)
            c_backward = torch.zeros(batch_size, self.h_size, device=x.device)
            
            # Allow pre-initialized states if provided
            if h is not None and c is not None:
                if isinstance(h, tuple) and len(h) == 2:
                    h_forward, h_backward = h
                if isinstance(c, tuple) and len(c) == 2:
                    c_forward, c_backward = c
            
            # Apply layer normalization to initial states if specified
            if self.layer_norm:
                h_forward = self.ln_forward_h(h_forward)
                c_forward = self.ln_forward_c(c_forward)
                h_backward = self.ln_backward_h(h_backward)
                c_backward = self.ln_backward_c(c_backward)
            
            # Store outputs for both directions
            outputs_forward = []
            outputs_backward = []
            
            # Forward direction processing
            for t in range(seq_len):
                c_forward, h_forward = self.forward_lstm_cell(x[t], c_forward, h_forward)
                
                # Apply layer normalization if specified
                if self.layer_norm:
                    h_forward = self.ln_forward_h(h_forward)
                    c_forward = self.ln_forward_c(c_forward)
                
                # Apply recurrent dropout if specified
                if self.recurrent_dropout > 0 and self.training:
                    h_forward = h_forward * self.recurrent_dropout_mask_forward
                
                outputs_forward.append(h_forward)
            
            # Backward direction processing
            for t in range(seq_len - 1, -1, -1):
                c_backward, h_backward = self.backward_lstm_cell(x[t], c_backward, h_backward)
                
                # Apply layer normalization if specified
                if self.layer_norm:
                    h_backward = self.ln_backward_h(h_backward)
                    c_backward = self.ln_backward_c(c_backward)
                
                # Apply recurrent dropout if specified
                if self.recurrent_dropout > 0 and self.training:
                    h_backward = h_backward * self.recurrent_dropout_mask_backward
                
                outputs_backward.insert(0, h_backward)  # Insert at beginning to maintain sequence order
            
            # Combine the forward and backward outputs
            combined_outputs = []
            for t in range(seq_len):
                if self.merge_mode == 'concat':
                    combined = torch.cat([outputs_forward[t], outputs_backward[t]], dim=1)
                elif self.merge_mode == 'sum':
                    combined = outputs_forward[t] + outputs_backward[t]
                elif self.merge_mode == 'mul':
                    combined = outputs_forward[t] * outputs_backward[t]
                elif self.merge_mode == 'avg':
                    combined = (outputs_forward[t] + outputs_backward[t]) / 2
                else:
                    raise ValueError(f"Unsupported merge mode: {self.merge_mode}")
                
                combined_outputs.append(self.fc(combined))
            
            # Stack outputs along sequence dimension
            outputs = torch.stack(combined_outputs, dim=0)
            
            # Return outputs and final states from both directions
            final_states = (
                (h_forward, h_backward),  # final hidden states
                (c_forward, c_backward)   # final cell states
            )
                
            return outputs, final_states
        
        else:
            # Unidirectional LSTM mode
            # Initialize hidden and cell states if not provided
            if h is None or c is None:
                h = torch.zeros(batch_size, self.h_size, device=x.device)
                c = torch.zeros(batch_size, self.h_size, device=x.device)
            
            # Apply layer normalization to initial states if specified
            if self.layer_norm:
                h = self.ln_forward_h(h)
                c = self.ln_forward_c(c)
            
            outputs = []
            
            for t in range(seq_len):
                c, h = self.forward_lstm_cell(x[t], c, h)
                
                # Apply layer normalization if specified
                if self.layer_norm:
                    h = self.ln_forward_h(h)
                    c = self.ln_forward_c(c)
                
                # Apply recurrent dropout if specified
                if self.recurrent_dropout > 0 and self.training:
                    h = h * self.recurrent_dropout_mask_forward
                
                outputs.append(self.fc(h))
            
            # Stack outputs along sequence dimension
            outputs = torch.stack(outputs, dim=0)
                
            return outputs, (h, c)  # Return outputs and final states (h, c)
