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

#i generated this using Claude
class my_bidirectional_lstm(nn.Module):
    def __init__(self, inp_size, h_size, out_size, merge_mode='concat'):
        super(my_bidirectional_lstm, self).__init__()
        # initializations
        self.inp_size = inp_size
        self.h_size = h_size
        self.merge_mode = merge_mode
        self.fc = None
        
        # Forward and backward LSTM cells
        self.forward_lstm_cell = my_lstm_cell(inp_size, h_size)
        self.backward_lstm_cell = my_lstm_cell(inp_size, h_size)
        
        # Output projection depends on merge mode
        if out_size != None:
            if merge_mode == 'concat':
                self.fc = nn.Linear(h_size * 2, out_size)
            else:  # sum, mul, avg
                self.fc = nn.Linear(h_size, out_size)

    def forward(self, x):  # x is seq of inputs with shape [seq_len, batch_size, inp_size]
        batch_size = x.size(1)
        h_size = self.h_size
        seq_len = len(x)
        
        # Initialize hidden and cell states for both directions
        h_forward = torch.zeros(batch_size, h_size, device=x.device)
        c_forward = torch.zeros(batch_size, h_size, device=x.device)
        h_backward = torch.zeros(batch_size, h_size, device=x.device)
        c_backward = torch.zeros(batch_size, h_size, device=x.device)
        
        # Store outputs for both directions
        outputs_forward = []
        outputs_backward = []
        
        # Forward direction processing
        for t in range(seq_len):
            c_forward, h_forward = self.forward_lstm_cell(x[t], c_forward, h_forward)
            outputs_forward.append(h_forward)
        
        # Backward direction processing
        for t in range(seq_len - 1, -1, -1):
            c_backward, h_backward = self.backward_lstm_cell(x[t], c_backward, h_backward)
            outputs_backward.insert(0, h_backward)  # Insert at beginning to maintain sequence order
        
        # Combine the forward and backward outputs
        if self.fc != None:
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

        return h_forward, h_backward