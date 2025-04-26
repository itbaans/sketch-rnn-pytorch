import tkinter as tk
import torch
from my_skecth_rnn import sketch_rnn, SKETCH_RNN_LOSS
from data_loader import make_batch
import hyper_param as hp
import torch.nn as nn


class DrawCapture:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Draw here!')
        self.canvas = tk.Canvas(self.root, bg='white', width=400, height=400)
        self.canvas.pack()
        
        self.old_x = None
        self.old_y = None
        self.drawing = []  # Will store (dx, dy, pen_state)
        
        self.canvas.bind('<ButtonPress-1>', self.pen_down)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.pen_up)
        
        self.root.bind('<Return>', self.finish)  # Press Enter to finish
        
        self.root.mainloop()
    
    def pen_down(self, event):
        self.old_x = event.x
        self.old_y = event.y
        self.drawing.append([0, 0, 1])  # Pen down
    
    def draw(self, event):
        dx = event.x - self.old_x
        dy = event.y - self.old_y
        self.drawing.append([dx, dy, 1])  # Pen moving (pen_state = 1)
        self.old_x = event.x
        self.old_y = event.y
        self.canvas.create_line(event.x - dx, event.y - dy, event.x, event.y, fill='black', width=2)
    
    def pen_up(self, event):
        self.drawing.append([0, 0, 0])  # Pen up
    
    def finish(self, event):
        import numpy as np
        seq = np.array(self.drawing)
        print("Captured sequence:")
        print(seq)
        print("Length of sequence:", seq.shape[0])
        # Here you could call your make_batch function or model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        old_model = sketch_rnn(input_size=5, enc_hsize=hp.enc_hidden_size, dec_hsize=hp.dec_hidden_size, z_size=hp.Nz, dec_out_size=6 * 20 + 3,
                   rec_dropout=hp.dropout)#, layer_norm=True)
        old_model.load_state_dict(torch.load('sketch_rnn_model.pth'))

        old_model.to(device)
        old_model.eval()
        batch, _ = make_batch(1, device='cuda', data_c = [seq])

        old_model(batch, hp.Nmax, hp.temperature, hp.M)
        self.root.destroy()

DrawCapture()
