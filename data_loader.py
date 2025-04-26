import numpy as np
import torch
import hyper_param as hp
from torch.autograd import Variable

def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.Nmax and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data


dataset = np.load("sketch-rrnn/data/cat.npz", encoding='latin1', allow_pickle=True)
data = dataset['train']
data = purify(data)
data = normalize(data)

data_test = dataset['test']
data_test = purify(data_test)
data_test = normalize(data_test)

hp.Nmax = max_size(data)

def make_batch(batch_size, device = 'cpu', data_c = None, test=False):
    batch_idx = np.random.choice(len(data),batch_size)
    if data_c is None:
        if not test:
            batch_sequences = [data[idx] for idx in batch_idx]
        else:
            batch_idx = np.random.choice(len(data_test),batch_size)
            batch_sequences = [data_test[idx] for idx in batch_idx] 
    else:
        print(data_c)
        data_c = purify(data_c)
        print(data_c)
        data_c = normalize(data_c)
        hp.Nmax = max_size(data_c)
        batch_sequences = data_c
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:,0])
        new_seq = np.zeros((hp.Nmax,5))
        new_seq[:len_seq,:2] = seq[:,:2]
        new_seq[:len_seq-1,2] = 1-seq[:-1,2]
        new_seq[:len_seq,3] = seq[:,2]
        new_seq[(len_seq-1):,4] = 1
        new_seq[len_seq-1,2:4] = 0
        lengths.append(len(seq[:,0]))
        strokes.append(new_seq)
        indice += 1

    #limit to 1 strok

    if device == 'cuda':
        batch = torch.from_numpy(np.stack(strokes,1)).cuda().float()
    else:
        batch = torch.from_numpy(np.stack(strokes,1)).float()
    return batch, lengths
