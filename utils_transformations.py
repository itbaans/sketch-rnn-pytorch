import numpy as np

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
        seq = seq.astype('float64')
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data

def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes, max_seq_length):
    """removes too small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def convert_stroke3_to_stroke5(stroke3_sequence):

    if not isinstance(stroke3_sequence, np.ndarray):
        try:
            stroke3_sequence = np.array(stroke3_sequence, dtype=np.float32)
        except ValueError:
            print("Error: Input could not be converted to a NumPy array.")
            return None

    if len(stroke3_sequence.shape) != 2 or stroke3_sequence.shape[1] != 3:
        print(f"Error: Input shape is {stroke3_sequence.shape}, expected (N, 3).")
        return None

    N = stroke3_sequence.shape[0]
    if N == 0:
        print("Warning: Input sequence is empty.")
        # Returning just the start token might be an option, or None
        return None # Or np.array([[0., 0., 1., 0., 0.]], dtype=np.float32)

    # Initialize the output list with the start token
    # Use float type for consistency
    stroke5_list = [[0., 0., 1., 0., 0.]]

    # Iterate through the input stroke-3 sequence
    for i in range(N):
        dx, dy, lifted = stroke3_sequence[i]
        p1, p2, p3 = 0., 0., 0.  # Initialize one-hot state components

        if i == N - 1:
            # This is the very last point in the input sequence
            # It must signify the end of the drawing (p3 = 1)
            p3 = 1.
        else:
            # This is not the last point
            if lifted == 0:
                # Pen stays down (p1 = 1)
                p1 = 1.
            else:
                # Pen lifts up after this point (p2 = 1)
                p2 = 1.

        stroke5_list.append([dx, dy, p1, p2, p3])

    return np.array(stroke5_list, dtype=np.float32)

def pad_sequence(sequence, max_length):
    """Pad sequences to a fixed length."""
    if len(sequence) < max_length:
        padding = np.array([0., 0., 0., 0., 1.], dtype=sequence.dtype)
        return np.vstack((sequence, padding))
    return sequence[:max_length]

def pad_sequences(sequences, max_length):
    """Pad all sequences to the same length."""
    padded_sequences = []
    for seq in sequences:
        padded_seq = pad_sequence(seq, max_length)
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

