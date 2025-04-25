import torch
from my_skecth_rnn import sketch_rnn, SKETCH_RNN_LOSS
from data_loader import make_batch
import hyper_param as hp
import torch.nn as nn


#test a pass through the model
def test_model():
    # Create a dummy input tensor with the shape [seq, batch, 6*M + 3]
    seq = 10
    batch = 5
    M = 20
    dummy_input = torch.randn(seq, batch, 5)

    # Create a dummy mu and sig_hat tensor with the shape [batch, Nz]
    Nz = 128
    # Create an instance of the sketch_rnn model
    model = sketch_rnn(input_size=5, enc_hsize=256, dec_hsize=512, z_size=Nz, dec_out_size=6 * M + 3)

    # Forward pass through the model
    output, mu, sig_hat = model(dummy_input)

    # Print the shapes of the output, mu, and sig_hat tensors
    print("Output shape:", output.shape)  # Expected: [seq, batch, 6*M + 3]
    print("Mu shape:", mu.shape)  # Expected: [batch, Nz]
    print("Sig_hat shape:", sig_hat.shape)  # Expected: [batch, Nz]
    print("Mu:", mu)
    print("Sig_hat:", sig_hat)
    print("Output:", output)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = sketch_rnn(input_size=5, enc_hsize=hp.enc_hidden_size, dec_hsize=hp.dec_hidden_size, z_size=hp.Nz, dec_out_size=6 * 20 + 3,
                   rec_dropout=hp.dropout)#, layer_norm=True)
model = model.to(device)

loss_fnc = SKETCH_RNN_LOSS(Nmax=hp.Nmax, M=hp.M, Nz=hp.Nz)

enc_opt = torch.optim.Adam(model.encoder.parameters(), lr=hp.lr)
dec_opt = torch.optim.Adam(model.decoder.parameters(), lr=hp.lr)

def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr']>hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

def training(model, loss_fnc, enc_opt, dec_opt, num_epochs=10):
    for epoch in range(num_epochs):
        model.encoder.train()
        model.decoder.train()
        
        batch, _ = make_batch(hp.batch_size, device='cuda')
        y, mu, sig_hat, _ = model(batch)

        loss = loss_fnc(y, batch, mu, sig_hat, hp.R, hp.eta_min, hp.wKL, epoch, hp.KL_min)

        enc_opt.zero_grad()
        dec_opt.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(model.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm_(model.decoder.parameters(), hp.grad_clip)

        enc_opt.step()
        dec_opt.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # if epoch%1==0:
        #     enc_opt = lr_decay(enc_opt)
        #     dec_opt = lr_decay(dec_opt)

# training(model, loss_fnc, enc_opt, dec_opt, num_epochs=5000)

# #save the model
# torch.save(model.state_dict(), 'sketch_rnn_model.pth')

old_model = sketch_rnn(input_size=5, enc_hsize=hp.enc_hidden_size, dec_hsize=hp.dec_hidden_size, z_size=hp.Nz, dec_out_size=6 * 20 + 3,
                   rec_dropout=hp.dropout)#, layer_norm=True)
old_model.load_state_dict(torch.load('sketch_rnn_model.pth'))

old_model.to(device)
old_model.eval()
batch, _ = make_batch(1, device='cuda')

old_model(batch, hp.Nmax, hp.temperature, hp.M)










