import torch

M=64
d=2
n_tx = 4
n_rx = 4
l_seq = 10

h_proak = [0.407, 0.815, 0.407]
def get_h(tx, rx, h):
    h_mat = torch.zeros(tx, rx, len(h), dtype=torch.cdouble)
    for i in range(min(tx,rx)):
        h_mat[i,i,:] = torch.tensor(h)
    #h_mat += torch.rand(h_mat.shape) + 1j*torch.rand(h_mat.shape)
    return h_mat
h = get_h(n_tx, n_rx, h_proak)
#print(h.shape)

def compute_sir(h):
    h = torch.tensor(h, dtype=torch.float64)
    h0 = h[0]
    isi_power = torch.sum(torch.abs(h[1:])**2)
    signal_power = torch.abs(h0)**2
    return signal_power / isi_power if isi_power > 0 else float('inf')

sir_proak = compute_sir(h)