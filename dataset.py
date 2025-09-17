import torch
import math
from channels import n_tx, l_seq, M, d
import numpy as np

class Datasets(torch.utils.data.Dataset):
    def __init__(self, M, d, h, n_tx, n_rx, l_seq, total_len=100):
        self.d = d
        self.M = M
        self.h = h
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.l_seq = l_seq
        self.total_len = total_len

    @property
    def get_a_m_list(self):
        return [(2*m-1-math.sqrt(M))*d for m in range(int(math.sqrt(M)))]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        a_m_list = [(2*m-1-math.sqrt(self.M))*self.d for m in range(int(math.sqrt(self.M)))]
        x = torch.tensor(np.random.choice(a_m_list, (self.n_tx,self.l_seq)) + 1j*np.random.choice(a_m_list, (self.n_tx,self.l_seq)), dtype=torch.cdouble) # Changed to cdouble
        data_x = torch.zeros((max(self.n_tx, self.n_rx), self.l_seq + self.h.shape[-1] - 1),dtype=torch.cdouble) # Changed to cdouble
        data_x[0:n_tx,0:l_seq] = x
        
        # Also generate the received signal y
        data_y = torch.zeros_like(data_x)
        for i in range(self.l_seq): # Iterate over the original sequence length
            data_y[:, i:i+self.h.shape[-1]] += torch.matmul(torch.t(data_x[:,i]),self.h)

        return data_x, data_y