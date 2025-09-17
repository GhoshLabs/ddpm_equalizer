import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from channels import sir_proak, n_tx, n_rx, l_seq, h_proak

T = 200
beta_0 = 0.0001
beta_T = 0.08

def beta_schedule(target_snr, T):
    """
    Compute betas such that cumulative product of alphas (bar_alpha_T)
    ends at target SNR: bar_alpha_T / (1 - bar_alpha_T) = target_snr
    """
    bar_alpha_T = target_snr / (1 + target_snr)

    # Use torch.linspace instead of np.linspace
    log_snr = torch.linspace(math.log(1e6), math.log(target_snr), T)
    snr_t = torch.exp(log_snr)
    bar_alpha_t = snr_t / (1.0 + snr_t)

    alpha_t = torch.zeros(T)
    alpha_t[0] = bar_alpha_t[0]
    for t in range(1, T):
        alpha_t[t] = bar_alpha_t[t] / bar_alpha_t[t - 1]

    beta_t = 1 - alpha_t
    return beta_t

def get_cosine_schedule(T, s=0.008):
    """
    Generates a cosine beta schedule as described in
    "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672)
    """
    timesteps = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f_t = torch.cos(((timesteps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bars = f_t / f_t[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clip(betas, 0.0001, 0.999)
betas = beta_schedule(sir_proak, T).to(torch.double)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def diff_fwd(data_0, betas, T):
    fwd_seq = torch.empty((T, *data_0.shape),dtype=torch.cdouble, device=data_0.device) # Add batch dimension and specify device
    fwd_seq[0] = data_0 # Assign with batch dimension
    betas = betas.to(data_0.device) # Move betas to the same device as data_0
    for t in range(T-1):
        noise = (torch.randn_like(data_0.real) + 1j * torch.randn_like(data_0.imag)) / math.sqrt(2)
        fwd_seq[t+1] = fwd_seq[t]*torch.sqrt(1-betas[t]) + torch.sqrt(betas[t])*noise
    return fwd_seq

class Backbone(nn.Module):
    def __init__(self, n_steps, input_dim = (max(n_rx, n_tx), l_seq+len(h_proak)-1)):
        super().__init__()
        # Assuming input_dim is (channels, height, width) for Conv2d
        # We need to adjust input_dim to be a tuple of (channels, height, width)
        # Based on the usage, it seems input_dim is (batch_size, channels, height, width)
        # Let's assume input_dim is (height, width) and we will use 1 channel
        in_channels = 2 # Two channels for real and imaginary parts
        out_channels = 1 # Output complex number (real and imaginary parts)
        height, width = input_dim
        in_channels = 4 # 2 for x_t (real/imag) + 2 for y (real/imag)
        self.linear_model1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.GELU()
        )
        # Condition time t
        self.embedding_layer = nn.Embedding(n_steps, 256)
        #self.embedding_layer.weight.data = self.embedding_layer.weight.data.double()


        self.linear_model2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Example kernel size and padding
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Example kernel size and padding
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Conv2d(512, 2, kernel_size=3, padding=1), # Output 2 channels for real and imaginary parts
        )
    def forward(self, x, idx, y):
        # Reshape input for Conv2d: (batch_size, channels, height, width)
        # Split complex input into real and imaginary channels
        x_real = x.real.unsqueeze(1)
        x_imag = x.imag.unsqueeze(1)
        
        # Split the condition y and concatenate it as channels
        y_real = y.real.unsqueeze(1)
        y_imag = y.imag.unsqueeze(1)
        x = torch.cat([x_real, x_imag, y_real, y_imag], dim=1) # Concatenate x_t and y

        x = self.linear_model1(x)
        # The embedding needs to be added to the feature maps.
        # We need to expand the embedding to match the spatial dimensions of x.
        emb = self.embedding_layer(idx).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        x = self.linear_model2(x + emb)

        # Split output channels back into real and imaginary parts and combine into complex
        x_real = x[:, 0, :, :].squeeze(1)
        x_imag = x[:, 1, :, :].squeeze(1)
        x = torch.complex(x_real, x_imag)

        return x

def get_loss(model, x_0, t, alpha_bars_t, y):
    # Generate noise
    eps = (torch.randn_like(x_0.real) + 1j * torch.randn_like(x_0.imag)) / math.sqrt(2)
    
    # Apply noise to x_0 to get x_t
    # alpha_bars_t needs to be reshaped to broadcast correctly with x_0
    sqrt_alpha_bar = torch.sqrt(alpha_bars_t).view(-1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bars_t).view(-1, 1, 1)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps

    # Predict noise
    noise_theta = model(x_t, t, y)

    # Calculate MSE for complex numbers by summing the MSE of real and imaginary parts
    loss = F.mse_loss(eps.real, noise_theta.real) + F.mse_loss(eps.imag, noise_theta.imag)
    return loss

@torch.no_grad()
def sample_timestep(model, x, t, y):
    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    predicted_noise = model(x, torch.full((x.shape[0],), t, device=x.device), y)
    x = 1 / (alphas[t] ** 0.5) * (x - (1 - alphas[t]) / ((1-alpha_bars[t]) ** 0.5) * predicted_noise)
    if t > 0:
        noise = (torch.randn_like(x.real) + 1j * torch.randn_like(x.imag)) / math.sqrt(2)
        x += torch.sqrt(betas[t]) * noise
    return x