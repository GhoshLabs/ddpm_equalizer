import matplotlib.pyplot as plt
import torch

def plot_scatter(x):
    # Ensure the input is a CPU tensor for plotting
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    # Convert to plain Python lists to avoid any implicit numpy conversion by matplotlib
    real_parts = x.real.flatten().tolist()
    imag_parts = x.imag.flatten().tolist()
    plt.scatter(real_parts, imag_parts)
    plt.show()