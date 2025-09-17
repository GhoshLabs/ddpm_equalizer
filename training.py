import torch
from torch.optim import Adam
from dataset import Datasets
from diff_model import Backbone, get_loss, T, alpha_bars
from channels import h, n_tx, n_rx, l_seq, M, d
import os

model = Backbone(T)
model

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# Cast the model parameters to double
model = model.to(torch.double)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100
batchsize = 10

dataloader = torch.utils.data.DataLoader(Datasets(M, d, h, n_tx, n_rx, l_seq, total_len=batchsize*100), batch_size=batchsize)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()
      
      x_0, y = batch # Unpack the batch
      x_0 = x_0.to(device)
      y = y.to(device)

      t = torch.randint(0, T, (batchsize,), device=device).long()
      alpha_bars_t = alpha_bars.to(device)[t]
      loss = get_loss(model, x_0, t, alpha_bars_t, y) # Pass y to get_loss
      loss.backward()
      optimizer.step()

      if epoch % 5 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

print("\nTraining complete.")

# Create a directory to save models if it doesn't exist
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, 'ddpm_equalizer_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model state dictionary saved to {model_path}")