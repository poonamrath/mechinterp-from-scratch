# minimal sae
# src/sae_model.py
import torch
import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_sae, bias=True)
        self.dec = nn.Linear(d_sae, d_in, bias=False)

    def forward(self, x):
        # x: [batch, d_in]
        z = self.enc(x)
        a = torch.relu(z)  # simple sparse-ish nonlinearity starter
        x_hat = self.dec(a)
        return x_hat, a
