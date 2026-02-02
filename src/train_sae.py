# src/train_sae.py
import os
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import trange

from src.config import Config
from src.sae_model import SAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmap", type=str, default="artifacts/cache/gpt2_l6_mlpout_fp16.mmap")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--save", type=str, default="artifacts/sae/sae.pt")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # Load memmap
    data = np.memmap(args.mmap, dtype=np.float16, mode="r")
    n = data.shape[0] // cfg.d_in
    data = data.reshape(n, cfg.d_in)

    device = torch.device("cpu")  # SAE training on CPU is fine & stable on 16GB
    sae = SAE(cfg.d_in, cfg.d_sae).to(device)
    opt = AdamW(sae.parameters(), lr=cfg.sae_lr)

    def sample_batch(batch_size):
        idx = np.random.randint(0, n, size=(batch_size,))
        x = torch.from_numpy(data[idx].astype(np.float32))  # fp32 for stability
        return x.to(device)

    for step in trange(args.steps, desc="Training SAE"):
        x = sample_batch(args.batch)
        x_hat, a = sae(x)

        recon = torch.mean((x_hat - x) ** 2)
        sparsity = torch.mean(torch.abs(a))
        loss = recon + cfg.sae_l1 * sparsity

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step} | recon {recon.item():.6f} | l1 {sparsity.item():.6f} | loss {loss.item():.6f}")

    torch.save({"state_dict": sae.state_dict(), "cfg": cfg}, args.save)
    print("saved:", args.save)

if __name__ == "__main__":
    main()
