# src/cache_activations.py
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.model import load_model_and_tokenizer, get_module_by_name

def iter_token_batches(tok, text: str, ctx_len: int):
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    # chop into fixed blocks
    n = (len(ids) // ctx_len) * ctx_len
    ids = ids[:n].view(-1, ctx_len)
    for i in range(ids.shape[0]):
        yield ids[i:i+1]  # batch=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tokens", type=int, default=250_000)
    parser.add_argument("--out", type=str, default="artifacts/cache/gpt2_l6_mlpout_fp16.mmap")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, tok, device = load_model_and_tokenizer(cfg)
    mlp = get_module_by_name(model, cfg.hook_module)

    # Tiny corpus for first pass (replace later with streaming dataset)
    # Keep it local & deterministic for now.
    base = (
        "Alice and Bob went to the store. "
        "They talked about books and music. "
        "Carol and Dave played chess. "
        "Eve wrote a letter. "
    )
    text = base * 20000  # enough characters to get a few hundred K tokens

    # Prepare memmap: [n_tokens, d_in]
    n_tokens = args.n_tokens
    d_in = cfg.d_in
    mmap = np.memmap(args.out, dtype=np.float16, mode="w+", shape=(n_tokens, d_in))

    write_ptr = 0

    def hook_fn(module, inp, out):
        # out: [1, seq, hidden]
        nonlocal write_ptr
        x = out.detach().to("cpu", dtype=torch.float16).numpy()[0]  # [seq, hidden]
        seq = x.shape[0]
        take = min(seq, n_tokens - write_ptr)
        if take > 0:
            mmap[write_ptr:write_ptr+take] = x[:take]
            write_ptr += take

    handle = mlp.register_forward_hook(hook_fn)

    # Iterate prompts in ctx blocks
    pbar = tqdm(total=n_tokens, desc="Caching activations")
    for batch_ids in iter_token_batches(tok, text, cfg.ctx_len):
        if write_ptr >= n_tokens:
            break
        batch_ids = batch_ids.to(device)
        with torch.no_grad():
            _ = model(input_ids=batch_ids)
        pbar.update(min(cfg.ctx_len, n_tokens - (write_ptr - cfg.ctx_len)))
    pbar.close()

    handle.remove()
    mmap.flush()
    print(f"Wrote {write_ptr} token activations to {args.out}")

if __name__ == "__main__":
    main()