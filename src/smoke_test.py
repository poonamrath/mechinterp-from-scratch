"""
Smoke test to verify the mechanistic interpretability setup works.

This script:
1. Loads GPT-2 (small) with config settings onto MPS device
2. Registers a forward hook on the MLP at layer 6 to capture activations
3. Runs a forward pass on sample text

Expected output:
- device: mps
- logits shape: (1, seq_len, vocab_size)
- hook activation shape: (1, seq_len, 768)

If this runs without errors, your environment is ready for:
- Activation caching
- Activation patching
- SAE training

Usage:
    python -m src.smoke_test
"""
import torch
from src.config import Config
from src.model import load_model_and_tokenizer, get_module_by_name

def main():
    cfg = Config()
    model, tok, device = load_model_and_tokenizer(cfg)

    text = "Hello mechanistic interpretability!"
    enc = tok(text, return_tensors="pt").to(device)

    acts = {}

    def hook_fn(module, inp, out):
        # out shape: [batch, seq, hidden]
        acts["mlp_out"] = out.detach()

    mlp = get_module_by_name(model, cfg.hook_module)
    handle = mlp.register_forward_hook(hook_fn)

    with torch.no_grad():
        out = model(**enc)

    handle.remove()

    print("device:", device)
    print("logits:", tuple(out.logits.shape))
    print("hook activation:", tuple(acts["mlp_out"].shape))

if __name__ == "__main__":
    main()