# src/model.py
# load GPT-2 on MPS + a hook helper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device(cfg):
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_dtype(cfg):
    if cfg.dtype == "float16":
        return torch.float16
    if cfg.dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32

def load_model_and_tokenizer(cfg):
    device = get_device(cfg)
    dtype = get_dtype(cfg)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    # GPT-2 has no pad token by default; set for batching.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.eval()

    # Move to device; on MPS float16 is usually fine for forward
    model.to(device=device, dtype=dtype)

    return model, tok, device

def get_module_by_name(model, module_name: str):
    cur = model
    for part in module_name.split("."):
        cur = getattr(cur, part)
    return cur