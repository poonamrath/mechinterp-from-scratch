# src/patching.py
import torch
from src.model import get_module_by_name

@torch.no_grad()
def logprob_of_target_next_token(model, tok, device, prompt: str, target: str):
    enc = tok(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits  # [1, seq, vocab]
    next_logits = logits[0, -1, :]  # last position predicts next token
    target_id = tok(" " + target, add_special_tokens=False)["input_ids"]
    if len(target_id) != 1:
        # fallback: take first token (common for names but not guaranteed)
        target_id = target_id[0]
    else:
        target_id = target_id[0]
    logp = torch.log_softmax(next_logits, dim=-1)[target_id].item()
    return logp

@torch.no_grad()
def run_with_optional_patch(model, tok, device, cfg, prompt: str, patch_tensor=None):
    """
    If patch_tensor is provided, it is inserted at cfg.hook_module output.
    patch_tensor shape must match the module output: [batch, seq, hidden]
    """
    mlp = get_module_by_name(model, cfg.hook_module)

    def hook_fn(module, inp, out):
        if patch_tensor is None:
            return out
        # Replace entire activation (simple first version)
        return patch_tensor

    handle = mlp.register_forward_hook(hook_fn)
    enc = tok(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits
    handle.remove()
    return logits

@torch.no_grad()
def get_hook_activation(model, tok, device, cfg, prompt: str):
    mlp = get_module_by_name(model, cfg.hook_module)
    acts = {}

    def hook_fn(module, inp, out):
        acts["x"] = out.detach().clone()

    h = mlp.register_forward_hook(hook_fn)
    enc = tok(prompt, return_tensors="pt").to(device)
    _ = model(**enc)
    h.remove()
    return acts["x"]
