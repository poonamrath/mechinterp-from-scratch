from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    model_name: str = "gpt2"     # GPT-2 small
    device: str = "mps"          
    dtype: str = "float16"       # for model forward
    ctx_len: int = 128

    # Hook target for caching / patching
    # We'll hook the output of MLP for a mid layer
    layer: int = 6
    hook_module: str = "transformer.h.6.mlp"  # GPT-2 MLP module path
    hook_point: str = "mlp_out"               # our name, not a model attr

    # Caching
    cache_dir: str = "artifacts/cache"
    cache_dtype: str = "float16"

    # SAE
    d_in: int = 768            # GPT-2 small hidden size
    d_sae: int = 4096
    sae_lr: float = 1e-3
    sae_l1: float = 1e-3