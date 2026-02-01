# mechinterp-from-scratch

A hands-on, local-first project for learning **mechanistic interpretability** by
reverse-engineering small transformer models, training **sparse autoencoders (SAEs)**,
and performing **causal interventions** on internal representations.

This repo is designed to run **entirely on Apple Silicon (M1, 16GB RAM)** using the MPS backend.

---

## Motivation

Modern foundation models are powerful but opaque.  
Mechanistic interpretability (mechinterp) aims to **open the black box** by identifying
internal features and circuits, then **testing hypotheses causally** via interventions
(activation patching, ablations, feature steering).

This project is inspired by:
- transformer circuit analysis (e.g. IOI)
- sparse feature discovery with SAEs
- interpretability-to-action pipelines (e.g. feature ablation improves generalization)

The idea is to **personal mechinterp lab** that is simple enough to understand end-to-end, and extend toward:
- alignment & safety research
- multimodal interpretability
- double click on specific sorts of models (eg: biology foundation models)

---

## What this repo contains

The project progresses in three stages:

1. **Circuit-level mechinterp**
   - Reproduce a minimal IOI-style causal patching experiment
   - Identify which internal activations matter for behavior

2. **Feature discovery with Sparse Autoencoders**
   - Cache internal activations from a mid-layer of GPT-2 small
   - Train an SAE to discover sparse, interpretable features
   - Inspect features via top-activating examples

3. **Interpretability → Action**
   - Identify spurious or shortcut features
   - Ablate those features at inference time
   - Measure behavioral / generalization changes

All experiments are runnable locally.

---

## Hardware & constraints

Tested configuration:
- Apple M1
- 16GB RAM
- macOS
- PyTorch MPS backend

Design constraints:
- Small models only (GPT-2 small / Pythia-scale)
- Single-layer activation caching
- Aggressive use of disk-backed memmaps
- SAE training on CPU for stability

## Repo structure
mechinterp-from-scratch/
README.md
requirements.txt
notebooks/
00_setup_mps.ipynb # environment & MPS sanity checks
10_ioi_minimal.ipynb # minimal IOI-style causal patching
src/
config.py # central experiment config
model.py # model + tokenizer loading
ioi.py # IOI-style dataset
patching.py # activation patching utilities
cache_activations.py # disk-backed activation caching
sae_model.py # sparse autoencoder
train_sae.py # SAE training loop
smoke_test.py # end-to-end sanity test

Running smoke test (model+hook) should return these outputs
device: mps
logits shape
activation hook shape

Things notebooks/01_ioi_minimal.ipynb does:
generate clean vs corrupted IOI prompts
measure target token log-probs
patch a mid-layer activation from clean → corrupt
observe causal recovery of correct behavior

Caching activations with `python -m src.cache_activations --n_tokens 25000`
writes a disk-backed memmap of activations:
one layer; one hook site; float16; ~400-500Mb

Train a sparse autoencoder with `python -m src.train_sae --steps 2000`
input_dim: 768; dictionary size: 4096; reconstruction + L1 sparsity loss

Next steps:
- Feature browser notebook (top-activating token windows)
- SHIFT-style spurious feature ablation experiments
- Multi-layer or residual-stream SAEs
- Multimodal interpretability (e.g. Gemma-style models)
- Feature-level steering & editing