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

## Resources and Inspiration
- IOI paper (Interpretability in the wild - a circuit for indirect object identification in GPT-2 small): https://arxiv.org/abs/2211.00593
- On the biology of an LLM: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- https://docs.google.com/document/d/1dKAjGPdKdyemy5rZUI96nYwNDonKfXM6H7p58FF5rcE/edit?usp=sharing
- https://www.alignmentforum.org/posts/6P8GYb4AjtPXx6LLB/tips-and-code-for-empirical-research-workflows
- https://www.lesswrong.com/posts/wSKPuBfgkkqfTpmWJ/auditing-language-models-for-hidden-objectives
- relationship between mechinterp and alignment (empirical alignment research vs. mechanistic interpretability). for example, anthropic's model organisms work, neel nanda's move toward pragmatic interpretability (https://www.youtube.com/watch?v=k93o4R145Os)... 
- mechinterp's primary goals: *why* the models *do* what they do, *why* they do one thing vs another, *how* they're doing it
TODO: add a mindmap of related concepts that intuitively seem related

## Grounding in domain
Personal fave: bio (the "language" of bio is different from language models). what does this mean in how we would approach interpretability?
- Push on my intuition: any sufficiently capable system trained under constraint will learn an *internal* structure that exceeds its *explicit objective* and you only see it if you look inside. Examples: LLMs do it (altered behavior when they're being evaluated vs when they think they're not); sorting algos does it (clustering, even though explicit objective is sorting); biological systems *are* pretty much it, such that a virtual cell model may predict perturbation response accurately while internally representing something else that's salient to the cell's survival....(i guess we acknowledge this to be the case)