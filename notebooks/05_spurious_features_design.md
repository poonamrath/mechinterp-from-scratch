# 05 – Spurious Features Ablation: Design Notes

## Concept

Demonstrate a full "interpretability → action" loop inspired by SHIFT (Marks et al.):
1. Train a classifier with a known spurious cue
2. Identify SAE features correlated with the spurious signal
3. Human-judge features as "task-irrelevant"
4. Ablate those features at inference
5. Measure generalization improvement

---

## Task: Format-Cue Topic Classification (Option B)

- **Labels:** Sports vs Politics
- **Spurious signal:** Training data has format markers
  - Sports: starts with "###"
  - Politics: starts with "@@@"
- **Test set:** No markers (OOD)

This creates a clean, controllable shortcut that the model will exploit.

---

## Metrics

### Classification Metrics

| Metric | What it measures | Why include |
|--------|------------------|-------------|
| **Accuracy** | % correct predictions | Primary metric, easy to interpret |
| **Accuracy gap (train - test)** | Shortcut reliance | Directly shows spurious dependence |
| **Per-class accuracy** | Sports vs politics separately | Ensures we're not just flipping bias |

### Ablation-Specific Metrics

| Metric | What it measures | Why include |
|--------|------------------|-------------|
| **Test accuracy delta** | Improvement after ablation | The main result |
| **Train accuracy delta** | How much train drops | Sanity check — should drop if shortcut removed |
| **# features ablated** | Intervention size | Shows targeted, not brute-force ablation |

### Feature Selection Metrics

| Metric | What it measures | Why include |
|--------|------------------|-------------|
| **Correlation with marker** | How predictive feature is of "###" vs "@@@" | Primary ranking criterion |
| **Feature activation frequency** | How often the feature fires | Optional — helps filter dead features |

---

## Expected Outcomes

- Train accuracy: ~95%+ (model learns the marker shortcut)
- Test accuracy (no markers): ~50-60% (near random without the cue)
- Test accuracy after ablation: Improved (target: 70%+)

---

## Alternative: Option A (Sentiment + Name Cue)

Instead of format markers, use a name cue:
- Positive sentiment always mentions "Alice"
- Negative sentiment always mentions "Bob"
- Test set uses neutral names

---

## References

- [Sparse Feature Circuits (Marks et al.)](https://arxiv.org/abs/2403.19647) — SHIFT method
- [Towards Monosemanticity (Anthropic)](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
