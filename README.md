# RoPE Float32 Precision Fix in Large Language Model Training

**Discovery:** Rotary Position Embeddings (RoPE) computed in bfloat16 cause deterministic NaN gradients during large language model training, consistently appearing between steps 5,000–6,000 across independent restarts.

**Author:** Prashant Shewale, Pune, India  
**Project:** MAX — A 9.13B parameter language model trained from scratch on Google TPUs  
**Date:** 2026

---

## The Bug

During training of a 9.13B parameter transformer model on Google TPU v4-8 hardware, NaN (Not a Number) loss values appeared consistently at step 5,750 across three independent restarts with identical hyperparameters.

**Before fix — training log (real data):**
```
Step 5,700:  loss = 4.58
Step 5,750:  loss = NaN  ← consistent across 3 restarts
Step 5,800:  loss = NaN
Step 5,850:  loss = NaN
```

The consistency across restarts ruled out random noise. This was a deterministic, reproducible failure — the hallmark of a systematic numerical precision issue.

---

## Root Cause

RoPE (Rotary Position Embedding) encodes positional information using sinusoidal functions at varying frequencies. At high frequencies (short-period sinusoids), the values cycle rapidly and require fine-grained numerical precision to represent correctly.

**bfloat16** uses 8 bits for the exponent and only 7 bits for the mantissa (significand), giving approximately 2-3 decimal digits of precision.

**float32** uses 8 bits for the exponent and 23 bits for the mantissa, giving approximately 7 decimal digits of precision.

When RoPE frequencies are computed in bfloat16, the high-frequency components are rounded to zero or to incorrect values. This causes the attention mechanism to receive corrupted positional information, which propagates through backpropagation, causing gradient overflow and eventual NaN collapse.

The NaN appears specifically at steps 5,000–6,000 because this is when the optimizer (AdamW) has accumulated enough gradient history that the corrupted positional signal amplifies into a catastrophic overflow.

---

## The Fix

Keep RoPE computation in float32 regardless of the model's overall training precision.

**Buggy code:**
```python
def precompute_rope(head_dim, seq_len, base=10000):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)

# Bug: casting RoPE outputs to bfloat16
cos = cos.astype(jnp.bfloat16)
sin = sin.astype(jnp.bfloat16)
```

**Fixed code (model.py):**
```python
def precompute_rope(head_dim, seq_len, base=10000):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)

# Fix: RoPE stays in float32 — never cast to bfloat16
# "Keep RoPE in float32 — bfloat16 loses precision at high frequencies"
cos, sin = precompute_rope(self.cfg.head_dim, T, self.cfg.rope_base)
```

---

## Additional Fixes Applied

Three related numerical stability fixes were identified and applied together:

**Fix 1 — RoPE precision (primary fix):**
Keep cos/sin tensors in float32. Never cast to bfloat16.

**Fix 2 — Causal mask value:**
```python
# Buggy — causes NaN in softmax when combined with precision issues
causal = jnp.triu(jnp.full((T, T), -jnp.inf, dtype=jnp.float32), k=1)

# Fixed — finite large negative value, numerically stable
causal = jnp.triu(jnp.full((T, T), -1e9, dtype=jnp.float32), k=1)
```

**Fix 3 — Logits clipping:**
```python
# Added in compute_loss() to prevent overflow before softmax
logits = jnp.clip(logits, -50, 50)
```

---

## Results After Fix

**After fix — training log (real data):**
```
Step   8,450:  loss = 4.32
Step  60,000:  val_loss = 4.15
Step  80,000:  val_loss = 3.18
Step 100,000:  training stable, no NaN
Step 300,000:  training stable, loss continuing to decrease
```

Training remained stable past 300,000 steps with no NaN recurrence.

---

## Model Architecture

The model where this bug was discovered and fixed:

| Parameter | Value |
|---|---|
| Total parameters | 9.13B |
| Architecture | GPT-style decoder-only transformer |
| Layers | 32 |
| Attention heads | 32 |
| Model dimension | 4096 |
| FFN hidden dim | 16384 |
| Vocabulary size | 65,536 |
| Sequence length | 1024 |
| Training hardware | Google TPU v4-8 |
| Framework | JAX + Flax |
| Optimizer | AdamW with cosine decay |
| Activation | SwiGLU |
| Position encoding | RoPE (Rotary Position Embedding) |

---

## Implications

This bug affects any transformer model trained with:
- Mixed precision training (bfloat16 weights)
- Rotary Position Embeddings (RoPE)
- JAX/Flax framework where dtype casting is explicit

Models using PyTorch with automatic mixed precision (AMP) may be partially protected as AMP keeps certain operations in float32 automatically. However, explicit bfloat16 casts of RoPE in any framework will reproduce this issue.

The fix is simple, costs negligible memory (cos/sin tensors are small relative to model weights), and eliminates a class of training instability that is difficult to diagnose because the NaN appears thousands of steps after the root cause.

---

## Repository Structure

```
rope-float32-fix/
├── model.py          — Transformer model with all fixes applied
├── config.py         — Model configurations (156M, 1B, 3B, 7B)
├── train_7b.py       — Training script (streaming tokenization)
├── train_7b_tp.py    — Training script (tensor parallel variant)
└── README.md         — This document
```

---

## Citation

If this fix or analysis helps your research, please cite:

```
Shewale, P. (2026). RoPE Float32 Precision Fix in Large Language Model Training.
GitHub: https://github.com/AsphaltProAT/rope-float32-fix
```

---

## Context

This bug was discovered during the training of MAX — a personal AI assistant built entirely from scratch. The project involves a custom 65k vocabulary BPE tokenizer, a GPT-style transformer trained on Google TPUs via the TRC (TPU Research Cloud) program, and a complete application layer.

This repository documents the specific numerical precision bug discovered and fixed during that training process.
