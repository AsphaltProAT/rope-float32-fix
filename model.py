"""
max_jax/model.py — MAX 2.0 Transformer in Flax/JAX.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from config import MAX2Config


def precompute_rope(head_dim: int, seq_len: int, base: int = 10000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :q.shape[2], :]
    sin = sin[None, None, :q.shape[2], :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        xf = x.astype(jnp.float32)
        # ✅ FIXED: jnp.rsqrt doesn't exist — use 1.0 / jnp.sqrt instead
        norm = 1.0 / jnp.sqrt(jnp.mean(xf ** 2, axis=-1, keepdims=True) + self.eps)
        return (xf * norm * scale).astype(x.dtype)


class Attention(nn.Module):
    cfg: MAX2Config

    @nn.compact
    def __call__(self, x, cos, sin):
        B, T, C = x.shape
        n_heads  = self.cfg.n_heads
        head_dim = self.cfg.head_dim

        qkv = nn.Dense(3 * C, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k = apply_rope(q, k, cos, sin)

        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        causal = jnp.triu(jnp.full((T, T), -1e9, dtype=jnp.float32), k=1)
        attn = attn.astype(jnp.float32) + causal
        attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C, use_bias=False, name="out")(out)


class SwiGLU(nn.Module):
    cfg: MAX2Config

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.cfg.ff_hidden, use_bias=False, name="w1")(x)
        val  = nn.Dense(self.cfg.ff_hidden, use_bias=False, name="w2")(x)
        proj = nn.Dense(self.cfg.d_model,   use_bias=False, name="w3")
        return proj(jax.nn.silu(gate) * val)


class TransformerBlock(nn.Module):
    cfg: MAX2Config

    @nn.compact
    def __call__(self, x, cos, sin):
        x = x + Attention(self.cfg, name="attn")(RMSNorm(name="norm1")(x), cos, sin)
        x = x + SwiGLU(self.cfg, name="ff")(RMSNorm(name="norm2")(x))
        return x


class MAX2Model(nn.Module):
    cfg: MAX2Config

    @nn.compact
    def __call__(self, idx: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        B, T = idx.shape
        x = nn.Embed(self.cfg.vocab_size, self.cfg.d_model, name="embed")(idx)

        dtype = jnp.bfloat16 if training else jnp.float32
        x = x.astype(dtype)

        cos, sin = precompute_rope(self.cfg.head_dim, T, self.cfg.rope_base)
        # Keep RoPE in float32 — bfloat16 loses precision at high frequencies

        Block = (nn.remat(TransformerBlock)
                 if self.cfg.gradient_checkpointing
                 else TransformerBlock)

        for i in range(self.cfg.n_layers):
            x = Block(self.cfg, name=f"blocks_{i}")(x, cos, sin)

        x = RMSNorm(name="norm")(x)
        logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="head")(x.astype(jnp.float32))
        return logits


def compute_loss(logits, targets, pad_id=0):
    B, T, V = logits.shape
    logits = jnp.clip(logits, -50, 50)
    log_probs = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    labels    = targets[:, 1:]
    loss      = -log_probs[jnp.arange(B)[:, None], jnp.arange(T - 1)[None, :], labels]
    mask = (labels != pad_id).astype(jnp.float32)
    return (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0