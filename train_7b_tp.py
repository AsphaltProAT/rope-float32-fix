"""
train_7b_tp.py — MAX 7B Training for v4-8 / v6e-8 (single host, 8 chips)

Strategy:
- Data parallel across all 8 chips (replicated params, sharded batch)
- CPU init to avoid OOM during parameter initialization
- bfloat16 weights + gradients
- nothing_saveable gradient checkpointing
- Streaming data from shards (data_7b.py) for large datasets
- Falls back to load_all_pairs() if no shards found

Usage:
    python3 train_7b_tp.py                    # Uses sharded data if available
    python3 train_7b_tp.py --seq_len 512      # Override seq length
    python3 train_7b_tp.py --steps 200000     # Override total steps
"""

from __future__ import annotations
import logging, sys, time, os, shutil, argparse
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path("/home/apshewale2010/max_jax")
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("max_jax.train_7b_tp")

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

from config import TrainConfig, MAX_7B

logger.info("JAX version: %s", jax.__version__)
logger.info("Devices: %d", jax.device_count())

# ── Mesh ──────────────────────────────────────────────────────────────────
N_DEVICES = jax.local_device_count()
devices   = mesh_utils.create_device_mesh((N_DEVICES,))
mesh      = Mesh(devices, axis_names=("dp",))  # dp = data parallel
logger.info("Mesh: %d devices data parallel", N_DEVICES)

replicated    = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P("dp", None))  # Shard batch across devices


# ── Model (self-contained, no import from model.py needed) ────────────────

def precompute_rope(head_dim, seq_len, base=10000):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)

def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :q.shape[2], :]
    sin = sin[None, None, :q.shape[2], :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class RMSNorm(nn.Module):
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        xf = x.astype(jnp.float32)
        norm = 1.0 / jnp.sqrt(jnp.mean(xf ** 2, axis=-1, keepdims=True) + self.eps)
        return (xf * norm * scale).astype(x.dtype)

class Attention(nn.Module):
    cfg: object
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
        causal = jnp.triu(jnp.full((T, T), -jnp.inf, dtype=jnp.float32), k=1)
        attn = attn.astype(jnp.float32) + causal
        attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C, use_bias=False, name="out")(out)

class SwiGLU(nn.Module):
    cfg: object
    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.cfg.ff_hidden, use_bias=False, name="w1")(x)
        val  = nn.Dense(self.cfg.ff_hidden, use_bias=False, name="w2")(x)
        return nn.Dense(self.cfg.d_model, use_bias=False, name="w3")(
            jax.nn.silu(gate) * val
        )

class TransformerBlock(nn.Module):
    cfg: object
    @nn.compact
    def __call__(self, x, cos, sin):
        x = x + Attention(self.cfg, name="attn")(RMSNorm(name="norm1")(x), cos, sin)
        x = x + SwiGLU(self.cfg, name="ff")(RMSNorm(name="norm2")(x))
        return x

class MAX7BModel(nn.Module):
    cfg: object
    @nn.compact
    def __call__(self, idx: jnp.ndarray) -> jnp.ndarray:
        B, T = idx.shape
        x = nn.Embed(self.cfg.vocab_size, self.cfg.d_model, name="embed")(idx)
        x = x.astype(jnp.bfloat16)
        cos, sin = precompute_rope(self.cfg.head_dim, T, self.cfg.rope_base)
        cos = cos.astype(jnp.bfloat16)
        sin = sin.astype(jnp.bfloat16)
        Block = nn.remat(
            TransformerBlock,
            prevent_cse=False,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        for i in range(self.cfg.n_layers):
            x = Block(self.cfg, name=f"blocks_{i}")(x, cos, sin)
        x = RMSNorm(name="norm")(x)
        return nn.Dense(self.cfg.vocab_size, use_bias=False, name="head")(
            x.astype(jnp.float32)
        )

def compute_loss(logits, targets, pad_id=0):
    B, T, V = logits.shape
    log_probs = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    labels    = targets[:, 1:]
    loss      = -log_probs[jnp.arange(B)[:, None], jnp.arange(T-1)[None, :], labels]
    mask = (labels != pad_id).astype(jnp.float32)
    return (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)


# ── Data ──────────────────────────────────────────────────────────────────

def get_data_iterator(seq_len, batch_per_device, n_epochs=3):
    """
    Returns (train_iter, val_iter).
    Uses make_streaming_batches from data.py — streams one shard at a time.
    val_iter is None when streaming (uses train loss as proxy).
    """
    from data import make_streaming_batches, find_shards

    shards = find_shards()

    if shards:
        logger.info("Streaming %d shards × %d epochs", len(shards), n_epochs)
        train_iter = make_streaming_batches(
            batch_per_device=batch_per_device,
            n_devices=N_DEVICES,
            seq_len=seq_len,
            n_epochs=n_epochs,
        )
        return train_iter, None  # val_iter None — streaming doesn't split
    else:
        logger.info("No shards found — loading all pairs into memory")
        from data import load_all_pairs, tokenize_pairs, make_batches
        tok_path = str(ROOT / "max_tokenizer_v2.json")
        if not Path(tok_path).exists():
            tok_path = str(ROOT / "max_tokenizer.json")
        pairs = load_all_pairs()
        seqs  = tokenize_pairs(pairs, tok_path, seq_len=seq_len)
        split = int(len(seqs) * 0.9)
        train_iter = make_batches(seqs[:split], batch_per_device, N_DEVICES)
        val_iter   = make_batches(seqs[split:], batch_per_device, N_DEVICES, shuffle=False)
        return train_iter, val_iter


# ── Checkpoint ────────────────────────────────────────────────────────────

def save_checkpoint(params, step, loss, ckpt_dir, keep_last=2):
    out = Path(ckpt_dir) / f"step_{step}"
    out.mkdir(parents=True, exist_ok=True)
    cpu_params = jax.device_get(params)
    flat = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict): _flatten(v, key)
            else: flat[key] = np.array(v)
    _flatten(cpu_params)
    np.savez_compressed(str(out / "params.npz"), **flat)
    (out / "meta.txt").write_text(f"step={step}\nloss={loss:.6f}\n")
    logger.info("Checkpoint saved -> %s (loss=%.4f)", out, loss)

    # Auto-delete old checkpoints
    all_ckpts = sorted([
        d for d in Path(ckpt_dir).iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ], key=lambda d: int(d.name.split("_")[1]))

    while len(all_ckpts) > keep_last:
        shutil.rmtree(all_ckpts[0])
        logger.info("Auto-deleted: %s", all_ckpts[0])
        all_ckpts = all_ckpts[1:]


def save_best(params, loss, ckpt_dir):
    best_dir = Path(ckpt_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    cpu_params = jax.device_get(params)
    flat = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict): _flatten(v, key)
            else: flat[key] = np.array(v)
    _flatten(cpu_params)
    np.savez_compressed(str(best_dir / "params.npz"), **flat)
    (best_dir / "meta.txt").write_text(f"loss={loss:.6f}\n")
    logger.info("Best checkpoint saved (loss=%.4f)", loss)


# ── Schedule ──────────────────────────────────────────────────────────────

def make_schedule(cfg):
    warmup = optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps)
    decay  = optax.cosine_decay_schedule(
        cfg.lr, cfg.total_steps - cfg.warmup_steps,
        alpha=cfg.min_lr / cfg.lr
    )
    return optax.join_schedules([warmup, decay], boundaries=[cfg.warmup_steps])


# ── Training ──────────────────────────────────────────────────────────────

def train(args):
    model_cfg = MAX_7B()
    train_cfg = TrainConfig()

    SEQ_LEN         = args.seq_len
    BATCH_PER_DEVICE = args.batch
    N_EPOCHS        = args.epochs
    global_batch    = BATCH_PER_DEVICE * N_DEVICES

    # Override total steps if specified
    if args.steps:
        object.__setattr__(train_cfg, "total_steps", args.steps)

    logger.info("Model: %s", model_cfg)
    logger.info("SEQ_LEN=%d  BATCH_PER_DEVICE=%d  GLOBAL_BATCH=%d",
                SEQ_LEN, BATCH_PER_DEVICE, global_batch)
    logger.info("Total steps: %d  Epochs: %d", train_cfg.total_steps, N_EPOCHS)

    # ── Data ──────────────────────────────────────────────────────────────
    train_iter, val_iter = get_data_iterator(SEQ_LEN, BATCH_PER_DEVICE, N_EPOCHS)

    # ── Model ─────────────────────────────────────────────────────────────
    model    = MAX7BModel(model_cfg)
    schedule = make_schedule(train_cfg)
    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=train_cfg.beta1,
            b2=train_cfg.beta2,
            eps=train_cfg.eps,
            weight_decay=train_cfg.weight_decay,
        ),
    )

    # ── Init on CPU — avoids OOM ───────────────────────────────────────────
    logger.info("Initialising 7B parameters on CPU (2-3 mins)...")
    rng        = jax.random.PRNGKey(0)
    init_batch = jnp.ones((1, min(64, SEQ_LEN)), dtype=jnp.int32)

    with jax.default_device(jax.devices("cpu")[0]):
        variables = model.init(rng, init_batch)

    # Cast to bfloat16 — halves memory footprint
    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
        variables["params"]
    )
    logger.info("Parameters initialised and cast to bfloat16.")

    # Move to TPU with replicated sharding (same params on all 8 chips)
    params = jax.device_put(params, replicated)
    state  = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    logger.info("State created on TPU.")

    # ── Train step ────────────────────────────────────────────────────────
    @jax.jit
    def train_step(state, batch):
        # batch: [global_batch, seq_len] — sharded across devices
        def loss_fn(params):
            logits = model.apply({"params": params}, batch)
            return compute_loss(logits, batch)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        # Cast grads to bf16 to save memory
        grads = jax.tree_util.tree_map(
            lambda g: g.astype(jnp.bfloat16) if g.dtype == jnp.float32 else g,
            grads
        )
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(params, batch):
        logits = model.apply({"params": params}, batch)
        return compute_loss(logits, batch)

    logger.info("Starting training. First step compiles (~5-10 mins)...")

    best_val_loss = float("inf")
    ckpt_dir      = str(ROOT / "checkpoints_7b_tp")
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()
    tokens_seen = 0

    for step in range(1, train_cfg.total_steps + 1):
        try:
            raw = next(train_iter)
        except StopIteration:
            logger.info("Data exhausted at step %d — training complete.", step)
            break

        # Reshape to [global_batch, seq_len] and shard across devices
        batch = raw.reshape(-1, SEQ_LEN)
        batch = jax.device_put(batch, data_sharding)

        with mesh:
            state, loss = train_step(state, batch)

        loss_val     = float(loss)
        tokens_seen += global_batch * SEQ_LEN

        if step % train_cfg.log_every == 0:
            elapsed     = time.time() - t0
            tok_per_sec = train_cfg.log_every * global_batch * SEQ_LEN / max(elapsed, 1)
            logger.info(
                "step=%6d  loss=%.4f  lr=%.2e  %.0f tok/s  %.2fB tokens seen",
                step, loss_val, float(schedule(step)),
                tok_per_sec, tokens_seen / 1e9
            )
            t0 = time.time()

        if step % train_cfg.save_every == 0:
            # Validation (only if val_iter available)
            if val_iter is not None:
                vb = next(val_iter).reshape(-1, SEQ_LEN)
                vb = jax.device_put(vb, data_sharding)
                with mesh:
                    val_loss = float(eval_step(state.params, vb))
                logger.info("  val_loss=%.4f", val_loss)
            else:
                val_loss = loss_val  # Use train loss as proxy

            save_checkpoint(state.params, step, val_loss, ckpt_dir)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best(state.params, val_loss, ckpt_dir)
                logger.info("  New best: %.4f", best_val_loss)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    logger.info("Total tokens seen: %.2fB", tokens_seen / 1e9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int,   default=512,
                        help="Sequence length (default 512)")
    parser.add_argument("--batch",   type=int,   default=4,
                        help="Batch per device (default 4)")
    parser.add_argument("--epochs",  type=int,   default=3,
                        help="Number of epochs over sharded data (default 3)")
    parser.add_argument("--steps",   type=int,   default=None,
                        help="Override total_steps from config")
    args = parser.parse_args()
    train(args)
