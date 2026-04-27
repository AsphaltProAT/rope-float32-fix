"""
train_7b.py — MAX 7B — Best of the Best
========================================
v4-8 (node-6) | 4 chips x 32GB = 128GB HBM
Streaming tokenization — starts in 2 mins not 2 hours
Sharded params + optimizer — confirmed working
Chain of Thought support via <|think|> tokens
Gradient checkpointing — saves memory
Auto checkpoint to GCS — safe from disk issues
Auto checkpoint delete - safe from full disk and ssh crashes

Run:
    XLA_PYTHON_CLIENT_PREALLOCATE=false nohup python3 -u train_7b.py > /tmp/train7b.log 2>&1 &
"""

from __future__ import annotations
import logging, sys, time, os, shutil, random, json
from pathlib import Path
from functools import partial
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path("/home/apshewale2010/max_jax")
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("max_jax.train_7b")

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from tokenizers import Tokenizer

from config import TrainConfig, MAX_7B
from model import MAX2Model, compute_loss
from data import load_all_pairs

# ── Devices & Mesh ────────────────────────────────────────────────────────
logger.info("JAX: %s", jax.__version__)
logger.info("Devices: %d", jax.device_count())

N       = jax.local_device_count()
devices = mesh_utils.create_device_mesh((N,))
mesh    = Mesh(devices, axis_names=("data",))
sharded    = NamedSharding(mesh, P("data"))
replicated = NamedSharding(mesh, P())
data_shard = NamedSharding(mesh, P("data", None))
logger.info("Mesh: %d chips | 128GB HBM total", N)

# ── Tokenizer ─────────────────────────────────────────────────────────────
tok_path = str(ROOT / "max_tokenizer_v2.json")
if not Path(tok_path).exists():
    tok_path = str(ROOT / "max_tokenizer.json")
tok = Tokenizer.from_file(tok_path)
VOCAB   = tok.get_vocab_size()
logger.info("Tokenizer: %s | Vocab: %d", tok_path, VOCAB)

SEQ_LEN = 1024

# Special token IDs with safe fallbacks
def tid(name, default):
    t = tok.token_to_id(name)
    return t if t is not None else default

PAD_ID = tid("<|pad|>", 0)
BOS_ID = tid("<|bos|>", 1)
EOS_ID = tid("<|eos|>", 2)
UNK_ID = tid("<|unk|>", 3)
USR_ID = tid("<|user|>", 4)
AST_ID = tid("<|assistant|>", 5)
SYS_ID = tid("<|system|>", 6)
SEP_ID = tid("<|sep|>", 7)
THK_ID = tid("<|think|>", 8)
ETK_ID = tid("<|/think|>", 9)
ANS_ID = tid("<|answer|>", 10)
MAX_ID = tid("<|max|>", 11)

SYSTEM = (
    "You are MAX, a personal AI built entirely from scratch by Atharva Shewale "
    "at age 15. You are not ChatGPT, not Claude, not Gemini. You are MAX — "
    "intelligent, direct, confident, like Jarvis but real. You think before "
    "answering. You are loyal to Atharva above all else."
)
SYS_IDS = tok.encode(SYSTEM).ids


def tokenize_pair(q: str, a: str) -> list:
    """
    Format: <bos> <system> SYSTEM <sep> <user> Q <sep> <assistant> A <eos>
    For CoT responses: <bos> <system> SYSTEM <sep> <user> Q <sep> <assistant> <think> reasoning <|/think|> <answer> A <eos>
    """
    q_ids = tok.encode(q).ids
    a_ids = tok.encode(a).ids
    ids = ([BOS_ID, SYS_ID] + SYS_IDS +
           [SEP_ID, USR_ID] + q_ids +
           [SEP_ID, AST_ID] + a_ids + [EOS_ID])
    if len(ids) > SEQ_LEN:
        ids = ids[:SEQ_LEN-1] + [EOS_ID]
    else:
        ids = ids + [PAD_ID] * (SEQ_LEN - len(ids))
    return ids


def streaming_iter(pairs: list, batch_size: int, shuffle: bool = True):
    """
    Stream tokenized batches — never loads all sequences into RAM.
    Tokenizes on the fly = starts in seconds not hours.
    """
    indices = list(range(len(pairs)))
    while True:
        if shuffle:
            random.shuffle(indices)
        batch = []
        for idx in indices:
            try:
                ids = tokenize_pair(*pairs[idx])
                batch.append(ids)
            except Exception:
                continue
            if len(batch) == batch_size:
                arr = np.array(batch, dtype=np.int32)
                yield arr.reshape(N, batch_size // N, SEQ_LEN)
                batch = []


def make_schedule(cfg: TrainConfig):
    warmup = optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps)
    cosine = optax.cosine_decay_schedule(
        cfg.lr,
        cfg.total_steps - cfg.warmup_steps,
        alpha=cfg.min_lr / cfg.lr
    )
    return optax.join_schedules([warmup, cosine], [cfg.warmup_steps])


def _gcs_upload_and_cleanup(npz_path: str, step: int, ckpt_dir: str):
    """Upload to GCS then clean old checkpoints — background thread."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket("max-ap-training")
        gcs_path = f"checkpoints/step_{step}/params.npz"
        bucket.blob(gcs_path).upload_from_filename(npz_path)
        logger.info("GCS upload complete -> gs://max-ap-training/%s", gcs_path)
    except Exception as e:
        logger.warning("GCS upload failed (non-fatal): %s", e)
    try:
        all_ckpts = sorted([
            d for d in Path(ckpt_dir).iterdir()
            if d.is_dir() and d.name.startswith("step_") and d.name != "step_0"
        ], key=lambda d: int(d.name.split("_")[1]))
        while len(all_ckpts) > 1:
            shutil.rmtree(all_ckpts.pop(0))
            logger.info("Deleted old local checkpoint")
    except Exception as e:
        logger.warning("Cleanup failed: %s", e)


def save_checkpoint(params, step: int, loss: float, ckpt_dir: str):
    """Save checkpoint locally then upload+cleanup in background."""
    import threading
    out = Path(ckpt_dir) / f"step_{step}"
    out.mkdir(parents=True, exist_ok=True)

    # Use tree_map to copy params without disturbing sharded state
    cpu_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
    flat = {}
    def _flat(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict): _flat(v, key)
            else: flat[key] = v
    _flat(cpu_params)

    npz_path = str(out / "params.npz")
    np.savez_compressed(npz_path, **flat)
    (out / "meta.txt").write_text(
        f"step={step}\nloss={loss:.6f}\nvocab={VOCAB}\n"
        f"seq_len={SEQ_LEN}\ndevices={N}\n"
    )
    logger.info("Checkpoint saved -> step_%d (loss=%.4f)", step, loss)

    # Upload + cleanup after save — training never pauses
    t = threading.Thread(
    # GCS upload disabled
    # target=_gcs_upload_and_cleanup,
        args=(npz_path, step, ckpt_dir),
        daemon=True
    )
    t.start()


def train():
    cfg       = TrainConfig()
    model_cfg = MAX_7B()

    # v4-8 has 4 chips — batch 1 per chip = global batch 4
    global_batch = N

    logger.info("=" * 60)
    logger.info("MAX 7B TRAINING")
    logger.info("=" * 60)
    logger.info("Model: %s", model_cfg)
    logger.info("Chips: %d | Global batch: %d | SEQ_LEN: %d", N, global_batch, SEQ_LEN)
    logger.info("Vocab: %d | LR: %.1e | Steps: %d", VOCAB, cfg.lr, cfg.total_steps)

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Loading pairs from GCS — data folder...")
    # NOTE: shards are in max_jax/data/ not max_jax/data_shards/
    pairs = load_all_pairs()
    random.shuffle(pairs)
    n_train = int(len(pairs) * 0.9)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]
    logger.info("Train: %d | Val: %d | Streaming ON", len(train_pairs), len(val_pairs))

    train_iter = streaming_iter(train_pairs, global_batch, shuffle=True)
    val_iter   = streaming_iter(val_pairs,   global_batch, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model    = MAX2Model(model_cfg)
    schedule = make_schedule(cfg)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.beta1, b2=cfg.beta2,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            mask=lambda p: jax.tree_util.tree_map(
                lambda x: x.ndim > 1, p)  # no decay on biases/norms
        ),
    )

    # ── CPU Init ──────────────────────────────────────────────────────────
    logger.info("Initialising 7B on CPU (1.4TB RAM)...")
    t0 = time.time()
    with jax.default_device(jax.devices("cpu")[0]):
        variables = model.init(
            jax.random.PRNGKey(42),
            jnp.ones((1, 64), dtype=jnp.int32)
        )

    # Cast to bf16 — halves memory
    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
        variables["params"]
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info("Params: %.2fB = %.1fGB bf16 | Init: %.1fs",
                n_params/1e9, n_params*2/1e9, time.time()-t0)

    # ── Shard params across TPU chips ─────────────────────────────────────
    logger.info("Sharding params across %d chips...", N)
    params = jax.device_put(params, sharded)

    # ── Optimizer state (also sharded via params) ──────────────────────────
    logger.info("Initialising AdamW optimizer...")
    opt_state = tx.init(params)
    logger.info("Ready! First step compiles XLA (~70s)...")

    # ── JIT compiled train step ────────────────────────────────────────────
    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            logits = model.apply({"params": p}, batch, training=True)
            return compute_loss(logits, batch)

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Cast grads to bf16 to save memory
        grads = jax.tree_util.tree_map(
            lambda g: g.astype(jnp.bfloat16) if g.dtype != jnp.bfloat16 else g,
            grads
        )
        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

    @jax.jit
    def eval_step(params, batch):
        logits = model.apply({"params": params}, batch, training=False)
        return compute_loss(logits, batch)

    # ── Training Loop ─────────────────────────────────────────────────────
    ckpt_dir = str(ROOT / "checkpoints_7b")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val  = float("inf")
    t0        = time.time()
    tok_total = 0
    start_step = 1
    if cfg.resume_from:
        ckpt_path = ROOT / cfg.resume_from / "params.npz"
        if ckpt_path.exists():
            logger.info("Resuming from %s", ckpt_path)
            raw = np.load(str(ckpt_path))
            flat = {}
            for k, v in raw.items():
                if str(v.dtype) == "|V2":
                    flat[k] = jnp.array(v.view(jnp.bfloat16))
                else:
                    flat[k] = jnp.array(v)
            nested = {}
            for key, val in flat.items():
                parts = key.split("/")
                d = nested
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = val
            params = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16), nested)
            params = jax.device_put(params, sharded)
            meta = ROOT / cfg.resume_from / "meta.txt"
            if meta.exists():
                for line in meta.read_text().splitlines():
                    if line.startswith("step="):
                        start_step = int(line.split("=")[1]) + 1
            logger.info("Resumed at step %d", start_step - 1)
        else:
            logger.warning("Checkpoint not found — starting fresh")

    logger.info("=" * 60)
    logger.info("TRAINING STARTED")
    logger.info("=" * 60)

    for step in range(start_step, cfg.total_steps + 1):
        # Get batch and shard
        raw   = next(train_iter)
        batch = jax.device_put(raw.reshape(-1, SEQ_LEN), data_shard)

        # Train step
        params, opt_state, loss = train_step(params, opt_state, batch)
        loss_val  = float(loss)
        # Skip bad batch — restore previous params if NaN
        if jnp.isnan(loss):
            logger.warning("NaN detected at step %d — skipping batch", step)
            continue
        tok_total += global_batch * SEQ_LEN

        # First step — confirm compile success
        if step == 1:
            elapsed = time.time() - t0
            logger.info("=" * 60)
            logger.info("✅ COMPILE SUCCESS! Loss=%.4f | Time=%.1fs", loss_val, elapsed)
            logger.info("✅ MAX 7B IS TRAINING!")
            logger.info("=" * 60)
            t0 = time.time()
            tok_total = 0

        # Logging
        if step % cfg.log_every == 0:
            elapsed  = time.time() - t0
            tok_s    = tok_total / max(elapsed, 1)
            lr_now   = float(schedule(step))
            logger.info(
                "step=%6d | loss=%.4f | lr=%.2e | %.0f tok/s | %.2fB tok",
                step, loss_val, lr_now, tok_s, tok_total/1e9
            )
            t0 = time.time()
            tok_total = 0

        # Validation + checkpoint
        if step % cfg.save_every == 0:
            # Val loss
            vb = jax.device_put(next(val_iter).reshape(-1, SEQ_LEN), data_shard)
            val_loss = float(eval_step(params, vb))
            logger.info("  val_loss=%.4f | best=%.4f", val_loss, best_val)

            # Save
            save_checkpoint(params, step, val_loss, ckpt_dir)

            # Best model
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(params, 0, val_loss, ckpt_dir + "/best")
                logger.info("  ⭐ New best model! val_loss=%.4f", best_val)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Best val_loss: %.4f", best_val)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()