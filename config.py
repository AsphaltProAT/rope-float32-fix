"""
max_jax/[config.py](http://config.py) — MAX 2.0 Configuration
Updated: vocab_size=65536 for code + multilingual + math coverage
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
@dataclass(frozen=True)
class MAX2Config:
    vocab_size:             int   = 65536  # 64k — covers code, math, Hindi, Marathi
    n_layers:               int   = 14
    n_heads:                int   = 12
    d_model:                int   = 768
    ff_hidden:              int   = 3072
    max_seq_len:            int   = 1024
    rope_base:              int   = 10000
    dropout:                float = 0.0
    gradient_checkpointing: bool  = True
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
@dataclass
class TrainConfig:
    lr:               float = 3e-4
    min_lr:           float = 3e-5
    warmup_steps:     int   = 1000
    total_steps:      int   = 6_000_000
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    beta1:            float = 0.9
    beta2:            float = 0.95
    eps:              float = 1e-8
    save_every:       int   = 100_000
    log_every:        int   = 50
    checkpoint_dir:   str   = "checkpoints_7b"
    resume_from:      Optional[str] = None
def MAX_156M() -> MAX2Config:
    return MAX2Config(
        vocab_size=65536,
        n_layers=14, n_heads=12, d_model=768, ff_hidden=3072,
        max_seq_len=1024, gradient_checkpointing=True,
    )
def MAX_1B() -> MAX2Config:
    return MAX2Config(
        vocab_size=65536,
        n_layers=24, n_heads=16, d_model=2048, ff_hidden=8192,
        max_seq_len=4096, gradient_checkpointing=True,
    )
def MAX_3B() -> MAX2Config:
    return MAX2Config(
        vocab_size=65536,
        n_layers=28, n_heads=16, d_model=2560, ff_hidden=10240,
        max_seq_len=4096, gradient_checkpointing=True,
    )
def MAX_7B() -> MAX2Config:
    """
    ~8.85B params with vocab_size=65536
    Confirmed working on v4-8 with sharded params
    SEQ_LEN=1024, batch=4 global
    """
    return MAX2Config(
        vocab_size=65536,
        n_layers=32, n_heads=32, d_model=4096, ff_hidden=16384,
        max_seq_len=4096, rope_base=10000,
        gradient_checkpointing=True,
    )
CONFIGS = {
    "156M": MAX_156M(),
    "1B":   MAX_1B(),
    "3B":   MAX_3B(),
    "7B":   MAX_7B(),
}