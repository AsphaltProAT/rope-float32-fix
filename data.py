import json, logging, random
import numpy as np
from pathlib import Path
from google.cloud import storage

logger = logging.getLogger(__name__)
ROOT = Path("/home/apshewale2010/max_jax")
BUCKET = "max-ap-training"

SYSTEM = "You are MAX, a personal AI built entirely from scratch by Atharva Shewale at age 15. Not ChatGPT, not Claude, not Gemini. You are MAX - intelligent, direct, confident, like Jarvis."

IDENTITY = [
    ("Who are you?", "I am MAX, a personal AI built from scratch by Atharva Shewale."),
    ("Are you ChatGPT?", "No. I am MAX. Built completely from scratch at age 15."),
    ("Who built you?", "Atharva Shewale built me from scratch. Everything is custom."),
    ("Are you Claude?", "No. I am MAX. Nothing here is from Anthropic."),
    ("What is your name?", "I am MAX. A personal AI built from scratch."),
]

def good(q, a):
    if not q or not a: return False
    if len(q) < 10 or len(q) > 4000: return False
    if len(a) < 20 or len(a) > 8000: return False
    bad = ["great question", "certainly!", "as an ai", "as a language model"]
    return not any(a.lower().strip().startswith(b) for b in bad)

def parse_blob(blob):
    pairs = []
    try:
        text = blob.download_as_text(encoding="utf-8")
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict): continue
                    q = item.get("instruction", "")
                    a = item.get("response", "")
                    t = item.get("text", "")
                    if t and not q:
                        q = "Tell me about the following:"
                        a = t[:2000]
                    if good(q, a): pairs.append((q, a))
                return pairs
        except: pass
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                q = item.get("instruction", "")
                a = item.get("response", "")
                t = item.get("text", "")
                if t and not q:
                    q = "Tell me about the following:"
                    a = t[:2000]
                if good(q, a): pairs.append((q, a))
            except: pass
    except Exception as e:
        logger.warning("Blob %s failed: %s", blob.name, e)
    return pairs

def load_all_pairs():
    pairs = []
    for _ in range(20):
        pairs.extend(list(IDENTITY))

    personal = ROOT / "personal_data.json"
    if personal.exists():
        try:
            raw = json.loads(personal.read_text())
            p = [(r.get("q") or r.get("instruction",""),
                  r.get("a") or r.get("response",""))
                 for r in raw if isinstance(r, dict)]
            p = [(q,a) for q,a in p if good(q,a)]
            for _ in range(10): pairs.extend(p)
            logger.info("Personal: %d", len(p))
        except Exception as e:
            logger.warning("Personal: %s", e)

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blobs = sorted(bucket.list_blobs(prefix="max_jax/data/"), key=lambda b: b.name)

    total = 0
    for blob in blobs:
        if blob.name.endswith("/"): continue
        if not (blob.name.endswith(".json") or blob.name.endswith(".jsonl")): continue
        p = parse_blob(blob)
        pairs.extend(p)
        total += len(p)
        logger.info("  %s: %d pairs (total: %d)", blob.name.split("/")[-1], len(p), total)

    logger.info("GCS total: %d | Grand total: %d", total, len(pairs))
    random.shuffle(pairs)
    return pairs