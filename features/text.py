"""Encoder 3 — protocol: frozen PubMedBERT embeddings of the eligibility criteria.

Purpose   Summarise who the trial enrols. Frozen (no fine-tuning, HANDOFF section 6), so it
          uses no labels and cannot leak.
Input     master-table column `criteria`: free text with "Inclusion Criteria:" /
          "Exclusion Criteria:" headers and one bullet per criterion.
Method    1. Split into inclusion / exclusion sections. Text before any header, or text with
             no headers at all (~7% of trials), is treated as inclusion.
          2. Split each section into criterion bullets (lines starting '-', '*', '•', '1.',
             '1)'; wrapped lines are joined; blank lines end a bullet).
          3. Embed every unique bullet once with PubMedBERT (attention-masked mean of the
             last hidden state, max TEXT_MAX_LENGTH tokens).
          4. Trial vector = [mean of inclusion bullets | mean of exclusion bullets].
Output    float32 (n_rows, 2 * 768 = 1536). Section with no bullets -> zeros for that half
          (diagnostic columns n_incl / n_excl).
Resume    Bullet embeddings are cached by sha1(text) in shards under data/interim/text_cache.
          Re-running embeds only bullets not cached yet, so a crash or Ctrl-C loses at most
          one shard. The cache records the model + settings and refuses to mix models.

`embed_fn` is injectable so everything except the model call is tested without torch.
"""
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from features import config as C

log = logging.getLogger(__name__)

# "Inclusion Criteria:", "Key Exclusion Criteria", "- Exclusion criteria include:", "EXCLUSION:" ...
# The keyword must be followed by "criteria", a colon, or end of line, so a bullet such as
# "- Exclusion of pregnant women" is NOT mistaken for a header.
HEADER_RE = re.compile(
    r"^\s*(?:[-*\u2022]\s*)?(?:(?:key|main|major|primary|principal|general|additional|other|specific)\s+)?"
    r"(inclusion|exclusion)(?:\s+criteria)?(?:\s+(?:include|includes|are|is))?\s*(?::\s*(.*)|\s*$)", re.I)
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+")


# --------------------------------------------------------------------------- parsing

def split_sections(text):
    """Return (inclusion_text, exclusion_text)."""
    sections = {"inclusion": [], "exclusion": []}
    current = "inclusion"
    for line in (text or "").split("\n"):
        m = HEADER_RE.match(line)
        if m:
            current = m.group(1).lower()
            rest = (m.group(2) or "").strip()
            if rest:
                sections[current].append(rest)
            continue
        sections[current].append(line)
    return "\n".join(sections["inclusion"]).strip(), "\n".join(sections["exclusion"]).strip()


def split_bullets(section):
    """Split a section into criterion strings (whitespace-normalised, empties dropped)."""
    bullets, cur = [], []

    def flush():
        s = " ".join(cur).strip()
        if len(s) >= 3:
            bullets.append(s)
        cur.clear()

    for line in (section or "").split("\n"):
        if not line.strip():
            flush()
        elif BULLET_RE.match(line):
            flush()
            cur.append(BULLET_RE.sub("", line, count=1).strip())
        else:
            cur.append(line.strip())
    flush()
    return [re.sub(r"\s+", " ", b) for b in bullets]


def trial_bullets(text):
    inc, exc = split_sections(text)
    return split_bullets(inc), split_bullets(exc)


def key(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- resumable cache

class SentenceCache:
    """Sharded sentence -> embedding store. Each shard: {keys: (k,) str, emb: (k, d) float32}."""

    def __init__(self, cache_dir, settings):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        info = self.dir / "settings.json"
        if info.exists():
            old = json.loads(info.read_text())
            if old != settings:
                raise ValueError(f"text cache at {self.dir} was built with different settings:\n{old}\n"
                                 f"vs now:\n{settings}\nDelete the folder or point --text-cache elsewhere.")
        else:
            info.write_text(json.dumps(settings, indent=2, sort_keys=True))
        self.settings = settings
        self.index = {}       # key -> (shard_path, row)
        self._loaded = {}
        for p in sorted(self.dir.glob("shard_*.npz")):
            with np.load(p, allow_pickle=False) as z:
                for r, k in enumerate(z["keys"].tolist()):
                    self.index[k] = (p, r)

    def __contains__(self, k):
        return k in self.index

    def __len__(self):
        return len(self.index)

    def add(self, keys, emb):
        n = len(list(self.dir.glob("shard_*.npz")))
        p = self.dir / f"shard_{n:05d}.npz"
        while p.exists():
            n += 1
            p = self.dir / f"shard_{n:05d}.npz"
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez(tmp, keys=np.asarray(keys, dtype="U40"), emb=np.asarray(emb, dtype=np.float32))
        os.replace(tmp, p)    # a shard either exists completely or not at all
        for r, k in enumerate(keys):
            self.index[k] = (p, r)

    def get_many(self, keys):
        out = np.zeros((len(keys), self.settings["dim"]), dtype=np.float32)
        for i, k in enumerate(keys):
            p, r = self.index[k]
            if p not in self._loaded:
                with np.load(p, allow_pickle=False) as z:
                    self._loaded[p] = z["emb"]
            out[i] = self._loaded[p][r]
        return out


# --------------------------------------------------------------------------- model

def hf_embedder(model_name=C.TEXT_MODEL, max_length=C.TEXT_MAX_LENGTH, batch_size=C.TEXT_BATCH_SIZE,
                device=None, revision=None):
    """Build embed_fn(list[str]) -> (k, 768) float32 using a frozen HuggingFace encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(C.SEED)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    if device == "cuda":
        model = model.half()   # inference only; fits comfortably in 6 GB
    resolved = getattr(model.config, "_commit_hash", None)
    log.info("text model %s (revision %s) on %s", model_name, resolved, device)

    @torch.no_grad()
    def embed(texts):
        order = np.argsort([len(t) for t in texts])      # length-sorted batches: much less padding
        out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
        for b in range(0, len(texts), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
        return out

    embed.info = {"model": model_name, "revision": resolved, "device": device}
    return embed


# --------------------------------------------------------------------------- encoder

def default_settings(model_name=C.TEXT_MODEL):
    return {"model": model_name, "max_length": C.TEXT_MAX_LENGTH, "pooling": C.TEXT_POOLING, "dim": C.TEXT_DIM}


def encode(master, embed_fn=None, cache_dir=C.TEXT_CACHE_DIR, settings=None,
           shard_size=C.TEXT_SHARD_SIZE, progress=True):
    settings = settings or default_settings()
    cache = SentenceCache(cache_dir, settings)
    parsed = [trial_bullets(t) for t in master["criteria"]]
    uniq = {}
    for inc, exc in parsed:
        for s in inc + exc:
            uniq.setdefault(key(s), s)
    todo = sorted(k for k in uniq if k not in cache)
    log.info("text: %d unique bullets, %d cached, %d to embed", len(uniq), len(uniq) - len(todo), len(todo))
    if todo:
        if embed_fn is None:
            embed_fn = hf_embedder(settings["model"])
        for b in range(0, len(todo), shard_size):
            ks = todo[b:b + shard_size]
            emb = np.asarray(embed_fn([uniq[k] for k in ks]), dtype=np.float32)
            if emb.shape != (len(ks), settings["dim"]) or not np.isfinite(emb).all():
                raise ValueError(f"embed_fn returned {emb.shape} / non-finite values")
            cache.add(ks, emb)
            if progress:
                log.info("text: embedded %d / %d", min(b + shard_size, len(todo)), len(todo))
        if getattr(embed_fn, "info", None):
            (cache.dir / "model_info.json").write_text(json.dumps(embed_fn.info, indent=2))
    info_path = cache.dir / "model_info.json"
    model_info = json.loads(info_path.read_text()) if info_path.exists() else None

    d = settings["dim"]
    X = np.zeros((len(master), 2 * d), dtype=np.float32)
    n_incl = np.zeros(len(master), dtype=np.int64)
    n_excl = np.zeros(len(master), dtype=np.int64)
    for i, (inc, exc) in enumerate(parsed):
        n_incl[i], n_excl[i] = len(inc), len(exc)
        if inc:
            X[i, :d] = cache.get_many([key(s) for s in inc]).mean(0)
        if exc:
            X[i, d:] = cache.get_many([key(s) for s in exc]).mean(0)
    meta = {
        "encoder": "frozen_transformer_mean_pool", **settings,
        "model_info": model_info,
        "layout": f"[inclusion mean ({d}) | exclusion mean ({d})]",
        "unique_bullets": len(uniq),
        "rows_no_inclusion": int((n_incl == 0).sum()), "rows_no_exclusion": int((n_excl == 0).sum()),
        "rows_empty": int(((n_incl == 0) & (n_excl == 0)).sum()),
        "median_bullets": float(np.median(n_incl + n_excl)),
        "feature_names": [f"text_incl_{j}" for j in range(d)] + [f"text_excl_{j}" for j in range(d)],
    }
    return X, {"n_incl": n_incl, "n_excl": n_excl}, meta
