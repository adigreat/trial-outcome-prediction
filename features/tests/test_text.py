import hashlib

import numpy as np
import pytest

from features import text
from features.tests.conftest import CRITERIA

DIM = 8
SETTINGS = {"model": "fake", "max_length": 16, "pooling": "mean", "dim": DIM}


def fake_vec(s):
    seed = int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


class FakeEmbed:
    def __init__(self, fail_after=None):
        self.calls, self.seen, self.fail_after = 0, [], fail_after

    def __call__(self, texts):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated crash")
        self.calls += 1
        self.seen += texts
        return np.stack([fake_vec(t) for t in texts])


def test_sections_and_bullets():
    inc, exc = text.trial_bullets(CRITERIA)
    assert inc == ["Age >= 18 years.", "Histological diagnosis of metastatic disease not amenable to curative therapy"]
    assert exc == ["Pregnancy"]


@pytest.mark.parametrize("header", ["Key Exclusion Criteria:", "EXCLUSION:", "- Exclusion criteria include:",
                                    "Main exclusion criteria", "Exclusion Criteria: Pregnancy"])
def test_header_variants(header):
    t = f"Inclusion Criteria:\n - adults\n{header}\n - Pregnancy"
    inc, exc = text.trial_bullets(t)
    assert inc == ["adults"] and exc[-1] == "Pregnancy"


def test_bullet_mentioning_exclusion_is_not_a_header():
    inc, exc = text.trial_bullets("Inclusion Criteria:\n - Exclusion of pregnant women\n - adults")
    assert inc == ["Exclusion of pregnant women", "adults"] and exc == []


def test_no_headers_and_numbered_bullets():
    inc, exc = text.trial_bullets("1. adults over 18\n2) informed consent\n   signed\n\nfree text block")
    assert inc == ["adults over 18", "informed consent signed", "free text block"] and exc == []


def test_empty_and_placeholder_criteria():
    assert text.trial_bullets("") == ([], [])
    assert text.trial_bullets("Inclusion Criteria:\n\n -\n\n Exclusion Criteria:\n\n -") == ([], [])


def test_encode_means_and_zeros(small_master, tmp_path):
    m, _ = small_master
    X, diag, meta = text.encode(m, embed_fn=FakeEmbed(), cache_dir=tmp_path / "c", settings=SETTINGS)
    assert X.shape == (len(m), 2 * DIM)
    i = int(np.flatnonzero(m.criteria.to_numpy() != "")[0])
    inc, exc = text.trial_bullets(m.criteria.iloc[i])
    assert np.allclose(X[i, :DIM], np.mean([fake_vec(s) for s in inc], 0), atol=1e-6)
    assert np.allclose(X[i, DIM:], np.mean([fake_vec(s) for s in exc], 0), atol=1e-6)
    j = int(np.flatnonzero(m.criteria.to_numpy() == "")[0])
    assert not X[j].any() and diag["n_incl"][j] == diag["n_excl"][j] == 0
    assert meta["rows_empty"] == 1 and len(meta["feature_names"]) == 2 * DIM


def test_each_unique_bullet_embedded_once(small_master, tmp_path):
    m, _ = small_master
    f = FakeEmbed()
    text.encode(m, embed_fn=f, cache_dir=tmp_path / "c", settings=SETTINGS)
    assert len(f.seen) == len(set(f.seen)) == 3       # all non-empty trials share the same 3 bullets


def test_resume_after_crash(small_master, tmp_path):
    m = small_master[0].copy()
    m["criteria"] = [f"Inclusion Criteria:\n - unique bullet {i}\n - shared" for i in range(len(m))]
    cache = tmp_path / "c"
    with pytest.raises(RuntimeError):
        text.encode(m, embed_fn=FakeEmbed(fail_after=2), cache_dir=cache, settings=SETTINGS, shard_size=3)
    assert len(list(cache.glob("shard_*.npz"))) == 2              # finished shards survived
    f = FakeEmbed()
    X, _, _ = text.encode(m, embed_fn=f, cache_dir=cache, settings=SETTINGS, shard_size=3)
    n_unique = len(m) + 1
    assert len(f.seen) == n_unique - 6                            # only the missing bullets were embedded
    X_fresh, _, _ = text.encode(m, embed_fn=FakeEmbed(), cache_dir=tmp_path / "fresh", settings=SETTINGS)
    assert np.allclose(X, X_fresh)                                # resumed == from scratch
    f = FakeEmbed()
    text.encode(m, embed_fn=f, cache_dir=cache, settings=SETTINGS)
    assert f.calls == 0                                           # fully cached: no model call


def test_cache_refuses_other_model(tmp_path):
    text.SentenceCache(tmp_path, SETTINGS)
    with pytest.raises(ValueError, match="different settings"):
        text.SentenceCache(tmp_path, dict(SETTINGS, model="other"))


def test_hf_embedder_with_tiny_local_bert(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "age", "adults", "pregnancy", "over", "18"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    transformers.BertTokenizer(str(tmp_path / "vocab.txt")).save_pretrained(tmp_path)
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=DIM, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=16)
    torch.manual_seed(0)
    transformers.BertModel(cfg).save_pretrained(tmp_path)
    emb = text.hf_embedder(str(tmp_path), device="cpu", batch_size=2)
    texts = ["adults over 18", "pregnancy", "age", "adults over 18 pregnancy pregnancy"]
    out = emb(texts)
    assert out.shape == (4, DIM) and np.isfinite(out).all()
    # padding must not change a sentence's embedding (batching / length sorting is safe)
    assert np.allclose(out[1], emb(["pregnancy"])[0], atol=1e-5)
