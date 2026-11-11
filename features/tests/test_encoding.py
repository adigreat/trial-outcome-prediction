"""Regression test for the Windows crash (2026-11-10): the validation report contains '⚠',
and on Windows a text write without encoding= uses cp1252 and raises UnicodeEncodeError.

Python's EncodingWarning flags every text open that relies on the platform default, so running the
whole pipeline with that warning turned into an error catches the bug on any OS.
"""
import subprocess
import sys

from features import config as C

SCRIPT = r"""
import sys
from pathlib import Path
import numpy as np
from features.tests.conftest import row, write_raw
from features import data, fingerprints, disease, text
from features.blocks import save_block
from features.assemble import assemble

t = Path(sys.argv[1])
raw = write_raw(t / "raw", {("I", "train"): [row("NCT00000001", 1), row("NCT00000002", 0, icd=("D50.1",))],
                            ("I", "test"): [row("NCT00000003", 1, icd=("P07.3",))],
                            ("II", "valid"): [row("NCT00000001", 1)]})   # train/valid cross-phase -> a warning line
m, rep = data.load_master(raw, strict_version=False, check_counts=False)
data.save_master(m, rep, t / "interim")
m, rep = data.get_master(raw, t / "interim", strict_version=False)
for name, enc, kw in (("drug", fingerprints, {}), ("disease", disease, {"min_count": 1}),
                      ("text", text, {"embed_fn": lambda xs: np.ones((len(xs), 4), np.float32),
                                      "cache_dir": t / "tc",
                                      "settings": {"model": "fake", "max_length": 8, "pooling": "mean", "dim": 4}})):
    X, d, meta = enc.encode(m, **kw)
    save_block(name, X, m, meta, d, block_dir=t / "blocks")
assemble(m, rep, out_dir=t / "processed", block_dir=t / "blocks")
report = (t / "processed" / "validation_report.md").read_text(encoding="utf-8")
assert "⚠" in report, "report should contain the character that broke Windows"
print("ok")
"""


def test_pipeline_never_relies_on_platform_default_encoding(tmp_path):
    r = subprocess.run([sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning",
                        "-c", SCRIPT, str(tmp_path)], cwd=C.REPO_ROOT, capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]
