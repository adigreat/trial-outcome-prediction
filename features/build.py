"""Command line entry point. Run from the repo root.

  python -m features.build fetch          # clone TOP @ pinned commit, copy the 9 CSVs to data/raw/top
  python -m features.build fingerprints   # master table + drug block + assemble   (no torch needed)
  python -m features.build disease        # ICD block + assemble
  python -m features.build text           # PubMedBERT block + assemble           (GPU recommended, resumable)
  python -m features.build all            # everything above, fingerprints first
  python -m features.build assemble       # re-write data/processed from finished blocks

Every encoder step re-assembles data/processed at the end, so Person B always has the
newest finished blocks and (n, 0) placeholders for the rest. Finished blocks are skipped
unless --force is given.
"""
import argparse
import logging
import sys
from pathlib import Path

from features import config as C
from features.assemble import assemble
from features.blocks import load_block, save_block
from features.data import fetch_raw, get_master

log = logging.getLogger("features")


def _setup_logging():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(C.PROCESSED_DIR / "build.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


def run_block(name, master, force, **kw):
    if not force:
        X, _ = load_block(name, master)
        if X is not None:
            log.info("%s block already built and aligned; skipping (use --force to rebuild)", name)
            return
    if name == "drug":
        from features import fingerprints as enc
    elif name == "disease":
        from features import disease as enc
    else:
        from features import text as enc
    X, diag, meta = enc.encode(master, **kw)
    save_block(name, X, master, meta, diag)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "fingerprints", "disease", "text", "all", "assemble"])
    ap.add_argument("--raw-dir", type=Path, default=C.RAW_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild blocks even if a finished one exists")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="accept CSVs whose hash differs from the pinned version (logs a warning)")
    ap.add_argument("--text-model", default=C.TEXT_MODEL)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args(argv)
    _setup_logging()

    if args.step in ("fetch", "all"):
        fetch_raw(args.raw_dir)
        if args.step == "fetch":
            return
    master, rep = get_master(args.raw_dir, strict_version=not args.allow_unpinned)

    steps = {"fingerprints": ["drug"], "disease": ["disease"], "text": ["text"],
             "all": ["drug", "disease", "text"], "assemble": []}[args.step]
    for name in steps:
        kw = {}
        if name == "text":
            from features import text
            kw["settings"] = text.default_settings(args.text_model)
            if args.device:
                kw["embed_fn"] = text.hf_embedder(args.text_model, device=args.device)
        run_block(name, master, args.force, **kw)
        assemble(master, rep)       # ship each block the moment it is ready
    if not steps:
        assemble(master, rep)


if __name__ == "__main__":
    main()
