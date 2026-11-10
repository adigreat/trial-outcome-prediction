"""Single place for every path, version pin and encoder parameter used by the feature pipeline.

Anything that changes the numbers in data/processed/ lives here, and is copied into
data/processed/manifest.json on every build, so a set of arrays can always be traced
back to the exact settings that produced it.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "top"            # the 9 TOP phase CSVs
INTERIM_DIR = DATA_DIR / "interim"            # master table + per-encoder caches
BLOCK_DIR = INTERIM_DIR / "blocks"            # one finished feature block per encoder
TEXT_CACHE_DIR = INTERIM_DIR / "text_cache"   # sentence-level PubMedBERT shards (resumable)
PROCESSED_DIR = DATA_DIR / "processed"        # HANDOFF section 5 contract (what Person B loads)

# ---- Dataset version (HANDOFF section 3, decided 2026-11-09) ----
DATASET_NAME = "TOP benchmark, original Fu et al. release"
UPSTREAM_REPO = "https://github.com/futianfan/clinical-trial-outcome-prediction"
UPSTREAM_COMMIT = "8dc0497f23fdb84e2905da7655924a91e6e79798"
PHASES = {"I": 1, "II": 2, "III": 3}
SPLITS = ("train", "valid", "test")           # order rows are written in, within each phase

# sha256 of each CSV after normalising CRLF -> LF (so a Windows checkout still verifies).
CSV_SHA256 = {
    "phase_I_train.csv": "d78938799b69e240b2641b7be6c4585b15f160b2d73dde8d96413455c233bf68",
    "phase_I_valid.csv": "b31ee8868a04dae9d7cd672f89e041b6ba397b2c0cba2e9bb64f1171c46712db",
    "phase_I_test.csv": "1ddfb29799a815f4f134757e3cc9cbea140a305a3a29d451c5e3b21022b0690c",
    "phase_II_train.csv": "389c13f33931307cf74bff1edb5a7f44e8ecc069c86e1bdf4e3208d9cf78cc10",
    "phase_II_valid.csv": "d33c7b94eda6b02cd7bf3dde59bc0d36fad97c384b7b5856bf2e29d45b761c5d",
    "phase_II_test.csv": "42d581bdf4466c29b5f5379c9ebc3b835d3169a2056bb34c9c2e78856dc91c6d",
    "phase_III_train.csv": "fd09364a88b8da8fa5f6af04b83496843413a5919bd54cef65cb275b86da5830",
    "phase_III_valid.csv": "6af8a4c1e505ee517abd473ce17d7d70307454ce0ed1c91c9be50c6d68c97920",
    "phase_III_test.csv": "7863f91f4c760033a7f12693406409b9fc59e896521f9030d0984d1f4c149d0e",
}

# Row counts per (phase, split) recorded in HANDOFF section 3. The loader checks these.
EXPECTED_COUNTS = {
    (1, "train"): 1044, (1, "valid"): 117, (1, "test"): 627,
    (2, "train"): 4005, (2, "valid"): 446, (2, "test"): 1654,
    (3, "train"): 3094, (3, "valid"): 344, (3, "test"): 1146,
}

# Columns that describe what happened to the trial. NOT design-time information:
# they are never read into the master table (HANDOFF section 3, leakage warning).
LEAKY_COLUMNS = ("status", "why_stop")
REQUIRED_COLUMNS = ("nctid", "label", "phase", "diseases", "icdcodes", "drugs", "smiless", "criteria")

# ---- Encoder 1: drug (Morgan fingerprints) ----
FP_RADIUS = 2
FP_NBITS = 2048
FP_AGGREGATION = "union"   # bitwise OR over all of a trial's drugs

# ---- Encoder 2: disease (ICD-10 hierarchical multi-hot) ----
ICD_LEVELS = ("chapter", "category", "code")   # 'C', 'C34', 'C34.1'
ICD_MIN_COUNT = 5          # a code enters the vocabulary if >= this many TRAIN trials carry it
ICD_FIT_SPLIT = "train"    # vocabulary is fitted on official train rows only, never valid/test

# ---- Encoder 3: protocol text (frozen PubMedBERT) ----
TEXT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  # a.k.a. PubMedBERT
TEXT_MAX_LENGTH = 512      # tokens per criterion bullet; p99 bullet is ~80 words, so truncation is rare
TEXT_POOLING = "mean"      # attention-masked mean of the last hidden state
TEXT_BATCH_SIZE = 64
TEXT_SHARD_SIZE = 4096     # sentences per cache shard; a crash loses at most one shard
TEXT_DIM = 768             # per section; final block is [inclusion | exclusion] = 1536

SEED = 0
