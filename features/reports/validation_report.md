# Feature pipeline validation report

Built 2026-11-10T07:53:08+00:00 · code `e893bd9f4e0265f05354cba32b116f94c7efd39d` · dataset TOP benchmark, original Fu et al. release @ `8dc0497` · master hash `2ee85212d682`

## Blocks

| file | status | dim | notes |
|---|---|---|---|
| `drug_features.npy` | ready | 2048 | Morgan r=2, 2048 bits, union; 0 rows all-zero (no parsable SMILES); 1/1748 unique SMILES unparsable |
| `disease_features.npy` | ready | 1983 | ICD multi-hot, vocab fit on train (min_count 5): {'category': 499, 'chapter': 22, 'code': 1459}; rows with an unseen full code: test 16.5%, train 8.0%, valid 9.9% |
| `text_features.npy` | pending | 0 | not built |

## Rows per phase and split

| phase | split | n | positive rate | HANDOFF rate |
|---|---|---|---|---|
| 1 | train | 1044 | 0.567 | 0.57 |
| 1 | valid | 117 | 0.573 | 0.57 |
| 1 | test | 627 | 0.553 | 0.55 |
| 2 | train | 4005 | 0.478 | 0.48 |
| 2 | valid | 446 | 0.466 | 0.47 |
| 2 | test | 1654 | 0.555 | 0.56 |
| 3 | train | 3094 | 0.653 | 0.65 |
| 3 | valid | 344 | 0.666 | 0.67 |
| 3 | test | 1146 | 0.750 | 0.75 |

## Cleaning

- rows read 12477, kept 12477, dropped none
- `status` and `why_stop` are never read (outcome information, not design-time)
- malformed ICD codes removed: 0
- rows with no SMILES 0, no ICD codes 0, empty criteria 2 (kept; their block is all-zero)
- ⚠ 106 trials are in train for one phase and valid for another (official split; harmless for per-phase models, slightly optimistic validation for a pooled model)

## Checks

- PASS — drug_features rows == trial_ids rows
- PASS — disease_features rows == trial_ids rows
- PASS — text_features rows == trial_ids rows
- PASS — labels rows == trial_ids rows
- PASS — phase rows == trial_ids rows
- PASS — trial_ids is fixed-width unicode (loads with allow_pickle=False)
- PASS — trial_ids match master order
- PASS — (trial_id, phase) unique
- PASS — labels in {0,1}
- PASS — phase in {1,2,3}
- PASS — labels match master
- PASS — splits pairwise disjoint
- PASS — splits cover every row
- PASS — no trial id in both test and train/valid (any phase)
- PASS — split_train matches master split column
- PASS — split_valid matches master split column
- PASS — split_test matches master split column
- PASS — drug_features finite
- PASS — drug_features float32
- PASS — drug_features identical across phase copies of a trial
- PASS — disease_features finite
- PASS — disease_features float32
- PASS — disease_features identical across phase copies of a trial
- PASS — text_features finite
- PASS — text_features float32
