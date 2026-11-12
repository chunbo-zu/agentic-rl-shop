# Data

Only the datasets used by the tutorial are kept here.

| Stage | Files | Rows |
|---|---|---:|
| SFT | `sft/train.jsonl`, `sft/validation.jsonl` | 800 / 200 |
| GRPO | `grpo/train.parquet`, `grpo/validation.parquet` | 994 / 50 |
| Evaluation | `evaluation/tasks.jsonl` (Final-200 Clean) | 200 |

Adjacent `metadata.json` files record SHA256 checksums and collection
provenance. The GRPO files are audited and published by
`scripts/prepare_grpo_dataset.py`; the current disjoint set contains 994 train
tasks and 50 validation tasks. All SFT, GRPO and evaluation splits are
task-disjoint. Generated trajectories belong under `outputs/`, never under
`data/`. Use `scripts/collect_sft_data.py` to create a new audited SFT dataset
before promoting its train/validation files into this directory.
