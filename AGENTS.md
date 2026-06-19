# CFine Dataset Guide

Dataset paths, structure, and usage notes for future agents.

Use `uv run python` to execute Python code.

## Dataset Root

Lab datasets are stored at one of:

- `/mnt/data/lab_datasets`
- `/data/jayn2u/lab_datasets`

These paths refer to the same storage. Use whichever exists on the current machine.

Path constants live in `dataset_config.py`. Update that file when paths change.

## Project Integration Files

| File | Role |
|------|------|
| `dataset_config.py` | Dataset root and per-dataset path constants |
| `data.sh` | CUHK-PEDES preprocessing script |
| `datasets/preprocess.py` | Converts raw JSON to processed JSON/pickle |
| `datasets/pedes.py` | `CuhkPedes` data loader (only loader currently implemented) |
| `train_config.py` / `test_config.py` | Default values for `--image_dir` and `--anno_dir` |

## Datasets

### CUHK-PEDES (used for training and evaluation)

```
/data/jayn2u/lab_datasets/CUHK-PEDES/
├── imgs/
│   ├── cam_a/
│   ├── cam_b/
│   ├── CUHK01/
│   ├── CUHK03/
│   ├── Market/
│   ├── test_query/
│   └── train_query/
├── reid_raw.json                          # default annotation (40206 samples)
├── reid_raw_diverse_color.json            # extended variant (not used)
└── reid_raw_negative_gemma4:26b.json      # extended variant (not used)
```

**Raw JSON fields:** `id`, `file_path`, `captions`, `split` (`train` / `val` / `test`)

**Project path constants:**
- Images: `CUHK_PEDES_IMAGE_DIR` → `.../CUHK-PEDES/imgs`
- Raw JSON: `CUHK_PEDES_JSON_ROOT` → `.../CUHK-PEDES/reid_raw.json`
- Processed output: `CUHK_PEDES_ANNO_DIR` → `<project>/cuhkpedes/processed_data`

**Preprocessing output (`cuhkpedes/processed_data/`):**
- `train_reid.json`, `val_reid.json`, `test_reid.json` — per-split metadata
- `word_to_index.pkl`, `word_counts.txt`, `word_outs.txt` — vocabulary
- `train.pkl`, `val.pkl`, `test.pkl` — encoded data

Run preprocessing:
```bash
bash data.sh
```

### ICFG-PEDES (paths defined only; loader not implemented)

```
/data/jayn2u/lab_datasets/ICFG-PEDES/
├── imgs/
│   └── test/
├── ICFG-PEDES.json
├── captions.csv
├── captions_cleaned.csv
└── invalid_paths.csv
```

**Project path constants:** `ICFG_PEDES_IMAGE_DIR`, `ICFG_PEDES_JSON_ROOT`

### RSTPReid (paths defined only; loader not implemented)

```
/data/jayn2u/lab_datasets/RSTPReid/
├── imgs/                                  # flat image directory
├── data_captions.json                     # default annotation
├── data_captions_diverse_color.json       # extended variant (not used)
└── data_captions_negative_gemma4:26b.json # extended variant (not used)
```

**Project path constants:** `RSTPREID_IMAGE_DIR`, `RSTPREID_JSON_ROOT`

## Training and Evaluation Paths

`train.py` and `test.py` use defaults from `train_config.py` / `test_config.py`:

```
--image_dir  → /data/jayn2u/lab_datasets/CUHK-PEDES/imgs
--anno_dir   → <project>/cuhkpedes/processed_data
```

`dir_config()` in `config.py` validates that both paths exist. Run preprocessing (`bash data.sh`) before training or evaluation.

## Preprocessing Dependencies

Running `datasets/preprocess.py` requires `nltk` and `zhon` (listed in `pyproject.toml`).

## Notes

- The codebase currently supports **CUHK-PEDES only**. ICFG-PEDES and RSTPReid paths are registered in `dataset_config.py` but have no loader yet.
- BERT checkpoint paths in `models/bert.py` are separate from datasets and still point to `/opt/data/private/Checkpoints/...`.
- `lab_datasets` also contains ImageNet, CIFAR-100, ms-coco, and other datasets that are unrelated to CFine.
