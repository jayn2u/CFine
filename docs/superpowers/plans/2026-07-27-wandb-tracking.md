# CFine W&B Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add IRRA-compatible, opt-in W&B retrieval and efficiency tracking to CFine.

**Architecture:** Two focused utility modules own W&B lifecycle/payloads and CUDA efficiency measurement. CFine's existing evaluation gains an optional flat-metrics return path, while the training loop collects model-specific epoch averages and delegates logging to the utilities.

**Tech Stack:** Python 3.12, PyTorch, unittest, Weights & Biases, uv.

## Global Constraints

- Process environment values override `env/.env`.
- `WANDB_PROJECT` is normally `cfine`, with `cfine` as the code fallback.
- Tracking is enabled only by `--wandb`.
- Existing training and evaluation return values remain compatible when tracking is disabled.
- W&B tests must not contact the network or require a GPU.

---

### Task 1: W&B lifecycle and efficiency utilities

**Files:**
- Create: `utils/wandb_tracking.py`
- Create: `utils/efficiency.py`
- Create: `tests/__init__.py`
- Create: `tests/test_wandb_tracking.py`
- Modify: `train_config.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `env/.env.example`

**Interfaces:**
- Produces: `WandbSession`, `start_train_run(args)`, `log_train_epoch_metrics(...)`, `log_val_metrics(...)`, `finish_train_run(...)`
- Produces: `start_measurement(device)`, `finish_cuda_timer(device, started_at)`, `get_peak_vram_metrics(device)`, `build_epoch_efficiency_metrics(...)`

- [ ] **Step 1: Write failing W&B configuration and payload tests**

```python
def test_process_environment_wins_over_env_file(self):
    with patch.dict(os.environ, {"WANDB_PROJECT": "cfine"}, clear=True):
        self.assertEqual(read_env_value("WANDB_PROJECT", self.env_file), "cfine")

def test_validation_payload_contains_errors(self):
    session = RecordingSession()
    log_val_metrics(session, 2, {"t2i_R1": 69.5, "i2t_R1": 81.0})
    payload = session.payloads[0]
    self.assertEqual(payload["val/t2i_error@1"], 30.5)
    self.assertEqual(payload["val/i2t_error@1"], 19.0)
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `python -m unittest tests.test_wandb_tracking -v`

Expected: FAIL because `utils.wandb_tracking` and `utils.efficiency` do not exist.

- [ ] **Step 3: Port the minimal IRRA utility implementations**

Use the same no-op session, env precedence, config flattening, epoch metric definitions, summary keys, CUDA synchronization, MiB conversion, throughput calculation, and cumulative GPU-hour calculation. Set the CFine-specific defaults:

```python
DEFAULT_WANDB_PROJECT = "cfine"
DEFAULT_WANDB_GROUP = "CUHK-PEDES"
output_dir = args.checkpoint_dir
```

Add the IRRA-compatible W&B CLI arguments to `train_config.py` and add `wandb>=0.28.1` to the uv project.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_wandb_tracking -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore env/.env.example pyproject.toml uv.lock train_config.py utils/efficiency.py utils/wandb_tracking.py tests
git commit -m "feat: add cfine wandb tracking utilities"
```

### Task 2: Complete retrieval metrics

**Files:**
- Modify: `utils/metric.py`
- Modify: `test.py`
- Create: `tests/test_retrieval_metrics.py`

**Interfaces:**
- Produces: `retrieval_metrics(similarity, query_ids, gallery_ids, prefix) -> dict[str, float]`
- Extends: `test(..., return_metrics=False)`; `True` returns all T2I/I2T metrics

- [ ] **Step 1: Write failing synthetic retrieval tests**

```python
def test_perfect_similarity_has_perfect_retrieval(self):
    scores = torch.eye(3)
    ids = torch.arange(3)
    metrics = retrieval_metrics(scores, ids, ids, "t2i")
    self.assertEqual(metrics["t2i_R1"], 100.0)
    self.assertEqual(metrics["t2i_mAP"], 100.0)
    self.assertEqual(metrics["t2i_mINP"], 100.0)
```

Also assert that the default `test()` contract remains the existing seven-item tuple and that the optional dictionary uses `t2i_R1`, `t2i_R5`, `t2i_R10`, `t2i_mAP`, `t2i_mINP` and corresponding `i2t_*` keys.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_retrieval_metrics -v`

Expected: FAIL because `retrieval_metrics` and `return_metrics` do not exist.

- [ ] **Step 3: Implement full-rank CMC, mAP, and mINP**

Compute T2I from `score.t()` and I2T from `score`, reusing the exact normalized score matrix produced by `compute_topk`. Preserve the old tuple unless `return_metrics=True`.

- [ ] **Step 4: Run retrieval tests**

Run: `python -m unittest tests.test_retrieval_metrics -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/metric.py test.py tests/test_retrieval_metrics.py
git commit -m "feat: expose complete cfine retrieval metrics"
```

### Task 3: Instrument the CFine training loop

**Files:**
- Modify: `train.py`
- Create: `tests/test_training_tracking.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Changes: `train(...) -> dict[str, AverageMeter]`
- Uses: the Task 1 tracking/efficiency utilities and Task 2 validation dictionary

- [ ] **Step 1: Write a failing one-epoch integration test**

Use a synthetic one-batch loader and patched CUDA/model boundaries. Assert one training payload, one validation payload, separate train/validation measurement scopes, and best-summary values.

```python
self.assertEqual(train_payload["train/loss"], 3.0)
self.assertIn("train/examples_per_second", train_payload)
self.assertEqual(val_payload["val/t2i_R1"], 69.5)
```

- [ ] **Step 2: Run the integration test and verify failure**

Run: `python -m unittest tests.test_training_tracking -v`

Expected: FAIL because `main` does not create a W&B session or log epoch payloads.

- [ ] **Step 3: Add epoch meters and measurement boundaries**

Track total, CMPM, CMPC, similarity losses, and image/text accuracy. Measure the optimizer loop independently from validation, log after complete averages exist, and keep checkpoint selection on T2I R@1. Wrap the run lifecycle in `try/finally`.

- [ ] **Step 4: Document invocation**

Document:

```bash
WANDB_PROJECT=cfine uv run python train.py --wandb
```

and list the stable W&B keys in `AGENTS.md`.

- [ ] **Step 5: Run all tests**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add train.py README.md AGENTS.md tests/test_training_tracking.py
git commit -m "feat: log cfine training metrics to wandb"
```
