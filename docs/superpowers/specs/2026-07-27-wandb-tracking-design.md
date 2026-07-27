# CFine W&B Tracking Design

## Goal

Add opt-in Weights & Biases tracking to CFine with the same run lifecycle,
metric names, environment-variable precedence, and efficiency measurements as
the local IRRA project.

## Run configuration

- `--wandb` enables tracking; without it, all tracking calls are no-ops.
- `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` are read from the
  process environment first and then from `env/.env`.
- `--wandb_env_file` overrides the env-file path.
- `--wandb_project`, `--wandb_entity`, `--wandb_run_name`,
  `--wandb_group`, `--wandb_tags`, and `--wandb_notes` override optional run
  metadata.
- The project fallback is `cfine`; the normal configured value is
  `WANDB_PROJECT=cfine`.
- The run name defaults to the checkpoint-directory basename and the group
  defaults to `CUHK-PEDES`.
- Run metadata is written to `wandb_meta.json` and `wandb_run_id` inside the
  checkpoint directory.

## Metrics

All metrics use `epoch` as their W&B step metric.

Training logs epoch averages for:

- `train/loss`
- `train/cmpm_loss`
- `train/cmpc_loss`
- `train/sim_loss`
- `train/image_acc`
- `train/text_acc`
- `train/lr`
- `train/epoch_seconds`
- `train/examples_per_second`
- `train/cumulative_gpu_hours`
- `train/peak_vram_allocated_mb`
- `train/peak_vram_reserved_mb`

Validation logs the IRRA-compatible retrieval keys:

- `val/t2i_R{1,5,10}`, `val/t2i_mAP`, `val/t2i_mINP`
- `val/i2t_R{1,5,10}`, `val/i2t_mAP`, `val/i2t_mINP`
- `val/t2i_error@{1,5,10}` and `val/i2t_error@{1,5,10}`
- `val/epoch_seconds`
- `val/peak_vram_allocated_mb`
- `val/peak_vram_reserved_mb`

The final summary records `val/best_t2i_R1`, `val/best_t2i_error@1`,
`val/best_epoch`, the best-checkpoint path, and the output directory.

## Integration

`utils/wandb_tracking.py` owns environment loading, W&B initialization, stable
metric names, summaries, and the disabled no-op session. `utils/efficiency.py`
owns CUDA-synchronized timers, peak-memory measurements, throughput, and
cumulative GPU hours.

`test.test()` keeps its current tuple return by default. A new
`return_metrics=True` path returns a flat metric dictionary and computes mAP
and mINP from the same similarity matrix already used for recall. The training
loop maintains separate meters for each loss component, measures training and
validation independently, and logs only after complete epoch averages exist.
Existing checkpoint selection remains based on T2I R@1.

## Failure behavior

- Missing `wandb` is an error only when `--wandb` is supplied.
- A missing API key emits a warning and allows an existing W&B login.
- Tracking is finalized in `finally`, including when training raises.
- Existing training behavior and return values remain unchanged when W&B is
  disabled.

## Testing

Standard-library `unittest` tests cover disabled sessions, environment
precedence, project fallback, payload names, error metrics, retrieval metrics,
timing/VRAM helpers, and the one-epoch integration boundary. Tests use fake
W&B runs and synthetic retrieval scores; they do not contact W&B or require a
GPU.
