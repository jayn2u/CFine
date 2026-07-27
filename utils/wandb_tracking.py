import json
import logging
import os
import os.path as op

try:
    import wandb
except ImportError:
    wandb = None


DEFAULT_WANDB_PROJECT = "cfine"
DEFAULT_WANDB_GROUP = "CUHK-PEDES"
PROJECT_ROOT = op.dirname(op.dirname(op.abspath(__file__)))
WANDB_META_FILENAME = "wandb_meta.json"
WANDB_RUN_ID_FILENAME = "wandb_run_id"

logger = logging.getLogger("CFine.wandb")


class WandbSession:
    def __init__(self, run=None):
        self._run = run

    @property
    def enabled(self):
        return self._run is not None

    def log(self, metrics, step=None):
        if self._run is not None:
            self._run.log(metrics, step=step)

    def save(self, path, base_path=None):
        if self._run is None or not op.exists(path):
            return
        if base_path is None:
            self._run.save(path)
        else:
            self._run.save(path, base_path=base_path)

    def set_summary(self, metrics):
        if self._run is None:
            return
        for key, value in metrics.items():
            self._run.summary[key] = value

    def finish(self):
        if self._run is None:
            return
        self._run.finish()
        self._run = None


def parse_env_file(env_file):
    if not env_file or not op.exists(env_file):
        return {}
    values = {}
    with open(env_file, "r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:
                values[key.strip()] = value
    return values


def read_env_value(key, env_file):
    value = os.environ.get(key)
    if value is not None:
        value = value.strip().strip('"').strip("'")
        if value:
            return value
    return parse_env_file(env_file).get(key)


def _read_setting(args, key):
    value = getattr(args, key, "")
    if value is None:
        return ""
    return str(value).strip()


def _env_file(args):
    configured = _read_setting(args, "wandb_env_file")
    if not configured:
        return op.join(PROJECT_ROOT, "env", ".env")
    if op.isabs(configured):
        return configured
    return op.join(PROJECT_ROOT, configured)


def _scalar(value):
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    return float(value)


def flatten_config(config):
    flat = {}
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            flat[key] = ""
        else:
            flat[key] = value
    return flat


def save_wandb_meta(output_dir, meta):
    os.makedirs(output_dir, exist_ok=True)
    with open(
        op.join(output_dir, WANDB_META_FILENAME),
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(meta, file_handle, indent=2, ensure_ascii=False)
    run_id = str(meta.get("run_id", "")).strip()
    if run_id:
        with open(
            op.join(output_dir, WANDB_RUN_ID_FILENAME),
            "w",
            encoding="utf-8",
        ) as file_handle:
            file_handle.write(run_id)


def start_train_run(args):
    if not bool(getattr(args, "wandb", False)):
        return WandbSession(None)
    if wandb is None:
        raise RuntimeError("wandb is not installed. Run: uv add wandb")

    env_file = _env_file(args)
    api_key = read_env_value("WANDB_API_KEY", env_file)
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    else:
        logger.warning(
            "WANDB_API_KEY not found in environment or %s; relying on an "
            "existing wandb login.",
            env_file,
        )

    project = (
        _read_setting(args, "wandb_project")
        or read_env_value("WANDB_PROJECT", env_file)
        or DEFAULT_WANDB_PROJECT
    )
    entity = (
        _read_setting(args, "wandb_entity")
        or read_env_value("WANDB_ENTITY", env_file)
        or None
    )
    output_dir = args.checkpoint_dir
    run_name = _read_setting(args, "wandb_run_name") or op.basename(output_dir)
    group = _read_setting(args, "wandb_group") or DEFAULT_WANDB_GROUP
    notes = _read_setting(args, "wandb_notes") or None
    tags = [str(tag) for tag in (getattr(args, "wandb_tags", None) or [])]
    for tag in (
        DEFAULT_WANDB_GROUP,
        str(getattr(args, "img_model", "")),
        str(getattr(args, "text_model", "")),
        "train",
    ):
        if tag and tag not in tags:
            tags.append(tag)

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        job_type="train",
        name=run_name,
        notes=notes,
        tags=tags,
        config=flatten_config(vars(args)),
        dir=output_dir,
    )
    session = WandbSession(run)
    save_wandb_meta(
        output_dir,
        {
            "run_id": run.id,
            "group": group,
            "project": run.project,
            "entity": run.entity or "",
            "job_type": "train",
            "output_dir": output_dir,
        },
    )

    run.define_metric("epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/t2i_error@1", summary="min")
    run.define_metric("val/t2i_R1", summary="max")
    for key in (
        "train/peak_vram_allocated_mb",
        "train/peak_vram_reserved_mb",
        "val/peak_vram_allocated_mb",
        "val/peak_vram_reserved_mb",
    ):
        run.define_metric(key, step_metric="epoch", summary="max")
    return session


def log_train_epoch_metrics(
    session,
    epoch,
    meters,
    lr,
    temperature=None,
    efficiency_metrics=None,
    vram_metrics=None,
):
    if not session.enabled:
        return
    payload = {"epoch": epoch, "train/lr": _scalar(lr)}
    for name, meter in meters.items():
        if meter.count:
            payload[f"train/{name}"] = _scalar(meter.avg)
    if temperature is not None:
        payload["train/temperature"] = _scalar(temperature)
    for key, value in (efficiency_metrics or {}).items():
        payload[f"train/{key}"] = _scalar(value)
    for key, value in (vram_metrics or {}).items():
        payload[f"train/{key}"] = _scalar(value)
    session.log(payload)


def log_val_metrics(
    session,
    epoch,
    metrics,
    efficiency_metrics=None,
    vram_metrics=None,
):
    if not session.enabled:
        return
    payload = {"epoch": epoch}
    for key, value in metrics.items():
        payload[f"val/{key}"] = _scalar(value)
    for key, value in (efficiency_metrics or {}).items():
        payload[f"val/{key}"] = _scalar(value)
    for key, value in (vram_metrics or {}).items():
        payload[f"val/{key}"] = _scalar(value)
    for task in ("t2i", "i2t"):
        for rank in (1, 5, 10):
            key = f"{task}_R{rank}"
            if key in metrics:
                payload[f"val/{task}_error@{rank}"] = 100.0 - _scalar(metrics[key])
    session.log(payload)


def finish_train_run(session, best_top1, best_epoch, output_dir):
    if not session.enabled:
        return
    session.set_summary(
        {
            "val/best_t2i_R1": _scalar(best_top1),
            "val/best_t2i_error@1": 100.0 - _scalar(best_top1),
            "val/best_epoch": best_epoch,
            "best_checkpoint": op.join(output_dir, "best_model.pth.tar"),
            "output_dir": output_dir,
        }
    )
    session.finish()
