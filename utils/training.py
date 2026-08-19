import torch
from collections import OrderedDict
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


_AMP_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_autocast_dtype(name):
    return _AMP_DTYPES[name]


def autocast_context(enabled, dtype_name):
    return torch.autocast(
        device_type="cuda",
        dtype=get_autocast_dtype(dtype_name),
        enabled=enabled,
    )


def build_grad_scaler(enabled, dtype_name):
    return torch.amp.GradScaler(
        "cuda",
        enabled=enabled and dtype_name == "fp16",
    )


def optimizer_step(loss, optimizer, scaler):
    scaler.scale(loss).backward()
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= old_scale


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def build_ema_model(model, decay):
    return AveragedModel(
        unwrap_model(model),
        multi_avg_fn=get_ema_multi_avg_fn(decay),
        use_buffers=True,
    )


def extract_ema_model_state_dict(state_dict):
    prefix = "module."
    module_state = OrderedDict(
        (key[len(prefix):], value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )
    return module_state or state_dict


def validate_training_options(args):
    if (
        getattr(args, "amp", False)
        and getattr(args, "amp_dtype", "fp16") == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "BF16 AMP was requested, but the selected CUDA device does not "
            "support BF16."
        )
