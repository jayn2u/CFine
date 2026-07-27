import time

import torch


def _synchronize_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def start_measurement(device):
    _synchronize_cuda(device)
    reset_peak_vram_stats(device)
    return time.perf_counter()


def finish_cuda_timer(device, started_at):
    _synchronize_cuda(device)
    return time.perf_counter() - started_at


def reset_peak_vram_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_vram_metrics(device):
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    mib = 1024.0 * 1024.0
    return {
        "peak_vram_allocated_mb": torch.cuda.max_memory_allocated(device) / mib,
        "peak_vram_reserved_mb": torch.cuda.max_memory_reserved(device) / mib,
    }


def build_epoch_efficiency_metrics(
    epoch_seconds,
    processed_examples,
    cumulative_seconds,
):
    examples_per_second = (
        processed_examples / epoch_seconds if epoch_seconds > 0.0 else 0.0
    )
    return {
        "epoch_seconds": epoch_seconds,
        "examples_per_second": examples_per_second,
        "cumulative_gpu_hours": cumulative_seconds / 3600.0,
    }
