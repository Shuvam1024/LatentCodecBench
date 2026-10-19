"""Time one encode/decode and record the reconstruction.

Pixels are in ``[-1, 1]``, so PSNR uses a peak of 2. The function does not
load weights. A CUDA out-of-memory error turns tiling on and retries once.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from latentcodecbench.metrics.psnr import psnr
from latentcodecbench.vaes.checkpoints import CHECKPOINTS
from latentcodecbench.vaes.wrapper import VideoVae, _torch

# Peak for pixels stored in [-1, 1].
PSNR_MAX_VALUE = 2.0

SMOKE_FIELDS: tuple[str, ...] = (
    "name",
    "repo_id",
    "revision",
    "license_name",
    "subfolder",
    "stored_dtype",
    "runtime_dtype",
    "spatial_tiling",
    "framewise_encoding",
    "framewise_decoding",
    "retried_with_tiling",
    "oom",
    "encode_seconds",
    "decode_seconds",
    "peak_vram_bytes",
    "psnr_db",
    "psnr_max_value",
    "latent_shape",
    "error",
)


def runtime_dtype_name(torch: Any) -> str:
    """float16 on T4 (capability 7), bfloat16 from capability 8, else float32.

    A T4 has float16 tensor cores and no bfloat16 ones. L4 and A100 are
    capability 8 or 9 and do have bfloat16.
    """
    if bool(torch.cuda.is_available()):
        major, _minor = torch.cuda.get_device_capability()
        if int(major) >= 8:
            return "bfloat16"
        return "float16"
    return "float32"


def runtime_dtype(torch: Any) -> Any:
    """The torch dtype :func:`runtime_dtype_name` names."""
    return getattr(torch, runtime_dtype_name(torch))


def tiling_flags(module: Any) -> dict[str, bool]:
    """Spatial tiling, and the framewise flags when the class has them."""
    return {
        "spatial_tiling": bool(getattr(module, "use_tiling", False)),
        "framewise_encoding": bool(getattr(module, "use_framewise_encoding", False)),
        "framewise_decoding": bool(getattr(module, "use_framewise_decoding", False)),
    }


def enable_memory_tiling(module: Any) -> None:
    """Turn on the memory switches the loaded class actually has.

    Wan's ``enable_tiling`` is spatial. LTX and Hunyuan also chunk time.
    On LTX the encode path reads ``use_framewise_decoding`` before it will
    chunk, so both framewise flags are set.
    """
    enable = getattr(module, "enable_tiling", None)
    if enable is not None:
        enable()
    if hasattr(module, "use_framewise_encoding"):
        module.use_framewise_encoding = True
    if hasattr(module, "use_framewise_decoding"):
        module.use_framewise_decoding = True


def measure_reconstruction(
    vae: VideoVae,
    video: Any,
    *,
    runtime_dtype: str,
) -> dict[str, Any]:
    """Encode and decode ``video``. One tiling retry after an out-of-memory error."""
    torch = _torch()
    checkpoint = CHECKPOINTS[vae.geometry.name]
    row = _blank_row(checkpoint, runtime_dtype)
    _copy_flags(row, vae.module)
    for attempt in (1, 2):
        try:
            encode_seconds, decode_seconds, latents, decoded = _forward(vae, video, torch)
        except Exception as exc:
            row["peak_vram_bytes"] = _peak_bytes(torch)
            if attempt == 1 and _is_oom(exc):
                _release(torch)
                enable_memory_tiling(vae.module)
                row["retried_with_tiling"] = True
                _copy_flags(row, vae.module)
                continue
            row["oom"] = _is_oom(exc)
            row["error"] = _error_text(exc)
            return row
        row["encode_seconds"] = encode_seconds
        row["decode_seconds"] = decode_seconds
        row["peak_vram_bytes"] = _peak_bytes(torch)
        row["latent_shape"] = [int(v) for v in latents.shape[1:]]
        row["psnr_db"] = _psnr(video, decoded)
        _copy_flags(row, vae.module)
        return row
    return row


def load_failure_row(name: str, runtime_dtype: str, exc: BaseException) -> dict[str, Any]:
    """A smoke row for a checkpoint that never finished loading."""
    row = _blank_row(CHECKPOINTS[name], runtime_dtype)
    row["oom"] = _is_oom(exc)
    row["error"] = _error_text(exc)
    return row


def write_smoke(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    clip: dict[str, Any],
) -> tuple[Path, Path]:
    """Write ``path`` as JSON and a sibling CSV with the same stem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    csv_path = path.with_suffix(".csv")
    document = {"clip": clip, "rows": rows}
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SMOKE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in SMOKE_FIELDS})
    return json_path, csv_path


def _blank_row(checkpoint: Any, runtime_dtype: str) -> dict[str, Any]:
    return {
        "name": checkpoint.name,
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "license_name": checkpoint.license_name,
        "subfolder": checkpoint.subfolder,
        "stored_dtype": checkpoint.stored_dtype,
        "runtime_dtype": runtime_dtype,
        "spatial_tiling": False,
        "framewise_encoding": False,
        "framewise_decoding": False,
        "retried_with_tiling": False,
        "oom": False,
        "encode_seconds": None,
        "decode_seconds": None,
        "peak_vram_bytes": None,
        "psnr_db": None,
        "psnr_max_value": PSNR_MAX_VALUE,
        "latent_shape": None,
        "error": None,
    }


def _copy_flags(row: dict[str, Any], module: Any) -> None:
    row.update(tiling_flags(module))


def _forward(vae: VideoVae, video: Any, torch: Any) -> tuple[float, float, Any, Any]:
    _sync(torch)
    if bool(torch.cuda.is_available()):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    latents = vae.encode(video)
    _sync(torch)
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    decoded = vae.decode(latents)
    _sync(torch)
    decode_seconds = time.perf_counter() - started
    return encode_seconds, decode_seconds, latents, decoded


def _psnr(reference: Any, distorted: Any) -> float:
    ref = reference.detach().float().cpu().numpy()
    dist = distorted.detach().float().cpu().numpy()
    return psnr(ref, dist, max_value=PSNR_MAX_VALUE)


def _sync(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize()


def _peak_bytes(torch: Any) -> int | None:
    if not bool(torch.cuda.is_available()):
        return None
    return int(torch.cuda.max_memory_allocated())


def _release(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return "x".join(str(part) for part in value)
    return str(value)
