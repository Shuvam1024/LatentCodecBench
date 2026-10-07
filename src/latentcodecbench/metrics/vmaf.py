"""VMAF through ffmpeg's libvmaf filter.

The first input is the distorted clip and the second is the reference,
which is the order libvmaf documents. The score is the pooled mean in
the JSON log (``pooled_metrics.vmaf.mean``). libvmaf is not a pure
identity: two identical clips score near 100, not a bitwise 100, because
the model uses integer features. Callers should treat "above 99" as the
identical-input check.

``n_threads=1`` keeps the log stable across machines.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from latentcodecbench.data.preprocess import ffmpeg_binary


class VmafError(RuntimeError):
    """ffmpeg could not produce a VMAF score."""


def vmaf(distorted: Path, reference: Path) -> float:
    """Pooled mean VMAF of ``distorted`` against ``reference``."""
    if not distorted.is_file():
        raise FileNotFoundError(distorted)
    if not reference.is_file():
        raise FileNotFoundError(reference)
    with tempfile.TemporaryDirectory(prefix="lcb-vmaf-") as tmp:
        log_path = Path(tmp) / "vmaf.json"
        completed = subprocess.run(
            [
                ffmpeg_binary(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(distorted),
                "-i",
                str(reference),
                "-lavfi",
                f"libvmaf=log_fmt=json:log_path={log_path}:n_threads=1",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not log_path.is_file():
            detail = completed.stderr.strip() or "libvmaf produced no log"
            raise VmafError(detail)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    try:
        mean = payload["pooled_metrics"]["vmaf"]["mean"]
    except (KeyError, TypeError) as exc:
        raise VmafError("libvmaf log has no pooled_metrics.vmaf.mean") from exc
    return float(mean)
