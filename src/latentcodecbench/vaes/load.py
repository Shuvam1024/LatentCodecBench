"""Load one pinned VAE from its ``vae`` subfolder.

The call is ``AutoencoderKL*.from_pretrained(repo, subfolder="vae", revision=...)``.
That requests the VAE config and weight file. The transformer is not requested.
"""

from __future__ import annotations

import importlib
from typing import Any

from latentcodecbench.vaes.checkpoints import CHECKPOINTS
from latentcodecbench.vaes.spec import GEOMETRIES
from latentcodecbench.vaes.wrapper import VaeUnavailable, VideoVae


def _diffusers() -> Any:
    try:
        return importlib.import_module("diffusers")
    except ImportError as exc:
        raise VaeUnavailable("video VAEs need the vae extra (torch and diffusers)") from exc


def load_vae(name: str, *, dtype: Any | None = None, device: Any | None = None) -> VideoVae:
    """Construct :class:`VideoVae` from the pinned checkpoint.

    ``dtype`` is forwarded as ``torch_dtype``. ``device`` is applied with
    ``.to(device)`` after the weights load, and does not change that dtype.
    """
    try:
        checkpoint = CHECKPOINTS[name]
    except KeyError as exc:
        raise KeyError(name) from exc
    geometry = GEOMETRIES[name]
    cls = getattr(_diffusers(), geometry.diffusers_class)
    kwargs: dict[str, Any] = {
        "subfolder": checkpoint.subfolder,
        "revision": checkpoint.revision,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    module = cls.from_pretrained(checkpoint.repo_id, **kwargs)
    if device is not None:
        module = module.to(device)
    return VideoVae(geometry, module)
