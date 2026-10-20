"""Write the git commit into the wheel, then restore the source file."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

_ASSIGNMENT = re.compile(r'^GIT_SHA = ".*"$', re.MULTILINE)
_BUILD_INFO = Path("src") / "latentcodecbench" / "_build_info.py"


def git_head(root: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    sha = completed.stdout.strip()
    if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
        return sha
    return ""


def with_sha(source: str, sha: str) -> str:
    updated, count = _ASSIGNMENT.subn(f'GIT_SHA = "{sha}"', source, count=1)
    if count != 1:
        message = "src/latentcodecbench/_build_info.py has no GIT_SHA assignment"
        raise RuntimeError(message)
    return updated


def get_build_hook() -> type:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    class CustomBuildHook(BuildHookInterface):
        def initialize(self, version: str, build_data: dict[str, Any]) -> None:
            sha = git_head(self.root)
            if not sha:
                return
            path = Path(self.root) / _BUILD_INFO
            original = path.read_text(encoding="utf-8")
            updated = with_sha(original, sha)
            if updated == original:
                return
            path.write_text(updated, encoding="utf-8")
            self._restore = (path, original)

        def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
            restore = getattr(self, "_restore", None)
            if restore is None:
                return
            path, original = restore
            path.write_text(original, encoding="utf-8")
            self._restore = None

    return CustomBuildHook
