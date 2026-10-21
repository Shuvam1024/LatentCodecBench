"""Copy files through a directory the caller already has.

These helpers are the directory copy from the earlier sweeps. The
artifact notebook does not call them. It talks to the Drive API from
``drive_backup`` and keeps its credential file on local disk.

Each copy still runs in a child process with a timeout. A hung read is
killed, and the caller keeps the local file. Nothing here calls
``drive.mount``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from latentcodecbench.analysis.table import transfer_file

_DIR_STATE = """
import os
import sys

sys.exit(0 if os.path.isdir(sys.argv[1]) else 3)
"""

_LIST = """
import os
import sys

directory, suffix = sys.argv[1], sys.argv[2]
if not os.path.isdir(directory):
    sys.exit(3)
for name in sorted(os.listdir(directory)):
    path = os.path.join(directory, name)
    if name.endswith(suffix) and os.path.isfile(path):
        print(name)
"""


@dataclass(frozen=True)
class StageResult:
    """How a directory copy went.

    ``status`` is ``ok``, ``absent``, ``timeout``, or ``failed``.
    ``copied`` is files written on this call. ``present`` is files that
    were already at the destination and were left alone.
    """

    status: str
    copied: int
    present: int


def pull_resume(local: Path, remote: Path, *, timeout: float) -> str:
    """Replace ``local`` with ``remote`` when ``remote`` exists.

    Returns ``copied``, ``absent``, ``timeout``, or ``failed``. An absent
    remote leaves ``local`` as it was. A timeout also leaves ``local``:
    the child is killed before the destination is renamed into place.
    """
    return _copy_status(transfer_file(remote, local, timeout=timeout))


def push_copy(local: Path, remote: Path, *, timeout: float, require_dir: Path) -> str:
    """Copy ``local`` onto ``remote`` when ``require_dir`` is already a directory.

    Returns ``copied``, ``skipped``, ``timeout``, or ``failed``. ``skipped``
    means ``require_dir`` is absent, so this function does not create a
    stand-in directory on the local disk. A timeout while checking the
    directory, or while copying, returns ``timeout`` and keeps ``local``.
    """
    state = directory_state(require_dir, timeout=timeout)
    if state == "timeout":
        return "timeout"
    if state == "absent":
        return "skipped"
    if state != "present":
        return "failed"
    return _copy_status(transfer_file(local, remote, timeout=timeout))


def stage_files(
    remote_dir: Path,
    local_dir: Path,
    *,
    suffix: str,
    timeout: float,
) -> StageResult:
    """Copy ``suffix`` files from ``remote_dir`` that are missing locally.

    A missing remote directory is ``absent``. A timeout on the listing, or
    on one file, stops the copy and returns ``timeout`` with the counts
    from the files already handled. Files already in ``local_dir`` are
    counted as ``present`` and are not overwritten.
    """
    status, names = _list_files(remote_dir, suffix=suffix, timeout=timeout)
    if status != "ok":
        return StageResult(status=status, copied=0, present=0)
    local_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    present = 0
    for name in names:
        dest = local_dir / name
        if dest.is_file():
            present += 1
            continue
        code = transfer_file(remote_dir / name, dest, timeout=timeout)
        if code == 0:
            copied += 1
            continue
        if code is None:
            return StageResult(status="timeout", copied=copied, present=present)
        return StageResult(status="failed", copied=copied, present=present)
    return StageResult(status="ok", copied=copied, present=present)


def directory_state(path: Path, *, timeout: float) -> str:
    """``present``, ``absent``, ``timeout``, or ``failed``.

    The check runs in a child process so a stuck stat does not hold the
    caller.
    """
    code, _stdout = _child(_DIR_STATE, [str(path)], timeout)
    if code is None:
        return "timeout"
    if code == 0:
        return "present"
    if code == 3:
        return "absent"
    return "failed"


def _list_files(directory: Path, *, suffix: str, timeout: float) -> tuple[str, tuple[str, ...]]:
    code, stdout = _child(_LIST, [str(directory), suffix], timeout)
    if code is None:
        return "timeout", ()
    if code == 3:
        return "absent", ()
    if code != 0:
        return "failed", ()
    names = tuple(line for line in stdout.splitlines() if line)
    return "ok", names


def _copy_status(code: int | None) -> str:
    if code == 0:
        return "copied"
    if code == 3:
        return "absent"
    if code is None:
        return "timeout"
    return "failed"


def _child(code: str, args: list[str], timeout: float) -> tuple[int | None, str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, _stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            os.killpg(proc.pid, 9)
        proc.communicate()
        return None, ""
    return proc.returncode, stdout
