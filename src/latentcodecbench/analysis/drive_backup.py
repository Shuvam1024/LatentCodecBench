"""Back up the propagation CSV through the Drive API.

The notebook writes the CSV on local disk. This module downloads the
Drive copy at startup, merges it with the local file, and uploads the
local file by id after each row. Each API call is a child process with
a timeout. Nothing here mounts a filesystem.

Credentials live in a JSON file the caller chooses. On Colab that path
is ``/content/latentcodecbench/drive-credentials.json``, which survives
Restart session. A missing or expired file is ``needs-auth``. The
notebook prompts once in that case and writes the file. Later cells,
including after a kernel restart, load the file and do not prompt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DRIVE_FOLDER = "LatentCodecBench"
RESULTS_FOLDER = "results"
CSV_NAME = "artifact-propagation.csv"
TARBALL_NAME = "canonical-clips.tar"
CSV_TIMEOUT = 60.0
TARBALL_TIMEOUT = 600.0

_WORKER = "latentcodecbench.analysis.drive_worker"


def prepare_credentials(path: Path) -> str:
    """``ready`` when ``path`` can authorize a call, else ``needs-auth``.

    A refresh token counts as ready even when the access token is expired,
    because the child refreshes it. An access token with no refresh token
    counts only until shortly before its expiry. An expiry that cannot be
    read counts as ``needs-auth``.
    """
    payload = _read_json(path)
    if payload is None:
        return "needs-auth"
    if payload.get("refresh_token"):
        return "ready"
    token = payload.get("token") or payload.get("access_token")
    if not token:
        return "needs-auth"
    expiry = payload.get("expiry")
    if not isinstance(expiry, str) or expiry == "":
        return "ready"
    moment = _parse_expiry(expiry)
    if moment is None or moment <= datetime.now(UTC) + timedelta(minutes=2):
        return "needs-auth"
    return "ready"


def write_authorized_user(path: Path, payload: dict[str, Any]) -> None:
    """Atomically store an authorized-user JSON file, mode 0600."""
    token = payload.get("token") or payload.get("access_token")
    if not token and not payload.get("refresh_token"):
        raise ValueError("credentials have no token")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def credential_payload(creds: Any) -> dict[str, Any]:
    """JSON-ready dict from a google-auth credentials object."""
    to_json = getattr(creds, "to_json", None)
    if callable(to_json):
        parsed = json.loads(to_json())
        if isinstance(parsed, dict) and (parsed.get("token") or parsed.get("refresh_token")):
            return dict(parsed)
    scopes = list(getattr(creds, "scopes", None) or [])
    return {
        "token": getattr(creds, "token", None),
        "refresh_token": getattr(creds, "refresh_token", None),
        "token_uri": getattr(creds, "token_uri", None) or "https://oauth2.googleapis.com/token",
        "client_id": getattr(creds, "client_id", None),
        "client_secret": getattr(creds, "client_secret", None),
        "scopes": scopes,
    }


def ensure_results_file(credentials: Path, *, timeout: float = CSV_TIMEOUT) -> tuple[str, str]:
    """Return ``(status, file_id)`` for the results CSV, creating it if needed."""
    status, stdout, _stderr = drive_call(["ensure-csv", str(credentials)], timeout=timeout)
    if status != "ok":
        return status, ""
    file_id = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if not file_id:
        return "failed", ""
    return "ok", file_id


def download_file(
    credentials: Path,
    file_id: str,
    dest: Path,
    *,
    timeout: float = CSV_TIMEOUT,
) -> str:
    """Download ``file_id`` onto ``dest``. Status is ``ok`` or an error name."""
    status, _stdout, _stderr = drive_call(
        ["download", str(credentials), file_id, str(dest)],
        timeout=timeout,
    )
    return status


def upload_file(
    credentials: Path,
    file_id: str,
    source: Path,
    *,
    timeout: float = CSV_TIMEOUT,
) -> str:
    """Replace the Drive file ``file_id`` with ``source``."""
    status, _stdout, _stderr = drive_call(
        ["upload", str(credentials), file_id, str(source)],
        timeout=timeout,
    )
    return status


def download_named(
    credentials: Path,
    filename: str,
    dest: Path,
    *,
    timeout: float = TARBALL_TIMEOUT,
) -> str:
    """Download ``filename`` from the ``LatentCodecBench`` folder."""
    status, _stdout, _stderr = drive_call(
        ["download-name", str(credentials), filename, str(dest)],
        timeout=timeout,
    )
    return status


def drive_call(args: list[str], *, timeout: float) -> tuple[str, str, str]:
    """Run the Drive worker. Returns ``(status, stdout, stderr)``."""
    code, stdout, stderr = _run_argv(
        [sys.executable, "-m", _WORKER, *args],
        timeout=timeout,
    )
    return _status(code), stdout, stderr


def status_lines(
    local_path: Path,
    remote_path: Path | None,
    *,
    file_id: str,
    drive_status: str,
) -> list[str]:
    """Local and Drive row counts for the notebook status cell."""
    from latentcodecbench.analysis.propagation import read_propagation

    local_count = len(read_propagation(local_path)) if local_path.is_file() else 0
    location = f"My Drive/{DRIVE_FOLDER}/{RESULTS_FOLDER}/{CSV_NAME}"
    lines = [f"local {local_count} rows {local_path}"]
    readable = drive_status == "ok" and remote_path is not None and remote_path.is_file()
    if not readable:
        lines.append(f"drive {drive_status} {location}")
        return lines
    assert remote_path is not None
    remote_count = 0 if remote_path.stat().st_size == 0 else len(read_propagation(remote_path))
    ident = file_id or "(no id)"
    lines.append(f"drive {remote_count} rows {location} id={ident}")
    return lines


def extract_clip_tarball(archive: Path, dest: Path) -> int:
    """Extract ``.y4m`` members of ``archive`` into ``dest`` as flat names.

    A member that is absolute, contains ``..``, or repeats a file name
    raises ``ValueError`` and leaves ``dest`` unchanged for that member's
    write. Returns how many y4m files were written.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    seen: set[str] = set()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            name = _flat_y4m_name(member.name)
            if name is None:
                continue
            if name in seen:
                raise ValueError(f"duplicate y4m name {name} in {archive}")
            seen.add(name)
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"missing payload for {member.name}")
            target = dest / name
            temporary = target.with_name(target.name + ".partial")
            with temporary.open("wb") as out:
                while True:
                    chunk = extracted.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(temporary, target)
            written += 1
    if written == 0:
        raise ValueError(f"{archive} contains no y4m files")
    return written


def _flat_y4m_name(member_name: str) -> str | None:
    path = Path(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe tar member {member_name}")
    if path.suffix != ".y4m":
        return None
    return path.name


def _status(code: int | None) -> str:
    if code is None:
        return "timeout"
    if code == 0:
        return "ok"
    if code == 2:
        return "auth-failed"
    if code == 3:
        return "absent"
    if code == 4:
        return "unavailable"
    return "failed"


def _run_argv(argv: list[str], *, timeout: float) -> tuple[int | None, str, str]:
    proc = subprocess.Popen(
        argv,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            os.killpg(proc.pid, 9)
        proc.communicate()
        return None, "", ""
    return proc.returncode, stdout, stderr


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_expiry(text: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)
