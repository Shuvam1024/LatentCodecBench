"""Back up the propagation CSV through the Drive API.

The notebook writes the CSV on local disk. This module downloads the
Drive copy at startup, merges it with the local file, and uploads the
local file by id after each row. Each API call is a child process with
a timeout. Nothing here mounts a filesystem.

Credentials live in a JSON file the caller chooses. On Colab that path
is ``/content/latentcodecbench/drive-credentials.json``, which survives
Restart session. A missing or expired file is ``needs-auth``. The
notebook prompts once in that case, refreshes an access token in-process,
and writes the file. Later cells, including after a kernel restart, load
the file and do not prompt. The upload child reads that access token.
Colab's preinstalled Drive libraries are used as they are.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DRIVE_FOLDER = "LatentCodecBench"
RESULTS_FOLDER = "results"
CSV_NAME = "artifact-propagation.csv"
TARBALL_NAME = "canonical-clips.tar"
_RESULTS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.csv$")
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
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
    """JSON-ready dict from a google-auth credentials object.

    The live access token wins over ``to_json``. Colab's object often has
    an empty token until ``refresh(Request())``, and ``to_json`` can still
    describe that empty state.
    """
    parsed: dict[str, Any] = {}
    to_json = getattr(creds, "to_json", None)
    if callable(to_json):
        try:
            loaded = json.loads(to_json())
        except (TypeError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            parsed = dict(loaded)
    token = getattr(creds, "token", None) or parsed.get("token") or parsed.get("access_token")
    refresh = getattr(creds, "refresh_token", None) or parsed.get("refresh_token")
    if token:
        parsed["token"] = token
    if refresh:
        parsed["refresh_token"] = refresh
    expiry = _expiry_text(getattr(creds, "expiry", None))
    if expiry is None and isinstance(parsed.get("expiry"), str):
        expiry = parsed["expiry"]
    if expiry:
        parsed["expiry"] = expiry
    scopes = list(getattr(creds, "scopes", None) or parsed.get("scopes") or [])
    parsed["scopes"] = scopes
    if not parsed.get("token_uri"):
        parsed["token_uri"] = (
            getattr(creds, "token_uri", None) or "https://oauth2.googleapis.com/token"
        )
    client_id = getattr(creds, "client_id", None)
    if client_id:
        parsed["client_id"] = client_id
    client_secret = getattr(creds, "client_secret", None)
    if client_secret:
        parsed["client_secret"] = client_secret
    return parsed


def refresh_colab_credentials(path: Path) -> str:
    """Refresh Colab application-default credentials into ``path``.

    Returns ``ready`` or ``needs-auth``. This does not prompt. The notebook
    calls ``authenticate_user`` first when a person has to consent, then
    ``google.auth.default(scopes=[drive])`` and ``credentials.refresh``.
    The access token written here is what the upload child sends.
    """
    try:
        google_auth = importlib.import_module("google.auth")
        requests_mod = importlib.import_module("google.auth.transport.requests")
    except ImportError:
        print("DRIVE AUTH UNAVAILABLE: google.auth is not installed", file=sys.stderr)
        return "needs-auth"
    try:
        creds, _project = google_auth.default(scopes=[DRIVE_SCOPE])
        creds.refresh(requests_mod.Request())
        write_authorized_user(path, credential_payload(creds))
    except Exception as exc:
        print(f"DRIVE AUTH UNAVAILABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return "needs-auth"
    return prepare_credentials(path)


def push_local_csv(
    credentials: Path,
    file_id_path: Path,
    csv_path: Path,
    *,
    timeout: float,
    refresh: Callable[[Path], str] | None = None,
    reconcile: Callable[[Path, Path], str] | None = None,
) -> str:
    """Upload the whole local CSV. Each call re-reads ``credentials``.

    With ``reconcile``, the Drive copy is downloaded next to ``csv_path``
    first and ``reconcile(csv_path, drive_copy)`` merges it into the local
    file. Anything other than ``ok`` from that call is returned and nothing
    is uploaded, and so is a failed download. This is what stops a short
    local file (a fresh VM, or a resume cell that was skipped) from
    replacing a longer Drive file.

    A failed call does not stick. The next call checks the file again, so
    re-running the Drive auth cell while the sweep is paused is picked up
    on the next row. Once auth works, ``csv_path`` is uploaded in full,
    including rows that were only on local disk. An ``auth-failed`` response
    drops the saved token and tries ``refresh`` once before giving up.
    """
    if _ready(credentials, refresh) != "ready":
        return "needs-auth"
    file_id = ""
    if file_id_path.is_file():
        file_id = file_id_path.read_text(encoding="utf-8").strip()
    if not file_id:
        status, file_id = _ensure_file_id(
            credentials,
            file_id_path,
            timeout=timeout,
            refresh=refresh,
        )
        if not file_id:
            return status
    if reconcile is not None:
        drive_copy = csv_path.with_name(csv_path.stem + ".drive" + csv_path.suffix)
        drive_copy.unlink(missing_ok=True)
        fetched = download_file(credentials, file_id, drive_copy, timeout=timeout)
        if fetched != "ok":
            print(f"DRIVE download {fetched}; not uploading", file=sys.stderr)
            return fetched
        merged = reconcile(csv_path, drive_copy)
        if merged != "ok":
            print(f"DRIVE upload refused: {merged}", file=sys.stderr)
            return merged
    status = upload_file(credentials, file_id, csv_path, timeout=timeout)
    if status != "auth-failed" or refresh is None:
        return status
    _drop_token(credentials)
    if refresh(credentials) != "ready":
        return "auth-failed"
    return upload_file(credentials, file_id, csv_path, timeout=timeout)


def check_results_name(name: str) -> str:
    """Reject a results file name that is not a single CSV path segment."""
    if _RESULTS_NAME.fullmatch(name) is None:
        raise ValueError(f"refusing results name {name!r}")
    return name


def ensure_results_file(credentials: Path, *, timeout: float = CSV_TIMEOUT) -> tuple[str, str]:
    """Return ``(status, file_id)`` for the propagation CSV, creating it if needed."""
    return ensure_named_file(credentials, CSV_NAME, timeout=timeout)


def ensure_named_file(
    credentials: Path,
    name: str,
    *,
    timeout: float = CSV_TIMEOUT,
) -> tuple[str, str]:
    """Return ``(status, file_id)`` for ``LatentCodecBench/results/<name>``."""
    checked = check_results_name(name)
    status, stdout, _stderr = drive_call(
        ["ensure-named", str(credentials), checked],
        timeout=timeout,
    )
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
    filename: str = CSV_NAME,
    read_rows: Callable[[Path], list[Any]] | None = None,
) -> list[str]:
    """Local and Drive row counts for the notebook status cell.

    ``read_rows`` defaults to the propagation reader. A notebook that
    writes a different CSV passes its own reader and ``filename``.
    """
    if read_rows is None:
        from latentcodecbench.analysis.propagation import read_propagation

        reader: Callable[[Path], list[Any]] = read_propagation
    else:
        reader = read_rows
    local_count = len(reader(local_path)) if local_path.is_file() else 0
    location = f"My Drive/{DRIVE_FOLDER}/{RESULTS_FOLDER}/{filename}"
    lines = [f"local {local_count} rows {local_path}"]
    readable = drive_status == "ok" and remote_path is not None and remote_path.is_file()
    if not readable:
        lines.append(f"drive {drive_status} {location}")
        return lines
    assert remote_path is not None
    remote_count = 0 if remote_path.stat().st_size == 0 else len(reader(remote_path))
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


def _ready(credentials: Path, refresh: Callable[[Path], str] | None) -> str:
    if prepare_credentials(credentials) == "ready":
        return "ready"
    if refresh is None:
        return "needs-auth"
    return refresh(credentials)


def _ensure_file_id(
    credentials: Path,
    file_id_path: Path,
    *,
    timeout: float,
    refresh: Callable[[Path], str] | None,
) -> tuple[str, str]:
    status, file_id = ensure_results_file(credentials, timeout=timeout)
    if status == "auth-failed" and refresh is not None:
        _drop_token(credentials)
        if refresh(credentials) == "ready":
            status, file_id = ensure_results_file(credentials, timeout=timeout)
    if status != "ok" or not file_id:
        return status or "failed", ""
    file_id_path.parent.mkdir(parents=True, exist_ok=True)
    file_id_path.write_text(file_id + "\n", encoding="utf-8")
    return "ok", file_id


def _drop_token(credentials: Path) -> None:
    with suppress(OSError):
        credentials.unlink()


def _expiry_text(expiry: Any) -> str | None:
    if isinstance(expiry, datetime):
        moment = expiry if expiry.tzinfo is not None else expiry.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat()
    if isinstance(expiry, str) and expiry.strip():
        return expiry
    return None


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
