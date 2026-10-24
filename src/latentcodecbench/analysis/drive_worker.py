"""Drive API commands for the propagation notebook.

Run as ``python -m latentcodecbench.analysis.drive_worker``. The parent
process kills this child on timeout. Exit 0 prints a file id when the
command creates or finds one. Exit 2 is an auth failure, 3 a missing
file, 4 a missing client library.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from latentcodecbench.analysis.drive_backup import (
    DRIVE_FOLDER,
    RESULTS_FOLDER,
    check_results_name,
)

_FOLDER = "application/vnd.google-apps.folder"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print("usage: <command> <credentials> ...", file=sys.stderr)
        return 5
    command, cred_path, rest = args[0], args[1], args[2:]
    try:
        service = _service(cred_path)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    try:
        if command == "ensure-csv":
            print(ensure_named(service, "artifact-propagation.csv"))
            return 0
        if command == "ensure-named" and len(rest) == 1:
            print(ensure_named(service, rest[0]))
            return 0
        if command == "download" and len(rest) == 2:
            download_media(service, rest[0], Path(rest[1]))
            return 0
        if command == "upload" and len(rest) == 2:
            upload_media(service, rest[0], Path(rest[1]))
            return 0
        if command == "download-name" and len(rest) == 2:
            file_id = find_named(service, [DRIVE_FOLDER], rest[0])
            if file_id is None:
                print(f"missing {rest[0]}", file=sys.stderr)
                return 3
            download_media(service, file_id, Path(rest[1]))
            return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown command {command}", file=sys.stderr)
    return 5


def ensure_csv(service: Any) -> str:
    """File id of ``LatentCodecBench/results/artifact-propagation.csv``."""
    return ensure_named(service, "artifact-propagation.csv")


def ensure_named(service: Any, name: str) -> str:
    """File id of ``LatentCodecBench/results/<name>``, created when absent.

    A second call returns the same id. ``name`` has to be one CSV file
    name, not a path.
    """
    checked = check_results_name(name)
    top = ensure_folder(service, DRIVE_FOLDER, "root")
    results = ensure_folder(service, RESULTS_FOLDER, top)
    existing = find_child(service, checked, results)
    if existing is not None:
        return existing
    created = (
        service.files()
        .create(
            body={"name": checked, "mimeType": "text/csv", "parents": [results]},
            fields="id",
        )
        .execute()
    )
    return str(created["id"])


def ensure_folder(service: Any, name: str, parent: str) -> str:
    found = find_child(service, name, parent, mime=_FOLDER)
    if found is not None:
        return found
    created = (
        service.files()
        .create(
            body={"name": name, "mimeType": _FOLDER, "parents": [parent]},
            fields="id",
        )
        .execute()
    )
    return str(created["id"])


def find_named(service: Any, folders: list[str], filename: str) -> str | None:
    """Walk ``folders`` from My Drive and return the file id, or ``None``."""
    parent = "root"
    for name in folders:
        found = find_child(service, name, parent, mime=_FOLDER)
        if found is None:
            return None
        parent = found
    return find_child(service, filename, parent)


def find_child(service: Any, name: str, parent: str, *, mime: str | None = None) -> str | None:
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{escaped}' and '{parent}' in parents and trashed = false"
    if mime is not None:
        query += f" and mimeType = '{mime}'"
    listed = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id,name)", pageSize=10)
        .execute()
    )
    files = listed.get("files", [])
    if not files:
        return None
    return str(files[0]["id"])


def download_media(service: Any, file_id: str, dest: Path) -> None:
    http = importlib.import_module("googleapiclient.http")
    request = service.files().get_media(fileId=file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".partial")
    with temporary.open("wb") as handle:
        downloader = http.MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()
    os.replace(temporary, dest)


def upload_media(service: Any, file_id: str, source: Path, *, uploader: Any | None = None) -> None:
    if uploader is None:
        http = importlib.import_module("googleapiclient.http")
        media: Any = http.MediaFileUpload(str(source), mimetype="text/csv", resumable=False)
    else:
        media = uploader(str(source))
    service.files().update(fileId=file_id, media_body=media).execute()


def _service(cred_path: str) -> Any:
    try:
        credentials_mod = importlib.import_module("google.oauth2.credentials")
        requests_mod = importlib.import_module("google.auth.transport.requests")
        discovery = importlib.import_module("googleapiclient.discovery")
    except ImportError:
        print("google API client is not installed", file=sys.stderr)
        raise SystemExit(4) from None
    creds = _load_credentials(credentials_mod, cred_path)
    if creds.expired and creds.refresh_token:
        creds.refresh(requests_mod.Request())
        _save_credentials(Path(cred_path), creds)
    token = getattr(creds, "token", None)
    refresh = getattr(creds, "refresh_token", None)
    if not token and not refresh:
        print("saved Drive credentials have no token", file=sys.stderr)
        raise SystemExit(2)
    if getattr(creds, "valid", True) is False and not refresh:
        print("saved Drive credentials are expired", file=sys.stderr)
        raise SystemExit(2)
    return discovery.build("drive", "v3", credentials=creds, cache_discovery=False)


def _load_credentials(credentials_mod: Any, cred_path: str) -> Any:
    """Load a saved token, including an access token Colab did not fully shape."""
    raw = json.loads(Path(cred_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("saved Drive credentials are not a JSON object", file=sys.stderr)
        raise SystemExit(2)
    try:
        return credentials_mod.Credentials.from_authorized_user_info(raw)
    except ValueError:
        token = raw.get("token") or raw.get("access_token")
        refresh = raw.get("refresh_token")
        if not token and not refresh:
            print("saved Drive credentials have no token", file=sys.stderr)
            raise SystemExit(2) from None
        return credentials_mod.Credentials(
            token=token,
            refresh_token=refresh,
            token_uri=raw.get("token_uri") or "https://oauth2.googleapis.com/token",
            client_id=raw.get("client_id"),
            client_secret=raw.get("client_secret"),
            scopes=raw.get("scopes"),
        )


def _save_credentials(path: Path, creds: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(creds.to_json(), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
