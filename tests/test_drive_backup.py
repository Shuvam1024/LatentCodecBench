"""Drive API resume: credential file, CSV merge, and the tarball extract."""

from __future__ import annotations

import io
import json
import re
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from latentcodecbench.analysis.drive_backup import (
    CSV_NAME,
    DRIVE_FOLDER,
    DRIVE_SCOPE,
    RESULTS_FOLDER,
    TARBALL_NAME,
    _run_argv,
    _status,
    credential_payload,
    extract_clip_tarball,
    prepare_credentials,
    push_local_csv,
    refresh_colab_credentials,
    status_lines,
    write_authorized_user,
)
from latentcodecbench.analysis.drive_worker import (
    ensure_csv,
    find_named,
    main,
    upload_media,
)
from latentcodecbench.analysis.propagation import (
    PropagationError,
    PropagationRow,
    merge_propagation,
    read_propagation,
    reconcile_with_drive,
    resume_local,
    write_propagation,
)


def _row(
    *,
    clip_id: str = "alpha",
    vae: str = "wan2_1",
    source_codec: str = "libx264",
    source_crf: int = 18,
    config_sha256: str = "c" * 64,
    latent_mse: float = 1.0,
) -> PropagationRow:
    return PropagationRow(
        run_id="run",
        git_sha="abc123",
        config_sha256=config_sha256,
        clip_id=clip_id,
        vae=vae,
        source_codec=source_codec,
        source_crf=source_crf,
        preset="medium",
        width=16,
        height=16,
        frames=4,
        latent_mse=latent_mse,
        latent_relative_l2=0.1,
        latent_cosine=0.9,
        psnr_recon_clean=30.0,
        psnr_recon_degraded=40.0,
        psnr_degraded_clean=28.0,
        blocking_clean=0.0,
        blocking_degraded=1.0,
        blocking_recon=0.5,
        blocking_retention=0.5,
        ringing_clean=0.0,
        ringing_degraded=1.0,
        ringing_recon=0.2,
        ringing_retention=0.2,
        clean_encode_seconds=0.1,
        encode_seconds=0.2,
        decode_seconds=0.3,
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
    )


def test_merge_unions_disjoint_keys_and_keeps_the_local_row() -> None:
    local = [_row(clip_id="alpha", latent_mse=1.0), _row(clip_id="beta", source_crf=30)]
    remote = [_row(clip_id="alpha", latent_mse=9.0), _row(clip_id="gamma")]
    merged = merge_propagation(local, remote)
    by_key = {(row.clip_id, row.source_crf): row for row in merged}
    assert set(by_key) == {("alpha", 18), ("beta", 30), ("gamma", 18)}
    assert by_key[("alpha", 18)].latent_mse == 1.0


def test_merge_rejects_a_different_config_and_a_duplicate_key() -> None:
    with pytest.raises(PropagationError, match="different configs"):
        merge_propagation([_row(config_sha256="a" * 64)], [_row(config_sha256="b" * 64)])
    with pytest.raises(PropagationError, match="duplicate row"):
        merge_propagation([_row(), _row(latent_mse=2.0)], [])
    with pytest.raises(PropagationError, match="more than one config"):
        merge_propagation(
            [_row(config_sha256="a" * 64), _row(clip_id="beta", config_sha256="b" * 64)],
            [],
        )


def test_resume_local_merges_and_refuses_to_rewrite_on_a_config_mismatch(tmp_path: Path) -> None:
    local_path = tmp_path / "local.csv"
    remote_path = tmp_path / "remote.csv"
    write_propagation(local_path, [_row(clip_id="alpha", latent_mse=1.0)])
    write_propagation(remote_path, [_row(clip_id="beta", latent_mse=4.0)])
    status, count = resume_local(local_path, remote_path)
    assert status == "merged"
    assert count == 2
    loaded = {(row.clip_id, row.latent_mse) for row in read_propagation(local_path)}
    assert loaded == {("alpha", 1.0), ("beta", 4.0)}

    clash = tmp_path / "clash.csv"
    write_propagation(clash, [_row(config_sha256="d" * 64)])
    before = local_path.read_bytes()
    with pytest.raises(PropagationError, match="different configs"):
        resume_local(local_path, clash)
    assert local_path.read_bytes() == before


def test_resume_local_covers_the_empty_sides(tmp_path: Path) -> None:
    local_path = tmp_path / "local.csv"
    remote_path = tmp_path / "remote.csv"
    assert resume_local(local_path, None) == ("local-only", 0)
    assert not local_path.exists()

    remote_path.write_bytes(b"")
    write_propagation(local_path, [_row()])
    assert resume_local(local_path, remote_path) == ("local-only", 1)

    local_path.unlink()
    write_propagation(remote_path, [_row(clip_id="gamma"), _row(clip_id="delta")])
    status, count = resume_local(local_path, remote_path)
    assert status == "remote-only"
    assert count == 2
    assert {row.clip_id for row in read_propagation(local_path)} == {"gamma", "delta"}


def test_prepare_credentials_reads_the_saved_file(tmp_path: Path) -> None:
    path = tmp_path / "drive-credentials.json"
    assert prepare_credentials(path) == "needs-auth"
    path.write_text("not json", encoding="utf-8")
    assert prepare_credentials(path) == "needs-auth"
    path.write_text("[]", encoding="utf-8")
    assert prepare_credentials(path) == "needs-auth"

    future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    past = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    soon = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
    path.write_text(json.dumps({"refresh_token": "r", "expiry": past}), encoding="utf-8")
    assert prepare_credentials(path) == "ready"
    path.write_text(json.dumps({"token": "t", "expiry": future}), encoding="utf-8")
    assert prepare_credentials(path) == "ready"
    path.write_text(json.dumps({"token": "t"}), encoding="utf-8")
    assert prepare_credentials(path) == "ready"
    path.write_text(json.dumps({"token": "t", "expiry": past}), encoding="utf-8")
    assert prepare_credentials(path) == "needs-auth"
    path.write_text(json.dumps({"token": "t", "expiry": soon}), encoding="utf-8")
    assert prepare_credentials(path) == "needs-auth"
    path.write_text(json.dumps({"token": "t", "expiry": "not-a-date"}), encoding="utf-8")
    assert prepare_credentials(path) == "needs-auth"


def test_write_authorized_user_is_private_and_payload_comes_from_the_object(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "drive-credentials.json"
    write_authorized_user(path, {"token": "abc", "refresh_token": "r"})
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["token"] == "abc"
    with pytest.raises(ValueError, match="no token"):
        write_authorized_user(path, {})

    class _JsonCreds:
        def to_json(self) -> str:
            return json.dumps({"token": "from-json", "refresh_token": "r"})

    assert credential_payload(_JsonCreds())["token"] == "from-json"

    class _Duck:
        def __init__(self) -> None:
            self.token = "access"
            self.refresh_token = None
            self.token_uri = "https://oauth2.googleapis.com/token"
            self.client_id = "id"
            self.client_secret = "secret"
            self.scopes = ["https://www.googleapis.com/auth/drive"]

    payload = credential_payload(_Duck())
    assert payload["token"] == "access"
    assert payload["client_id"] == "id"
    assert payload["scopes"] == ["https://www.googleapis.com/auth/drive"]

    class _Refreshed:
        def __init__(self) -> None:
            self.token = "ya29.live"
            self.refresh_token = None
            self.expiry = datetime(2030, 1, 1, tzinfo=UTC)
            self.scopes = [DRIVE_SCOPE]
            self.token_uri = None
            self.client_id = None
            self.client_secret = None

        def to_json(self) -> str:
            return json.dumps({"token": "stale"})

    refreshed = credential_payload(_Refreshed())
    assert refreshed["token"] == "ya29.live"
    assert refreshed["expiry"].startswith("2030-01-01")


def test_status_lines_count_local_and_drive_rows(tmp_path: Path) -> None:
    local = tmp_path / "artifact-propagation.csv"
    remote = tmp_path / "remote.csv"
    write_propagation(local, [_row(), _row(clip_id="beta")])
    write_propagation(remote, [_row()])
    lines = status_lines(local, remote, file_id="abc", drive_status="ok")
    assert lines[0] == f"local 2 rows {local}"
    assert lines[1] == (f"drive 1 rows My Drive/{DRIVE_FOLDER}/{RESULTS_FOLDER}/{CSV_NAME} id=abc")
    missing = status_lines(local, remote, file_id="", drive_status="needs-auth")
    assert missing[1].startswith("drive needs-auth ")
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    counted = status_lines(local, empty, file_id="abc", drive_status="ok")
    assert "drive 0 rows " in counted[1]


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as handle:
        for name, payload in members.items():
            data = io.BytesIO(payload)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, data)


def test_extract_clip_tarball_flattens_y4m_and_rejects_unsafe_members(tmp_path: Path) -> None:
    archive = tmp_path / "canonical-clips.tar"
    dest = tmp_path / "clips"
    _tar(archive, {"clips/aspen.y4m": b"y4m-a", "notes.txt": b"ignore"})
    assert extract_clip_tarball(archive, dest) == 1
    assert (dest / "aspen.y4m").read_bytes() == b"y4m-a"
    assert not (dest / "notes.txt").exists()

    unsafe = tmp_path / "unsafe.tar"
    _tar(unsafe, {"../note.txt": b"nope"})
    with pytest.raises(ValueError, match="unsafe"):
        extract_clip_tarball(unsafe, tmp_path / "unsafe-out")
    absolute = tmp_path / "absolute.tar"
    _tar(absolute, {"/tmp/aspen.y4m": b"nope"})
    with pytest.raises(ValueError, match="unsafe"):
        extract_clip_tarball(absolute, tmp_path / "absolute-out")

    duplicate = tmp_path / "duplicate.tar"
    _tar(duplicate, {"clips/aspen.y4m": b"one", "other/aspen.y4m": b"two"})
    with pytest.raises(ValueError, match="duplicate"):
        extract_clip_tarball(duplicate, tmp_path / "dup-out")

    empty = tmp_path / "empty.tar"
    _tar(empty, {"readme.txt": b"no y4m"})
    with pytest.raises(ValueError, match="no y4m"):
        extract_clip_tarball(empty, tmp_path / "empty-out")


class _Store:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}
        self.creates: list[str] = []
        self.updates: list[dict[str, Any]] = []
        self._n = 0

    def list_query(self, query: str) -> dict[str, Any]:
        name = re.search(r"name = '((?:\\'|[^'])*)'", query)
        parent = re.search(r"'((?:\\'|[^'])*)' in parents", query)
        assert name is not None and parent is not None
        found_name = name.group(1).replace("\\'", "'").replace("\\\\", "\\")
        file_id = self.items.get((parent.group(1), found_name))
        if file_id is None:
            return {"files": []}
        return {"files": [{"id": file_id, "name": found_name}]}

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        self._n += 1
        file_id = f"id{self._n}"
        parents = body["parents"]
        assert isinstance(parents, list)
        self.items[(str(parents[0]), str(body["name"]))] = file_id
        self.creates.append(str(body["name"]))
        return {"id": file_id}


class _Request:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def execute(self) -> dict[str, Any]:
        return self._payload


class _Files:
    def __init__(self, store: _Store) -> None:
        self._store = store

    def list(self, *, q: str, spaces: str, fields: str, pageSize: int) -> _Request:
        del spaces, fields, pageSize
        return _Request(self._store.list_query(q))

    def create(self, *, body: dict[str, Any], fields: str) -> _Request:
        del fields
        return _Request(self._store.create(body))

    def update(self, *, fileId: str, media_body: object) -> _Request:
        self._store.updates.append({"fileId": fileId, "media_body": media_body})
        return _Request({})


class _Service:
    def __init__(self, store: _Store) -> None:
        self._files = _Files(store)

    def files(self) -> _Files:
        return self._files


def test_ensure_csv_reuses_the_file_id_and_upload_updates_it(tmp_path: Path) -> None:
    store = _Store()
    service = _Service(store)
    first = ensure_csv(service)
    second = ensure_csv(service)
    assert first == second
    assert store.creates.count(CSV_NAME) == 1
    assert store.creates.count(DRIVE_FOLDER) == 1
    assert store.creates.count(RESULTS_FOLDER) == 1
    assert find_named(_Service(_Store()), [DRIVE_FOLDER], TARBALL_NAME) is None

    source = tmp_path / CSV_NAME
    source.write_text("clip\n", encoding="utf-8")
    upload_media(service, first, source, uploader=lambda path: {"path": path})
    assert store.updates == [{"fileId": first, "media_body": {"path": str(source)}}]


@pytest.mark.parametrize(
    ("code", "name"),
    [
        (None, "timeout"),
        (0, "ok"),
        (2, "auth-failed"),
        (3, "absent"),
        (4, "unavailable"),
        (1, "failed"),
        (9, "failed"),
    ],
)
def test_drive_call_status_names(code: int | None, name: str) -> None:
    assert _status(code) == name


def test_run_argv_kills_a_hung_child() -> None:
    code, stdout, stderr = _run_argv(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=0.2,
    )
    assert code is None
    assert stdout == ""
    assert stderr == ""


def test_refresh_colab_credentials_saves_a_drive_access_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "drive-credentials.json"

    class _Creds:
        def __init__(self) -> None:
            self.token: str | None = None
            self.refresh_token = None
            self.scopes = [DRIVE_SCOPE]
            self.expiry = datetime.now(UTC) + timedelta(hours=1)
            self.token_uri = "https://oauth2.googleapis.com/token"
            self.client_id = None
            self.client_secret = None

        def refresh(self, request: object) -> None:
            assert request == "request"
            self.token = "ya29.fresh"

    creds = _Creds()

    class _Auth:
        @staticmethod
        def default(scopes: list[str] | None = None) -> tuple[_Creds, str]:
            assert scopes == [DRIVE_SCOPE]
            return creds, "project"

    class _Requests:
        @staticmethod
        def Request() -> str:
            return "request"

    def fake_import(name: str, package: str | None = None) -> Any:
        del package
        if name == "google.auth":
            return _Auth
        if name == "google.auth.transport.requests":
            return _Requests
        raise ImportError(name)

    monkeypatch.setattr(
        "latentcodecbench.analysis.drive_backup.importlib.import_module",
        fake_import,
    )
    assert refresh_colab_credentials(path) == "ready"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["token"] == "ya29.fresh"
    assert prepare_credentials(path) == "ready"


def test_refresh_colab_credentials_logs_when_the_token_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "drive-credentials.json"

    class _Creds:
        token = None
        refresh_token = None

        def refresh(self, request: object) -> None:
            del request

    class _Auth:
        @staticmethod
        def default(scopes: list[str] | None = None) -> tuple[_Creds, None]:
            del scopes
            return _Creds(), None

    class _Requests:
        @staticmethod
        def Request() -> object:
            return object()

    def fake_import(name: str, package: str | None = None) -> Any:
        del package
        if name == "google.auth":
            return _Auth
        if name == "google.auth.transport.requests":
            return _Requests
        raise ImportError(name)

    monkeypatch.setattr(
        "latentcodecbench.analysis.drive_backup.importlib.import_module",
        fake_import,
    )
    assert refresh_colab_credentials(path) == "needs-auth"
    assert not path.exists()
    captured = capsys.readouterr()
    assert "DRIVE AUTH UNAVAILABLE" in captured.err
    assert "credentials have no token" in captured.err


def test_push_retries_on_the_next_call_and_uploads_the_full_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credentials = tmp_path / "drive-credentials.json"
    file_id_path = tmp_path / "artifact-propagation.id"
    csv_path = tmp_path / "artifact-propagation.csv"
    csv_path.write_text("clip,vae\nalpha,wan2_1\nbeta,hunyuan\n", encoding="utf-8")
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    refresh_count = {"n": 0}
    uploads: list[tuple[str, str]] = []

    def refresh(path: Path) -> str:
        refresh_count["n"] += 1
        if refresh_count["n"] == 1:
            return "needs-auth"
        write_authorized_user(path, {"token": "ya29.new", "expiry": future})
        return "ready"

    def ensure(path: Path, *, timeout: float) -> tuple[str, str]:
        del path, timeout
        return "ok", "file-9"

    def upload(path: Path, file_id: str, source: Path, *, timeout: float) -> str:
        del path, timeout
        uploads.append((file_id, source.read_text(encoding="utf-8")))
        return "ok"

    monkeypatch.setattr("latentcodecbench.analysis.drive_backup.ensure_results_file", ensure)
    monkeypatch.setattr("latentcodecbench.analysis.drive_backup.upload_file", upload)
    assert (
        push_local_csv(
            credentials,
            file_id_path,
            csv_path,
            timeout=5,
            refresh=refresh,
        )
        == "needs-auth"
    )
    assert uploads == []
    assert (
        push_local_csv(
            credentials,
            file_id_path,
            csv_path,
            timeout=5,
            refresh=refresh,
        )
        == "ok"
    )
    assert file_id_path.read_text(encoding="utf-8").strip() == "file-9"
    assert uploads == [("file-9", "clip,vae\nalpha,wan2_1\nbeta,hunyuan\n")]


def test_push_replaces_a_rejected_token_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credentials = tmp_path / "drive-credentials.json"
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    write_authorized_user(credentials, {"token": "old", "expiry": future})
    file_id_path = tmp_path / "artifact-propagation.id"
    file_id_path.write_text("file-1\n", encoding="utf-8")
    csv_path = tmp_path / "artifact-propagation.csv"
    csv_path.write_text("full\n", encoding="utf-8")
    seen: list[str] = []

    def refresh(path: Path) -> str:
        write_authorized_user(path, {"token": "new", "expiry": future})
        return "ready"

    def upload(path: Path, file_id: str, source: Path, *, timeout: float) -> str:
        del file_id, source, timeout
        token = str(json.loads(path.read_text(encoding="utf-8"))["token"])
        seen.append(token)
        return "auth-failed" if token == "old" else "ok"

    monkeypatch.setattr("latentcodecbench.analysis.drive_backup.upload_file", upload)
    assert (
        push_local_csv(
            credentials,
            file_id_path,
            csv_path,
            timeout=5,
            refresh=refresh,
        )
        == "ok"
    )
    assert seen == ["old", "new"]
    assert json.loads(credentials.read_text(encoding="utf-8"))["token"] == "new"


def test_worker_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    assert main([]) == 5
    monkeypatch.setattr("latentcodecbench.analysis.drive_worker._service", lambda _path: object())
    assert main(["nope", "creds.json"]) == 5


def test_worker_reports_a_missing_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cred = tmp_path / "creds.json"
    cred.write_text("{}", encoding="utf-8")

    def boom(name: str, package: str | None = None) -> Any:
        raise ImportError(name)

    monkeypatch.setattr("latentcodecbench.analysis.drive_worker.importlib.import_module", boom)
    assert main(["ensure-csv", str(cred)]) == 4


def test_service_accepts_an_access_token_without_a_client_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "creds.json"
    cred.write_text(json.dumps({"token": "ya29.test"}), encoding="utf-8")
    built: dict[str, Any] = {}

    class _Creds:
        def __init__(self, **kwargs: Any) -> None:
            self.token = kwargs.get("token")
            self.refresh_token = kwargs.get("refresh_token")
            self.expired = False
            self.valid = True

        @classmethod
        def from_authorized_user_info(cls, info: dict[str, Any]) -> _Creds:
            del info
            raise ValueError("missing fields")

    class _CredentialsMod:
        Credentials = _Creds

    class _Requests:
        @staticmethod
        def Request() -> object:
            return object()

    class _Discovery:
        @staticmethod
        def build(
            api: str,
            version: str,
            *,
            credentials: object,
            cache_discovery: bool,
        ) -> str:
            del version, cache_discovery
            built["credentials"] = credentials
            built["api"] = api
            return "service"

    def fake_import(name: str, package: str | None = None) -> Any:
        del package
        if name == "google.oauth2.credentials":
            return _CredentialsMod
        if name == "google.auth.transport.requests":
            return _Requests
        if name == "googleapiclient.discovery":
            return _Discovery
        raise ImportError(name)

    monkeypatch.setattr(
        "latentcodecbench.analysis.drive_worker.importlib.import_module",
        fake_import,
    )
    from latentcodecbench.analysis.drive_worker import _service

    assert _service(str(cred)) == "service"
    loaded = built["credentials"]
    assert isinstance(loaded, _Creds)
    assert loaded.token == "ya29.test"
    assert built["api"] == "drive"


def _write_rows(path: Path, rows: list[PropagationRow]) -> None:
    write_propagation(path, rows)


def test_reconcile_fills_a_short_local_file_from_drive(tmp_path: Path) -> None:
    local = tmp_path / "local.csv"
    remote = tmp_path / "remote.csv"
    _write_rows(local, [_row(clip_id="alpha")])
    _write_rows(remote, [_row(clip_id="alpha"), _row(clip_id="beta"), _row(clip_id="gamma")])
    assert reconcile_with_drive(local, remote) == "ok"
    assert {row.clip_id for row in read_propagation(local)} == {"alpha", "beta", "gamma"}


def test_reconcile_refuses_a_config_mismatch_and_leaves_local_alone(tmp_path: Path) -> None:
    local = tmp_path / "local.csv"
    remote = tmp_path / "remote.csv"
    _write_rows(local, [_row(clip_id="alpha")])
    _write_rows(remote, [_row(clip_id="beta", config_sha256="d" * 64)])
    before = local.read_text(encoding="utf-8")
    assert reconcile_with_drive(local, remote) == "refused-config"
    assert local.read_text(encoding="utf-8") == before


def test_reconcile_accepts_an_empty_drive_file(tmp_path: Path) -> None:
    local = tmp_path / "local.csv"
    remote = tmp_path / "remote.csv"
    remote.write_text("", encoding="utf-8")
    _write_rows(local, [_row(clip_id="alpha")])
    assert reconcile_with_drive(local, remote) == "ok"


def _ready_credentials(tmp_path: Path) -> tuple[Path, Path]:
    credentials = tmp_path / "drive-credentials.json"
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    write_authorized_user(credentials, {"token": "tok", "expiry": future})
    file_id_path = tmp_path / "artifact-propagation.id"
    file_id_path.write_text("file-1\n", encoding="utf-8")
    return credentials, file_id_path


def test_push_merges_drive_rows_before_uploading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credentials, file_id_path = _ready_credentials(tmp_path)
    csv_path = tmp_path / "artifact-propagation.csv"
    _write_rows(csv_path, [_row(clip_id="alpha")])
    drive_rows = [_row(clip_id=name) for name in ("alpha", "beta", "gamma", "delta")]
    uploaded: list[set[str]] = []

    def download(path: Path, file_id: str, dest: Path, *, timeout: float) -> str:
        del path, file_id, timeout
        _write_rows(dest, drive_rows)
        return "ok"

    def upload(path: Path, file_id: str, source: Path, *, timeout: float) -> str:
        del path, file_id, timeout
        uploaded.append({row.clip_id for row in read_propagation(source)})
        return "ok"

    monkeypatch.setattr("latentcodecbench.analysis.drive_backup.download_file", download)
    monkeypatch.setattr("latentcodecbench.analysis.drive_backup.upload_file", upload)
    status = push_local_csv(
        credentials, file_id_path, csv_path, timeout=5, reconcile=reconcile_with_drive
    )
    assert status == "ok"
    assert uploaded == [{"alpha", "beta", "gamma", "delta"}]


@pytest.mark.parametrize(
    ("download_status", "reconcile_status"),
    [("timeout", "ok"), ("ok", "refused-fewer-rows")],
)
def test_push_does_not_upload_when_drive_cannot_be_merged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    download_status: str,
    reconcile_status: str,
) -> None:
    credentials, file_id_path = _ready_credentials(tmp_path)
    csv_path = tmp_path / "artifact-propagation.csv"
    _write_rows(csv_path, [_row(clip_id="alpha")])
    uploads: list[Path] = []

    def download(path: Path, file_id: str, dest: Path, *, timeout: float) -> str:
        del path, file_id, dest, timeout
        return download_status

    def upload(path: Path, file_id: str, source: Path, *, timeout: float) -> str:
        del path, file_id, timeout
        uploads.append(source)
        return "ok"

    def reconcile(local: Path, remote: Path) -> str:
        del local, remote
        return reconcile_status

    monkeypatch.setattr("latentcodecbench.analysis.drive_backup.download_file", download)
    monkeypatch.setattr("latentcodecbench.analysis.drive_backup.upload_file", upload)
    status = push_local_csv(credentials, file_id_path, csv_path, timeout=5, reconcile=reconcile)
    assert status == ("timeout" if download_status != "ok" else reconcile_status)
    assert uploads == []
