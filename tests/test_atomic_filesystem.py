import errno
from pathlib import Path

import pytest

import natural_pdf.utils.filesystem as filesystem
from natural_pdf.utils.filesystem import (
    atomic_output_path,
    publish_directory_noreplace,
    rename_noreplace,
)


def test_atomic_output_path_promotes_complete_file(tmp_path):
    destination = tmp_path / "result.pdf"
    destination.write_bytes(b"old")

    with atomic_output_path(destination) as temporary:
        assert temporary.parent.parent == destination.parent
        assert temporary != destination
        assert temporary.exists()
        temporary.write_bytes(b"new")
        assert destination.read_bytes() == b"old"

    assert destination.read_bytes() == b"new"
    assert list(tmp_path.glob("result.pdf.tmp-*")) == []


def test_atomic_output_path_failure_preserves_destination(tmp_path):
    destination = tmp_path / "result.pdf"
    destination.write_bytes(b"old")
    temporary: Path

    with pytest.raises(RuntimeError, match="write failed"):
        with atomic_output_path(destination) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("write failed")

    assert destination.read_bytes() == b"old"
    assert not temporary.exists()


def test_cleanup_failure_after_commit_does_not_report_export_failure(tmp_path, monkeypatch):
    destination = tmp_path / "result.pdf"

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(filesystem.shutil, "rmtree", fail_cleanup)

    with atomic_output_path(destination) as temporary:
        temporary.write_bytes(b"complete")

    assert destination.read_bytes() == b"complete"


def test_cleanup_failure_does_not_mask_writer_exception(tmp_path, monkeypatch):
    destination = tmp_path / "result.pdf"

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(filesystem.shutil, "rmtree", fail_cleanup)

    with pytest.raises(RuntimeError, match="original writer failure"):
        with atomic_output_path(destination) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("original writer failure")

    assert not destination.exists()


def test_rename_noreplace_preserves_existing_empty_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        rename_noreplace(source, destination)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert (source / "new.txt").read_text(encoding="utf-8") == "new"


def test_publish_directory_uses_safe_portable_fallback(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir(mode=0o750)
    (source / "nested").mkdir()
    (source / "nested" / "value.txt").write_text("value", encoding="utf-8")
    destination = tmp_path / "destination"

    def unsupported_rename(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "not supported")

    monkeypatch.setattr(filesystem, "rename_noreplace", unsupported_rename)

    publish_directory_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "value"


def test_portable_publish_fallback_never_replaces_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "valuable.txt").write_text("foreign", encoding="utf-8")

    def unsupported_rename(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "not supported")

    monkeypatch.setattr(filesystem, "rename_noreplace", unsupported_rename)

    with pytest.raises(FileExistsError):
        publish_directory_noreplace(source, destination)

    assert (destination / "valuable.txt").read_text(encoding="utf-8") == "foreign"
    assert (source / "new.txt").read_text(encoding="utf-8") == "new"
