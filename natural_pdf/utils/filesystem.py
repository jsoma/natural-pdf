"""Small filesystem transaction primitives shared by writers."""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import secrets
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple, Union

logger = logging.getLogger(__name__)


class PathIdentity(NamedTuple):
    """Stable identity for one filesystem object."""

    device: int
    inode: int


def path_identity(path: Union[str, Path]) -> PathIdentity:
    """Return the device/inode pair for *path* without following symlinks."""

    path_stat = os.lstat(path)
    return PathIdentity(path_stat.st_dev, path_stat.st_ino)


def path_has_identity(path: Union[str, Path], identity: PathIdentity) -> bool:
    """Return whether *path* still names the same object as *identity*."""

    try:
        return path_identity(path) == identity
    except OSError:
        return False


def rename_noreplace(source: Union[str, Path], destination: Union[str, Path]) -> None:
    """Atomically rename *source* only when *destination* does not exist.

    ``os.rename`` is not sufficient for directories on POSIX: it may replace
    an empty destination directory, creating a data-loss race after an
    existence check.  Linux and macOS both expose a kernel-level exclusive
    rename operation, while Windows' normal rename already has no-replace
    semantics.
    """

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)

    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        # <sys/stdio.h>: RENAME_EXCL
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
        return

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - only very old libc releases
            raise OSError(
                errno.ENOTSUP,
                "Atomic no-replace rename is unavailable on this system",
                os.fspath(destination),
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        # <linux/fs.h>: RENAME_NOREPLACE; AT_FDCWD is -100.
        result = renameat2(-100, source_bytes, -100, destination_bytes, 0x00000001)
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
        return

    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        os.rename(source, destination)
        return

    # Safety is more important than silently falling back to the racy POSIX
    # behavior. Callers can choose a non-atomic publication strategy if they
    # need to support a platform without an exclusive rename primitive.
    raise OSError(
        errno.ENOTSUP,
        "Atomic no-replace rename is unavailable on this system",
        os.fspath(destination),
    )


_RENAME_NOREPLACE_UNSUPPORTED = {
    errno.EINVAL,
    errno.ENOSYS,
    errno.ENOTSUP,
}
if hasattr(errno, "EOPNOTSUPP"):
    _RENAME_NOREPLACE_UNSUPPORTED.add(errno.EOPNOTSUPP)


def publish_directory_noreplace(source: Union[str, Path], destination: Union[str, Path]) -> None:
    """Publish a directory without replacing an existing destination.

    The kernel's atomic no-replace rename is preferred. On platforms or
    filesystems without that primitive, an exclusive destination directory is
    claimed first and populated while mode ``0700`` prevents other users from
    observing partial contents. The portable fallback is not an atomic
    visibility boundary, but it never overwrites an existing pathname and
    rolls moved entries back on failure.
    """

    try:
        rename_noreplace(source, destination)
        return
    except OSError as exc:
        if exc.errno not in _RENAME_NOREPLACE_UNSUPPORTED:
            raise

    source_path = Path(source)
    destination_path = Path(destination)
    source_stat = os.lstat(source_path)
    if not stat.S_ISDIR(source_stat.st_mode):
        raise NotADirectoryError(os.fspath(source_path))

    # mkdir is the portable atomic no-clobber claim. Keep it private until all
    # entries have moved and only then expose the source directory's mode.
    os.mkdir(destination_path, mode=0o700)
    moved_names = []
    try:
        with os.scandir(source_path) as entries:
            source_names = [entry.name for entry in entries]
        for name in source_names:
            os.rename(source_path / name, destination_path / name)
            moved_names.append(name)
        os.chmod(destination_path, stat.S_IMODE(source_stat.st_mode))
        os.rmdir(source_path)
    except BaseException:
        # Both roots are private transaction directories. Restore entries to
        # staging so callers retain a complete recoverable build.
        for name in reversed(moved_names):
            destination_child = destination_path / name
            source_child = source_path / name
            if os.path.lexists(destination_child) and not os.path.lexists(source_child):
                try:
                    os.rename(destination_child, source_child)
                except OSError:
                    pass
        try:
            os.rmdir(destination_path)
        except OSError:
            pass
        raise


def remove_private_directory(path: Union[str, Path], *, recursive: bool = True) -> bool:
    """Best-effort cleanup for a uniquely named private directory."""

    try:
        if recursive:
            shutil.rmtree(path)
        else:
            os.rmdir(path)
        return True
    except OSError:
        return False


@contextmanager
def atomic_output_path(destination: Union[str, Path]) -> Iterator[Path]:
    """Yield a unique sibling path and promote it only after successful use.

    Writers put their complete output at the yielded path inside a private
    directory beside the destination. A normal context exit atomically
    replaces ``destination``; an exception removes only that private directory
    and leaves any existing destination untouched. Keeping both paths on the
    same filesystem makes the final replace atomic.
    """

    destination_path = Path(destination)
    temporary_directory_name = tempfile.mkdtemp(
        prefix=f"{destination_path.name}.tmp-",
        suffix=f"-{secrets.token_hex(16)}",
        dir=destination_path.parent,
    )
    temporary_directory = Path(temporary_directory_name)
    temporary_path = temporary_directory / destination_path.name
    # Reserve the output filename too. Some writers (notably pikepdf/qpdf)
    # replace this inode as part of their own safe-save implementation, so the
    # private 0700 directory—not the child inode—is our ownership boundary.
    descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        yield temporary_path
        temporary_stat = os.lstat(temporary_path)
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise RuntimeError(
                f"Atomic output writer did not produce a regular file: {temporary_path}"
            )
        os.replace(temporary_path, destination_path)
        # Promotion has committed. Failure to clean an unexpected writer
        # sidecar must not turn a successful export into a reported failure.
        if not remove_private_directory(temporary_directory):
            logger.warning("Could not clean atomic-output directory: %s", temporary_directory)
    except BaseException:
        if not remove_private_directory(temporary_directory):
            logger.warning("Could not clean atomic-output directory: %s", temporary_directory)
        raise
