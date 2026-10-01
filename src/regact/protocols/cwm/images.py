"""Bounded previews in the dedicated agent-writable tmp/images directory.

Directory descriptors and no-follow opens prevent an agent-controlled symlink
from redirecting cleanup or writes outside that directory.
"""

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path


def _directory(parent: int, name: str) -> int:
    with suppress(FileExistsError):
        os.mkdir(name, 0o700, dir_fd=parent)
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


@contextmanager
def image_directory(workdir: Path) -> Iterator[int]:
    root = os.open(workdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        tmp = _directory(root, "tmp")
        try:
            images = _directory(tmp, "images")
            try:
                yield images
            finally:
                os.close(images)
        finally:
            os.close(tmp)
    finally:
        os.close(root)


def clear_images(workdir: Path) -> None:
    with image_directory(workdir) as folder:
        for name in os.listdir(folder):
            try:
                # Unlinks files/symlinks without following them.
                os.unlink(name, dir_fd=folder)
            except IsADirectoryError:
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise ValueError(
                        "Safe temporary image directory cleanup is unavailable"
                    ) from None
                shutil.rmtree(name, dir_fd=folder)


def save_preview(workdir: Path, observation_id: int, png: bytes) -> None:
    with image_directory(workdir) as folder:
        fd = os.open(
            f"obs_id_{observation_id}.png",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=folder,
        )
        with os.fdopen(fd, "wb") as output:
            output.write(png)


def select_preview_ids(ids: list[int], limit: int) -> list[int]:
    """First/last distinct observations in encounter order; odd extra goes first."""
    unique = list(dict.fromkeys(ids))
    if limit <= 0:
        return []
    if len(unique) <= limit:
        return unique
    first = (limit + 1) // 2
    last = limit // 2
    return unique[:first] + (unique[-last:] if last else [])
