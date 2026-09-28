"""Freeze submitted Python sources and static local imports, never the whole workdir."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from regact.protocols.cwm.store import atomic_json, canonical

MAX_BUNDLE_BYTES = 10 * 1024 * 1024
MAX_BUNDLE_FILES = 256


def workspace_file(root: Path, rel: str, *, create: bool = False) -> int:
    """Open through directory descriptors: reject symlink races and special files."""
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or any(p in ("..", ".") for p in parts):
        raise ValueError("invalid workspace-relative path")
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(part, dir_fd=parent)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= (os.O_WRONLY | os.O_CREAT | os.O_EXCL) if create else os.O_RDONLY
        fd = os.open(parts[-1], flags, 0o600, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("submitted files must be regular files")
        return fd
    except OSError as exc:
        raise ValueError(f"cannot safely open workspace file {rel}: {exc.strerror}") from exc
    finally:
        os.close(parent)


def read_source(root: Path, rel: str) -> bytes:
    with os.fdopen(workspace_file(root, rel), "rb") as handle:
        data = handle.read(MAX_BUNDLE_BYTES + 1)
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("submitted source exceeds 10 MiB")
    return data


def write_plan(root: Path, rel: str, contents: str) -> None:
    with os.fdopen(workspace_file(root, rel, create=True), "w") as handle:
        handle.write(contents)


def runtime_fingerprint() -> dict[str, Any]:
    entry = Path(__file__).with_name("worker_entry.py")
    return {
        "python": sys.version,
        "worker_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
        "packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
    }


def verify_bundle(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("runtime") != runtime_fingerprint():
        raise ValueError(
            "saved CWM runtime/dependencies differ; replay requires the original runtime"
        )
    actual = {}
    if {p.relative_to(path).as_posix() for p in path.rglob("*.py")} != set(manifest["files"]):
        raise ValueError("archived bundle file list changed")
    for name in manifest["files"]:
        file = path / name
        if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
            raise ValueError("invalid archived bundle path")
        actual[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if (
        actual != manifest["files"]
        or hashlib.sha256(canonical(actual).encode()).hexdigest() != manifest["digest"]
    ):
        raise ValueError("archived bundle integrity check failed")
    return cast(dict[str, Any], manifest)


def snapshot(
    workdir: Path, destination: Path, entries: list[str], *, model: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = workdir.resolve()
    sources: dict[str, bytes] = {}
    if model is not None:
        for path in model.rglob("*.py"):
            sources[path.relative_to(model).as_posix()] = path.read_bytes()
    pending = list(entries)
    seen: set[str] = set()
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = root / rel
        if path.is_symlink() or any(
            parent.is_symlink()
            for parent in path.parents
            if parent != root and root in parent.parents
        ):
            raise ValueError(f"symlinks are not supported in submitted bundles: {rel}")
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.suffix != ".py":
            raise ValueError(f"missing/invalid submitted Python file: {rel}")
        # Accepted model always wins over mutable workdir copies for planning/episodes.
        if rel not in sources or model is None:
            sources[rel] = read_source(root, rel)
        if len(sources) > MAX_BUNDLE_FILES or sum(map(len, sources.values())) > MAX_BUNDLE_BYTES:
            raise ValueError("submitted bundle exceeds 256 files or 10 MiB")
        tree = ast.parse(sources[rel], filename=rel)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = Path(rel).parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    prefix = ".".join(parent.parts)
                    module = ".".join(filter(None, (prefix, node.module)))
                else:
                    module = node.module or ""
                modules = [module, *[".".join(filter(None, (module, a.name))) for a in node.names]]
            for module in modules:
                if not module:
                    continue
                parts = module.split(".")
                for base in (Path(), Path("world_model")):
                    candidates = [
                        base / Path(*parts).with_suffix(".py"),
                        base / Path(*parts) / "__init__.py",
                    ]
                    candidates += [
                        base / Path(*parts[:i]) / "__init__.py" for i in range(1, len(parts))
                    ]
                    for candidate in candidates:
                        key = candidate.as_posix()
                        if (root / candidate).is_file() and key not in sources:
                            pending.append(key)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(sources.items())}
    identifier = hashlib.sha256(canonical(hashes).encode()).hexdigest()
    target = destination / identifier
    if not target.exists():
        destination.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".bundle-", dir=destination))
        try:
            for name, data in sources.items():
                p = temp / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            atomic_json(
                temp / "manifest.json",
                {
                    "digest": identifier,
                    "files": hashes,
                    "runtime": runtime_fingerprint(),
                    "bytes": sum(map(len, sources.values())),
                },
            )
            temp.rename(target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return target, json.loads((target / "manifest.json").read_text())


def description(path: Path) -> str:
    if path.is_symlink() or path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("invalid submitted file")
    text = ast.get_docstring(ast.parse(path.read_text()))
    if not text or not text.strip():
        raise ValueError(f"{path.name} needs a module-level docstring describing the goal")
    return text.strip()
