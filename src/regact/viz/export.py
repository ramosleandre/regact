"""Export a curated experiments folder as a static, server-less copy of the viewer.

    python -m regact.viz.export experiments_public <out_dir>

``experiments_public/`` holds the runs to publish (links to experiment folders kept elsewhere are
fine) and ``benchmarks.yaml``, the curated list the landing page shows (see
docs/public_viewer.md). The export answers every request the viewer would make with a file under
``<out_dir>/data/``, named by a hash of the request, so the same app.js runs from any static host.
Interactive world-model playback, which runs the agent's code, is not exported.

The export is public: absolute paths are replaced, and it stops if a credential from this
machine's environment, or anything shaped like a key, appears in what it would write.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import yaml

from regact.security.runtime import _SECRET_NAME
from regact.viz.app import build_app

_STATIC = Path(__file__).parent / "static"
_FNV_OFFSET, _FNV_PRIME, _MASK = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1
# Cloudflare Pages: 25 MiB per file, 20,000 files per deployment.
_MAX_FILE_BYTES = 25 * 1024 * 1024
_MAX_FILES = 19_000
_MAX_VIDEO_BYTES = 20 * 1024 * 1024
_PACK_CHUNK_BYTES = 2 * 1024 * 1024  # a replay fetches the chunks it shows, as it plays
_MAX_REPLAY_FRAMES = 3000  # per run: beyond, the published replay keeps every n-th frame
_WORKERS = 3  # parallel exporters: frame rendering is the slow part
_KEY_SHAPES = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{20,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
_ABSOLUTE_PATH = re.compile(r"/(?:home|lus|linkhome|gpfs|scratch|work|mnt|Users)/[^\s\"'<>`)\]]+")
_MEDIA_NAME = re.compile(r"[0-9a-f]{64}\.(?:png|jpg|webp|gif)")


def request_key(url: str) -> str:
    """The data file name of one viewer request; app.js computes the same (FNV-1a, 64 bits)."""
    value = _FNV_OFFSET
    for byte in url.encode("utf-8"):
        value = ((value ^ byte) * _FNV_PRIME) & _MASK
    return f"{value:016x}"


def _url(path: str, **params: Any) -> str:
    """A request as app.js writes it: encodeURIComponent for every value."""
    if not params:
        return path
    return path + "?" + urlencode(params, safe="-_.!~*'()", quote_via=_quote)


def _quote(value: str, safe: str = "", encoding: Any = None, errors: Any = None) -> str:
    from urllib.parse import quote

    return quote(str(value), safe="-_.!~*'()")


class LeakError(RuntimeError):
    """The export would publish a credential."""


class Exporter:
    def __init__(self, root: Path, out: Path) -> None:
        from starlette.testclient import TestClient

        self.root, self.out = root, out
        self.client = TestClient(build_app(str(root)))
        self.files = 0
        self.bytes = 0
        # Credentials present on this machine: none of them may appear in the export.
        self.secrets = {
            name: value
            for name, value in os.environ.items()
            if _SECRET_NAME.search(name) and len(value) >= 12
        }
        self.prefixes = sorted(
            {str(Path.home()), str(root.resolve()), *(str(p.resolve()) for p in root.iterdir())},
            key=len,
            reverse=True,
        )

    # -- writing -----------------------------------------------------------------------------
    def clean(self, value: Any, where: str) -> str:
        """``value`` as JSON text with this machine's paths replaced, once checked for secrets."""
        text = json.dumps(_scrub(value, self.prefixes))
        for name, value in self.secrets.items():
            if value in text:
                raise LeakError(f"{where}: contains the value of ${name}")
        if (shape := _KEY_SHAPES.search(text)) is not None:
            raise LeakError(f"{where}: contains a key-shaped string ({shape.group(0)[:6]}...)")
        return text

    def write(self, name: str, data: bytes) -> None:
        if len(data) > _MAX_FILE_BYTES:
            raise ValueError(f"{name}: {len(data)} bytes exceeds the host's per-file limit")
        path = self.out / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files += 1
        self.bytes += len(data)

    def save_timeline(self, name: str) -> Any:
        """The replay timeline; a very long run keeps every n-th frame (and every mispredicted
        one), so its replay stays a reasonable download."""
        url = _url("/api/game/replay", name=name)
        response = self.get(url)
        if response is None:
            return None
        timeline = response.json()
        frames = timeline.get("frames") or []
        if len(frames) > _MAX_REPLAY_FRAMES:
            step = math.ceil(len(frames) / _MAX_REPLAY_FRAMES)
            last = len(frames) - 1
            timeline["frames"] = [
                f for i, f in enumerate(frames) if i % step == 0 or i == last or f.get("diagnostic")
            ]
            timeline["thinned"] = step
        text = self.clean(timeline, url)
        self.write(request_key(url) + ".json", text.encode())
        return json.loads(text)

    def write_pack(self, name: str, urls: dict[str, int], images: list[str]) -> None:
        """A run's frames: an index, and the images in chunks under the host's file limit. Each
        image is stored once, however many requests show it."""
        chunks: list[list[str]] = [[]]
        size = 0
        where: list[list[int]] = []
        for image in images:
            if size + len(image) > _PACK_CHUNK_BYTES and chunks[-1]:
                chunks.append([])
                size = 0
            where.append([len(chunks) - 1, len(chunks[-1])])
            chunks[-1].append(image)
            size += len(image)
        for number, chunk in enumerate(chunks):
            self.write(request_key(f"pack:{name}:{number}") + ".json", json.dumps(chunk).encode())
        index = {"chunks": len(chunks), "urls": {url: where[i] for url, i in urls.items()}}
        self.write(request_key("pack:" + name) + ".json", json.dumps(index).encode())

    def get(self, url: str) -> Any:
        response = self.client.get(url)
        if response.status_code != 200:
            return None
        return response

    def save_json(self, url: str) -> Any:
        response = self.get(url)
        if response is None:
            return None
        text = self.clean(response.json(), url)
        self.write(request_key(url) + ".json", text.encode())
        return json.loads(text)

    # -- what to export ----------------------------------------------------------------------
    def export(self, curation: dict[str, Any]) -> dict[str, Any]:
        tree = self.save_json("/api/tree")
        scopes = [""] + [node["path"] for node in _walk(tree["tree"]) if node["kind"] != "task"]
        games: dict[str, dict[str, Any]] = {}
        for scope in scopes:
            listing = self.save_json(_url("/api/games", under=scope) if scope else "/api/games")
            for entry in listing["games"]:
                games[entry["name"]] = entry
            settings = self.get(_url("/api/settings", scope=scope))
            self.write(request_key(_url("/api/settings", scope=scope)) + ".json", settings.content)
        names = sorted(games)
        shares = [names[i :: _WORKERS] for i in range(_WORKERS)]
        with ProcessPoolExecutor(max_workers=_WORKERS, initializer=os.nice, initargs=(19,)) as pool:
            for files, size in pool.map(_export_games, repeat(self.root), repeat(self.out), shares):
                self.files += files
                self.bytes += size
        site = landing(curation, games)
        self.write("site.json", self.clean(site, "benchmarks.yaml").encode())
        return site

    def export_game(self, name: str) -> None:
        detail = self.save_json(_url("/api/game", name=name))
        artifacts = self.save_json(_url("/api/game/artifacts", name=name))
        self.save_json(_url("/api/game/logs", name=name))
        timeline = self.save_timeline(name)
        index_of: dict[str, int] = {}  # request -> its image in images
        images: list[str] = []
        seen: dict[bytes, int] = {}
        for frame in (timeline or {}).get("frames", []):
            urls = [
                _url("/api/game/replay/frame", name=name, obs=frame["obs"], view=view)
                for view in ("agent", "viewer")
            ]
            if frame.get("diagnostic"):
                urls += [
                    _url(
                        "/api/game/cwm/evidence-image",
                        name=name,
                        kind="diagnostic",
                        identifier=frame["diagnostic"],
                        side="predicted",
                        view=view,
                    )
                    for view in ("agent", "viewer")
                ]
            for url in urls:
                response = self.get(url)
                if response is not None:
                    if response.content not in seen:
                        seen[response.content] = len(images)
                        encoded = base64.b64encode(_webp(response.content)).decode()
                        images.append("data:image/webp;base64," + encoded)
                    index_of[url] = seen[response.content]
        if index_of:
            self.write_pack(name, index_of, images)
        for filename in sorted(set(_media_names(detail))):
            url = _url("/api/game/tool-image", name=name, filename=filename)
            if (response := self.get(url)) is not None:
                self.write(request_key(url) + Path(filename).suffix, response.content)
        for submission in (artifacts or {}).get("submissions", []):
            for video in submission.get("videos") or []:
                url = _url("/video", game=name, submission=submission["name"], filename=video)
                path = self.root / name / "workdir" / "submissions" / submission["name"] / video
                if path.is_file() and path.stat().st_size <= _MAX_VIDEO_BYTES:
                    self.write(request_key(url) + ".mp4", path.read_bytes())


def _webp(png: bytes) -> bytes:
    """A frame as lossless WebP: the same pixels in a fifth of the bytes of the rendered PNG."""
    from PIL import Image

    out = io.BytesIO()
    Image.open(io.BytesIO(png)).save(out, "WEBP", lossless=True, method=6)
    return out.getvalue()


def _export_games(root: Path, out: Path, names: list[str]) -> tuple[int, int]:
    """One worker's share of the games; returns the files and bytes it wrote."""
    exporter = Exporter(root, out)
    for name in names:
        exporter.export_game(name)
    return exporter.files, exporter.bytes


def _scrub(value: Any, prefixes: list[str]) -> Any:
    """Every string in ``value`` with this machine's absolute paths replaced."""
    if isinstance(value, str):
        for prefix in prefixes:
            value = value.replace(prefix, "~")
        return _ABSOLUTE_PATH.sub("~/...", value)
    if isinstance(value, dict):
        return {key: _scrub(item, prefixes) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, prefixes) for item in value]
    return value


def _walk(nodes: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for node in nodes:
        yield node
        yield from _walk(node.get("children") or [])


def _media_names(value: Any) -> Iterator[str]:
    """The recorded tool images a game's conversation refers to."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _media_names(item)
    elif isinstance(value, list):
        for item in value:
            yield from _media_names(item)
    elif isinstance(value, str) and _MEDIA_NAME.fullmatch(value):
        yield value


def landing(curation: dict[str, Any], games: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The landing page: each curated benchmark as rows of (agent, model, protocol) results."""
    problems = []
    for problem in curation.get("problems") or []:
        benchmarks = []
        for bench in problem.get("benchmarks") or []:
            under = str(bench["path"]).strip("/")
            runs = [g for n, g in games.items() if n == under or n.startswith(under + "/")]
            if not runs:
                raise ValueError(f"benchmarks.yaml: no run under {under!r}")
            benchmarks.append(
                {
                    "title": bench.get("title") or under,
                    "note": bench.get("note") or "",
                    "under": under,
                    "rows": _rows(runs),
                }
            )
        problems.append(
            {
                "name": problem["name"],
                "description": problem.get("description") or "",
                "benchmarks": benchmarks,
            }
        )
    return {
        "title": curation.get("title") or "regact benchmarks",
        "intro": curation.get("intro") or "",
        "problems": problems,
    }


def _rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        agent = run.get("agent") or {}
        protocol = (run.get("metrics") or {}).get("protocol") or "policy_search"
        key = (str(agent.get("name") or "?"), str(agent.get("model") or "?"), str(protocol))
        groups.setdefault(key, []).append(run)
    rows = []
    for (agent, model, protocol), members in sorted(groups.items()):
        metric_name, values = None, []
        for run in members:
            main = (run.get("state") or {}).get("main_metrics") or {}
            if main:
                metric_name = metric_name or next(iter(main))
                if isinstance(main.get(metric_name), (int, float)):
                    values.append(float(main[metric_name]))
        rows.append(
            {
                "agent": agent,
                "model": model,
                "protocol": protocol,
                "runs": len(members),
                "solved": sum(
                    (r.get("state") or {}).get("exit_reason") == "solved" for r in members
                ),
                "metric": metric_name,
                "mean": round(statistics.fmean(values), 3) if values else None,
            }
        )
    return rows


def copy_viewer(out: Path) -> None:
    """The viewer's own files, and an index.html that switches it to the exported data."""
    shutil.copytree(_STATIC, out / "static", dirs_exist_ok=True)
    index = (_STATIC / "index.html").read_text(encoding="utf-8")
    for attribute in ("href", "src"):
        index = index.replace(f'{attribute}="/static/', f'{attribute}="static/')
    marker = '<meta name="regact-static" content="data/" />'
    index = index.replace('<meta charset="utf-8" />', f'<meta charset="utf-8" />\n    {marker}')
    (out / "index.html").write_text(index, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regact.viz.export", description=__doc__.split("\n")[0])
    parser.add_argument("experiments", help="the curated folder, e.g. experiments_public")
    parser.add_argument("out", help="where to write the site (replaced)")
    args = parser.parse_args(argv)
    root, out = Path(args.experiments), Path(args.out)
    curation = yaml.safe_load((root / "benchmarks.yaml").read_text(encoding="utf-8")) or {}
    if out.exists():
        shutil.rmtree(out)
    exporter = Exporter(root, out)
    try:
        exporter.export(curation)
    except LeakError as exc:
        shutil.rmtree(out, ignore_errors=True)
        print(f"export stopped, nothing written: {exc}", file=sys.stderr)
        return 1
    copy_viewer(out)
    if exporter.files > _MAX_FILES:
        print(f"warning: {exporter.files} data files, near the host's limit", file=sys.stderr)
    print(f"exported {exporter.files} data files, {exporter.bytes / 1e6:.1f} MB, to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
