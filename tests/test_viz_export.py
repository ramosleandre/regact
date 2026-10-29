"""The static public export of the viewer (regact.viz.export)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from regact.viz.export import _url, main, request_key
from test_viz import _make_experiment

_URLS = [
    "/api/tree",
    "/api/games?under=arc%2Fcwm",
    _url("/api/game", name="arc/cwm/2026-10-20_00-22-46/ls20"),
    _url("/api/game/replay/frame", name="a b/é", obs=3, view="agent"),
    "pack:arc/cwm/2026-10-20_00-22-46/ls20",
]


def test_request_keys_match_the_viewers() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    source = (Path(__file__).parents[1] / "src/regact/viz/static/app.js").read_text()
    start = source.index("function requestKey(url)")
    function = source[start : source.index("\n}\n", start) + 2]
    script = function + f"\nfor (const u of {json.dumps(_URLS)}) console.log(requestKey(u));"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    assert out.split() == [request_key(u) for u in _URLS]
    # encodeURIComponent's spelling, as the viewer builds every query.
    assert _url("/api/games", under="a b/(x)") == "/api/games?under=a%20b%2F(x)"


def _curated(tmp_path: Path) -> Path:
    """A public root whose benchmark is a link to an experiment kept elsewhere."""
    experiment = Path(_make_experiment(tmp_path / "private"))
    public = tmp_path / "public"
    public.mkdir()
    (public / "bench").symlink_to(experiment)
    (public / "benchmarks.yaml").write_text(
        "title: Test\nproblems:\n  - name: ARC-AGI-3\n    benchmarks:\n"
        "      - title: One run\n        path: bench\n"
    )
    return public


def test_the_export_answers_the_viewers_requests_from_files(tmp_path: Path) -> None:
    public, out = _curated(tmp_path), tmp_path / "site"
    assert main([str(public), str(out)]) == 0
    index = (out / "index.html").read_text()
    assert 'name="regact-static" content="data/"' in index and '"/static/' not in index
    game_file = request_key(_url("/api/game", name="bench/ls20")) + ".json"
    game = json.loads((out / "data" / game_file).read_text())
    assert game["name"] == "bench/ls20" and game["turns"]
    site = json.loads((out / "data" / "site.json").read_text())
    (row,) = site["problems"][0]["benchmarks"][0]["rows"]
    assert row["runs"] == 1 and site["problems"][0]["benchmarks"][0]["under"] == "bench"
    assert (out / "static" / "app.js").is_file()
    assert "site_tokens.css" not in index  # the host site's design is opt-in
    assert main([str(public), str(out), "--tokens", "../assets/tokens.css"]) == 0
    sheets = (out / "index.html").read_text()
    assert sheets.index("static/style.css") < sheets.index("../assets/tokens.css")
    published = "".join(p.read_text() for p in (out / "data").glob("*.json"))
    assert str(tmp_path) not in published  # no absolute path of this machine


def test_the_export_stops_before_publishing_a_credential(tmp_path: Path, monkeypatch) -> None:
    public, out = _curated(tmp_path), tmp_path / "site"
    secret = "s3cr3t-value-of-a-token"
    monkeypatch.setenv("SOME_SERVICE_TOKEN", secret)
    transcript = tmp_path / "private" / "exp" / "ls20" / "logs" / "transcript.jsonl"
    leak = [
        {"type": "ToolCall", "id": "t9", "name": "Bash", "input": {"command": "env"}},
        {"type": "ToolResult", "id": "t9", "output": f"SOME_SERVICE_TOKEN={secret}"},
    ]
    transcript.write_text(transcript.read_text() + "".join("\n" + json.dumps(e) for e in leak))
    assert main([str(public), str(out)]) == 1
    assert not out.exists()


def test_a_frame_pack_stores_each_image_once_in_chunks(tmp_path: Path, monkeypatch) -> None:
    from regact.viz import export

    monkeypatch.setattr(export, "_PACK_CHUNK_BYTES", 10)
    exporter = export.Exporter(_curated(tmp_path), tmp_path / "site")
    images = ["data:a" + "x" * 8, "data:b" + "y" * 8]
    urls = {"/frame?obs=1&view=agent": 0, "/frame?obs=1&view=viewer": 0, "/f2": 1}
    exporter.write_pack("g", urls, images)
    data = tmp_path / "site" / "data"
    index = json.loads((data / (request_key("pack:g") + ".json")).read_text())
    assert index["chunks"] == 2 and index["urls"]["/frame?obs=1&view=viewer"] == [0, 0]
    chunk, i = index["urls"]["/f2"]
    chunk_file = data / (request_key(f"pack:g:{chunk}") + ".json")
    assert json.loads(chunk_file.read_text())[i] == images[1]


def test_a_very_long_replay_keeps_every_nth_frame_and_every_mispredicted_one(
    tmp_path: Path, monkeypatch
) -> None:
    from regact.viz import export

    monkeypatch.setattr(export, "_MAX_REPLAY_FRAMES", 4)
    exporter = export.Exporter(_curated(tmp_path), tmp_path / "site")
    frames = [{"obs": i, **({"diagnostic": 7} if i == 5 else {})} for i in range(10)]
    timeline = {"frames": frames, "markers": [], "items": [], "axes": [], "views": []}

    class Response:
        def json(self):
            return timeline

    monkeypatch.setattr(exporter, "get", lambda url: Response())
    kept = exporter.save_timeline("bench/ls20")
    assert [f["obs"] for f in kept["frames"]] == [0, 3, 5, 6, 9] and kept["thinned"] == 3


def test_runs_per_task_publishes_the_first_finished_run_of_each_task(tmp_path: Path) -> None:
    from regact.viz.export import select_runs

    root = tmp_path / "public"
    for stamp, exit_reason in (("2026-10-01_00-00-00", None), ("2026-10-02_00-00-00", "solved"),
                               ("2026-10-03_00-00-00", "solved")):
        for task in ("ls20", "ft09"):
            logs = root / "bench" / "cwm" / stamp / task / "logs"
            logs.mkdir(parents=True)
            (logs / "experiment_state.json").write_text(json.dumps({"exit_reason": exit_reason}))
    assert select_runs(root, 1) == {
        "bench/cwm/2026-10-02_00-00-00/ls20",
        "bench/cwm/2026-10-02_00-00-00/ft09",
    }  # the unfinished first launch is skipped, the later one is not published
