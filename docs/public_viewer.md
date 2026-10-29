# Public viewer

A static, server-less copy of `make viz` for a website: the same app, with every request answered by a pre-rendered file.

## Curate

`experiments_public/` (gitignored) holds what to publish:

- one entry per benchmark: a link to an experiment folder kept elsewhere is fine (`ln -s ../experiments/<experiment> experiments_public/<name>`);
- `benchmarks.yaml`, the landing page:

```yaml
title: Regact benchmarks
intro: One paragraph shown above the list.
problems:
  - name: ARC-AGI-3
    description: One line about the problem.
    benchmarks:
      - title: Claude Sonnet 5.5, first 3 levels
        path: arc-sonnet-3-levels     # an entry of experiments_public/
        note: Optional line under the title.
```

The landing page lists each benchmark as one row per (agent, model, protocol): runs, solved runs, and the mean of the problem's first main metric. Each benchmark links to its graphs and its runs. `make viz EXP=experiments_public` shows the same folder locally.

## Export

```bash
PYTHONPATH=src python -m regact.viz.export experiments_public <out_dir>
```

`<out_dir>` gets `index.html`, `static/` (the viewer) and `data/`. Each request the viewer makes is a file named by its FNV-1a hash (`request_key` in `export.py`, `requestKey` in `app.js`). A run's replay frames come as one pack. Serve the folder from any static host: it uses relative URLs only, and needs neither inline script nor inline style, so it runs under a strict Content-Security-Policy (`default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; ...`).

Not exported: interactive world-model playback, which runs the agent's code; the local viewer keeps it.

## What the export removes

- **Absolute paths:** this machine's paths (home, the curated root, the link targets) become `~`, and any other `/home/...`, `/lus/...` path becomes `~/...`.
- **Credentials:** the export stops, writing nothing, if the value of any credential-like variable in its environment, or a key-shaped string (API keys, GitHub tokens, private keys), would be published.
- **Large videos:** videos over 20 MB are left out.

Hosting limits it checks against (Cloudflare Pages): 25 MiB per file, 20,000 files.
