"""Kaggle deployment glue for regact.

The framework runs on Kaggle exactly as it runs on Adastra — an in-process Alan
agent driven by a local OpenAI-compatible LLM server — with two Kaggle-specific
needs that live here:

- :mod:`regact.kaggle.serve` boots a local **vLLM** server in the kernel (the
  cloud agent APIs are unreachable offline) and exposes a one-call ``ask`` to
  drive Alan against it. The model is switched with a single config field.

Everything heavy (torch / vllm) is imported lazily so this package imports on a
laptop without a GPU; only :func:`regact.kaggle.serve.serve_vllm` needs them.
"""

from __future__ import annotations
