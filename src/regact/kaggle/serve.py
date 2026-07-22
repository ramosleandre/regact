"""Boot a local vLLM server in a Kaggle kernel and ask an Alan agent questions.

On Kaggle there is no internet at scoring time, so the model is served *inside*
the kernel by vLLM (an OpenAI-compatible HTTP endpoint) and the in-process Alan
agent talks to it over ``http://127.0.0.1:<port>/v1`` — the same wiring that runs
on Adastra against SimpleLM, only the server changes.

Design choices (audited against the ARC-Prize-2026 Kaggle constraints):

- **Serve, don't embed.** We run ``vllm serve`` as a subprocess and poll
  ``/health`` rather than the in-process ``vllm.LLM(...)`` API — the embedded
  API couples the agent to vLLM internals and is the pattern that has OOM-crashed
  on Kaggle. A server is also what lets several game workers share one model.
- **Single GPU.** The competition accelerator is a single RTX 6000, so we never
  set tensor parallelism > 1.
- **Offline-safe.** ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` are forced and the
  model is addressed by a local path — a stray hub fetch would hang forever in an
  air-gapped kernel.
- **One-line model switch.** :class:`VLLMConfig.model_path` (+ ``served_name``) is
  the only thing to change to swap models; the safe serving flags travel with it.

torch / vllm are imported lazily, so this module imports on a laptop with no GPU.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8000


@dataclass
class VLLMConfig:
    """How to serve one model. Change ``model_path``/``served_name`` to switch models.

    Defaults are tuned for a single ~48GB RTX 6000 serving a quantized model with
    room for a long KV cache. ``extra_args`` is an escape hatch for per-model flags.
    """

    # The ONE-LINE model switch: a LOCAL directory (a Kaggle Model mount), never a
    # hub repo id (that needs internet). e.g. /kaggle/input/qwen-3/transformers/14b-awq/1
    model_path: str
    served_name: str = "local-model"  # the name Alan/openai addresses (openai/<served_name>)

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT

    # Quantization: None lets vLLM read it from the checkpoint (an FP8 model needs no
    # flag). Set "awq_marlin" for an AWQ model (the fast kernel; plain "awq" is ~10x
    # slower), or "fp8" to force fp8. Default None so an FP8 checkpoint just works.
    quantization: str | None = None
    dtype: str = "auto"  # let the checkpoint decide (bf16/fp8 on Blackwell, fp16 on older cards)
    gpu_memory_utilization: float = 0.90  # leave headroom for driver/context; back off if OOM
    max_model_len: int = 32768  # dominant KV-cache/OOM lever — drop to 16384 if tight
    max_num_seqs: int = 16  # concurrent sequences; bound memory under parallel workers
    kv_cache_dtype: str | None = None  # "fp8" halves KV memory; None = same as weights

    # Tool calling — Alan uses native tool calls (auto choice). The parser must match
    # the model family: "qwen3_coder" for Qwen3, "hermes" for many other open models.
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str | None = "qwen3"  # persist thinking traces across turns
    enable_prefix_caching: bool = True  # free throughput win for the shared system-prompt prefix
    # Chat-template kwargs (e.g. {"preserve_thinking": true} for Qwen3) as a JSON string.
    chat_template_kwargs: str | None = '{"preserve_thinking": true}'

    enforce_eager: bool = False  # set True only if CUDA-graph capture OOMs / to boot faster
    # Skip FlashInfer's autotune warmup — it segfaults in cuDNN on sm_120 FP8 (paired with
    # VLLM_DISABLED_KERNELS in _offline_env). Harmless elsewhere (heuristic tactics instead).
    disable_flashinfer_autotune: bool = True
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def to_argv(self) -> list[str]:
        """The ``vllm serve`` command line for this config."""
        argv = [
            "vllm",
            "serve",
            self.model_path,
            "--served-model-name",
            self.served_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
        ]
        if self.quantization:
            argv += ["--quantization", self.quantization]
        if self.kv_cache_dtype:
            argv += ["--kv-cache-dtype", self.kv_cache_dtype]
        if self.enable_auto_tool_choice:
            argv += ["--enable-auto-tool-choice", "--tool-call-parser", self.tool_call_parser]
        if self.reasoning_parser:
            argv += ["--reasoning-parser", self.reasoning_parser]
        if self.enable_prefix_caching:
            argv += ["--enable-prefix-caching"]
        if self.chat_template_kwargs:
            argv += ["--default-chat-template-kwargs", self.chat_template_kwargs]
        if self.enforce_eager:
            argv += ["--enforce-eager"]
        if self.disable_flashinfer_autotune:
            argv += ["--no-enable-flashinfer-autotune"]
        return argv + list(self.extra_args)


def probe_gpu(*, expect_name: str | None = None) -> dict[str, object]:
    """Report the GPU so the model/quantization choice can be confirmed at runtime.

    The competition card is a single RTX Pro 6000 (Blackwell, sm_120); its arch decides
    which quantizations + vLLM/flashinfer wheel work. Print this in the kernel BEFORE
    serving. Pass ``expect_name`` (e.g. "RTX Pro 6000") to ASSERT the card matches — a
    wrong card is a hard boot failure worth catching early. Returns an empty report (and
    logs) when torch/CUDA is unavailable — e.g. on a laptop.
    """
    report: dict[str, object] = {"available": False}
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("torch not importable — no GPU probe (expected off-Kaggle)")
        return report
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — no GPU probe")
        return report
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    report = {
        "available": True,
        "name": name,
        "capability": f"{cap[0]}.{cap[1]}",
        "capability_tuple": cap,
        "device_count": torch.cuda.device_count(),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    }
    logger.info("GPU: %s", report)
    if expect_name and expect_name.lower() not in str(name).lower():
        raise RuntimeError(
            f"GPU mismatch: expected a card matching {expect_name!r} but found {name!r}. "
            "The served model + vLLM wheel are arch-specific; aborting before a wrong-arch boot."
        )
    return report


class VLLMServer:
    """A running ``vllm serve`` subprocess. Use as a context manager or call ``close()``."""

    def __init__(self, config: VLLMConfig, process: subprocess.Popen[bytes]) -> None:
        self.config = config
        self.process = process

    @property
    def base_url(self) -> str:
        return self.config.base_url

    def close(self) -> None:
        """Terminate the server (SIGTERM, then SIGKILL)."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

    def __enter__(self) -> VLLMServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Where libcuda.so lives on the Kaggle GPU image. It is NOT on the default linker path,
# so flashinfer's just-in-time sm_120 kernel build fails with `ld: cannot find -lcuda`
# unless we prepend it (the fix the Milestone-1 winners use). The stubs dir is a fallback.
_CUDA_LIB_DIRS = ("/usr/local/nvidia/lib64", "/usr/local/cuda/lib64/stubs")


def _prepend_paths(env: dict[str, str], key: str, dirs: tuple[str, ...]) -> None:
    existing = [p for p in env.get(key, "").split(os.pathsep) if p]
    present = [d for d in dirs if os.path.isdir(d)]
    env[key] = os.pathsep.join([*present, *existing])


def _offline_env() -> dict[str, str]:
    """Env for the vLLM subprocess: no network reach + libcuda on the linker path.

    Prepends the CUDA lib dirs to LIBRARY_PATH (the compile-time linker path flashinfer's
    ninja build uses) and LD_LIBRARY_PATH (runtime), so JIT sm_120 kernel builds can link
    ``-lcuda`` on the Kaggle GPU image.
    """
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("VLLM_DO_NOT_TRACK", "1")
    _prepend_paths(env, "LIBRARY_PATH", _CUDA_LIB_DIRS)
    _prepend_paths(env, "LD_LIBRARY_PATH", _CUDA_LIB_DIRS)
    # Blackwell (sm_120) FP8: the FlashInfer scaled-mm kernel's autotuner probes cuDNN
    # tactics that have no valid plan on sm_120 and SEGFAULTS at startup. Disable that
    # kernel so FP8 falls back to CUTLASS / torch._scaled_mm (same footprint, still fits).
    # This is NVIDIA's own Blackwell-FP8 recipe (ai-dynamo). Paired with the serve flag
    # --no-enable-flashinfer-autotune below. Append (don't clobber a user-set value).
    _disabled = [k for k in env.get("VLLM_DISABLED_KERNELS", "").split(",") if k]
    if "FlashInferFP8ScaledMMLinearKernel" not in _disabled:
        _disabled.append("FlashInferFP8ScaledMMLinearKernel")
    env["VLLM_DISABLED_KERNELS"] = ",".join(_disabled)
    return env


def serve_vllm(
    config: VLLMConfig,
    *,
    ready_timeout_s: float = 1800.0,
    poll_interval_s: float = 5.0,
    log_path: str | None = None,
) -> VLLMServer:
    """Start ``vllm serve`` and block until ``/health`` is 200 (or it dies/times out).

    Args:
        config: what + how to serve.
        ready_timeout_s: max wait for readiness (big quantized models load slowly).
        poll_interval_s: health-poll cadence.
        log_path: where to tee the server's stdout/stderr; defaults to
            ``/kaggle/working/vllm.log`` on Kaggle else ``./vllm.log``.

    Returns:
        A live :class:`VLLMServer`. Raises ``RuntimeError`` if the server dies or
        never becomes healthy.
    """
    argv = config.to_argv()
    if log_path is None:
        base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        log_path = os.path.join(base, "vllm.log")
    logger.info("[vllm] serving %s -> %s (log: %s)", config.model_path, config.base_url, log_path)
    logger.info("[vllm] %s", " ".join(argv))

    log_file = open(log_path, "wb")  # noqa: SIM115 — handed to the long-lived subprocess
    process: subprocess.Popen[bytes] = subprocess.Popen(
        argv, stdout=log_file, stderr=subprocess.STDOUT, env=_offline_env()
    )
    server = VLLMServer(config, process)

    health = f"http://{config.host}:{config.port}/health"
    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            server.close()
            raise RuntimeError(
                f"vllm exited with code {process.returncode} before readiness; see {log_path}"
            )
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:  # fixed localhost url
                if resp.status == 200:
                    logger.info("[vllm] healthy at %s", config.base_url)
                    return server
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval_s)

    server.close()
    raise RuntimeError(f"vllm not healthy within {ready_timeout_s}s; see {log_path}")


async def ask(
    prompt: str,
    *,
    base_url: str,
    served_name: str,
    api_key: str = "local",
    system_prompt: str | None = None,
    cwd: str | None = None,
    args: dict[str, object] | None = None,
) -> str:
    """Ask one question to the served model through an in-process Alan agent.

    A thin convenience over :class:`alancode.AlanCodeAgent` for notebook probing —
    "is the server answering, and through Alan?" Returns Alan's final text. For the
    real game loop the framework drives Alan via the normal experiment runner; this
    is the minimal "ask the server a question" primitive.
    """
    from alancode import AlanCodeAgent

    defaults: dict[str, object] = {"permission_mode": "yolo", "memory": "off"}
    defaults.update(args or {})
    agent = AlanCodeAgent(
        model=f"openai/{served_name}",
        base_url=base_url,
        api_key=api_key,
        cwd=cwd or os.getcwd(),
        programmatic=True,
        custom_system_prompt=system_prompt,
        **defaults,
    )
    try:
        answer: str = await agent.query_async(prompt)
        return answer
    finally:
        await agent.close()
