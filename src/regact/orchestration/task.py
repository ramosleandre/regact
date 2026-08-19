"""Run one task end-to-end: the session builder.

Ties every layer together for a single game: build the env session + server,
front it over the right transport, bootstrap the agent's workdir, wire the
feature tools/hooks with a :class:`RunDeps`, build the prompt, drive the
keep-alive loop, and write the canonical artifacts under ``output_dir``.

The function stays short; each responsibility is a small helper. The entry
points (Block 9.3) build a problem from config and call this per task via the
Scheduler.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from regact.agent.base import CodeAgent, build_agent
from regact.agent.capabilities import uses_control_cli
from regact.config.schema import (
    AgentName,
    Lifecycle,
    RunConfig,
    redacted_config_dict,
)
from regact.env.lifecycle import EnvLifecyclePolicy, MultiInstancePolicy, SingleInstancePolicy
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.features.base import Feature, FeatureContext, RunDeps, build_features
from regact.features.controller import Controller
from regact.obs.errors import LogComponent
from regact.obs.logger import RunLogger
from regact.obs.transcript import TranscriptWriter
from regact.orchestration.env_transport import EnvConnection, serve_env
from regact.orchestration.loop import run_session
from regact.orchestration.signals import StopSignal
from regact.problems.base import BaseProblem
from regact.prompt.builder import PromptBuilder
from regact.security.egress_proxy import EgressProxy
from regact.security.netbridge import LoopbackMirror
from regact.security.runtime import SandboxRuntime, Wrapper, make_wrapper, resolve
from regact.session.state import ExperimentState
from regact.tools.base import LoggingTool, Tool
from regact.workspace.bootstrap import Workspace


def _warmup_problem(problem: BaseProblem) -> None:
    """Import the game lib off the agent's critical path (best-effort, runs in a bg thread)."""
    try:
        problem.warmup()
    except Exception as exc:  # best-effort; the real error, if any, surfaces at make_env
        logging.getLogger(__name__).debug("problem warmup failed: %s", exc)


def _regact_src_dir() -> str:
    """Absolute path of the dir containing the ``regact`` package (for subprocess imports)."""
    import regact

    return os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))


def _secret_module_paths(modules: tuple[str, ...]) -> list[str]:
    """On-disk dirs of the game packages (engine/data lib), to hide from the sandbox.

    Resolved via ``find_spec`` (no import) so a package the host lacks is just skipped.
    """
    import importlib.util

    paths: list[str] = []
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            paths.extend(spec.submodule_search_locations)
        elif spec.origin and spec.origin not in ("built-in", "frozen"):
            paths.append(os.path.dirname(spec.origin))
    return [os.path.realpath(p) for p in paths]


def _loopback_port(url: str | None) -> int | None:
    """The TCP port of ``url`` when it targets the local host, else ``None``."""
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _mirror_sockets(mirror: LoopbackMirror | None, ports: Sequence[int]) -> list[str]:
    """The socket files a wrap must expose for ``ports`` (empty without a mirror)."""
    if mirror is None:
        return []
    return [mirror.socket_path(port) for port in ports]


def _bridged(wrapper: Wrapper, mirror: LoopbackMirror | None, ports: Sequence[int]) -> Wrapper:
    """Compose ``wrapper`` with the in-sandbox bridge launcher for ``ports``."""
    if mirror is None or not ports:
        return wrapper
    prefix = mirror.argv_prefix(ports)
    return lambda argv: wrapper([*prefix, *argv])


def _requested_runtime(config: RunConfig) -> SandboxRuntime:
    """The runtime to hand the security layer: auto-detect when confined, unless
    ``sandbox_opts.backend`` forces one; ``NONE`` when unconfined."""
    if not config.sandbox:
        return SandboxRuntime.NONE
    backend = config.sandbox_opts.get("backend")
    return SandboxRuntime(backend) if backend else SandboxRuntime.AUTO


def _network_isolation(config: RunConfig) -> bool:
    """Whether to isolate the agent's network: a fresh net namespace (deny_egress ->
    ``--unshare-net``) + the loopback netbridge + the egress proxy. On by default under sandbox.

    ``sandbox_opts.network_isolation=false`` turns it OFF: the FILESYSTEM anti-cheat
    (``deny_read`` on the game engine + ``regact.problems``) stays, but the agent shares the host
    net namespace and reaches the env server on ``127.0.0.1`` directly (no bridge). SAFE ONLY
    where egress is impossible or acceptable (e.g. an offline HPC compute node) - it removes the
    egress block, so on a routable host the agent could exfiltrate or fetch answers.
    """
    return config.sandbox and bool(config.sandbox_opts.get("network_isolation", True))


def _collect_feature_metrics(
    features: list[Feature], deps: RunDeps, logger: RunLogger
) -> dict[str, Any]:
    """Every loaded feature's own submission numbers, keyed by feature name.

    Empty contributions are dropped so a submission only carries features that
    actually scored something. A faulty contributor is logged and skipped — extra
    metrics must never break the submission that carries them.
    """
    collected: dict[str, Any] = {}
    for feature in features:
        try:
            metrics = feature.submission_metrics(deps)
        except Exception as exc:
            logger.log(
                LogComponent.EVAL,
                "WARNING",
                "feature_metrics_failed",
                feature=feature.name,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        if metrics:
            collected[feature.name] = metrics
    return collected


def _lifecycle_policy(lifecycle: Lifecycle) -> EnvLifecyclePolicy:
    if lifecycle is Lifecycle.SINGLE_INSTANCE:
        return SingleInstancePolicy()
    return MultiInstancePolicy()


def _build_server(
    config: RunConfig,
    problem: BaseProblem,
    task_name: str,
    *,
    features: list[Feature],
    workdir: str,
    output_dir: str,
) -> EnvServer:
    """Register the task's :class:`EnvSession` (renderer + lifecycle + milestones +
    the loaded features' env wrappers, in ``features:`` list order)."""
    ctx = FeatureContext(
        problem_name=problem.name, task_name=task_name, workdir=workdir, output_dir=output_dir
    )
    wrappers = [wrap for feature in features if (wrap := feature.env_wrapper(ctx)) is not None]
    session = EnvSession(
        make_native=lambda: problem.make_env(task_name),
        key=task_name,
        renderer=problem.obs_renderer(task_name, mode=config.problem.obs_mode),
        lifecycle=_lifecycle_policy(config.problem.lifecycle),
        milestone_detector=problem.milestone_detector(task_name),
        step_budget=config.limits.max_actions_per_env,
        wrappers=wrappers,
    )
    server = EnvServer()
    server.register(task_name, session)
    return server


def _bootstrap_workdir(
    config: RunConfig,
    problem: BaseProblem,
    task_name: str,
    *,
    workdir: str,
    conn: EnvConnection,
    controller: Controller,
    features: list[Feature],
) -> None:
    Workspace(workdir).bootstrap(
        features,
        controller=controller,
        problem_name=problem.name,
        task_name=task_name,
        env_base_url=conn.base_url,
        game_id=task_name,
        lifecycle=config.problem.lifecycle,
        helper_templates=problem.helper_templates(task_name, info_mode=config.problem.info_mode),
    )


async def run_task(
    config: RunConfig,
    problem: BaseProblem,
    task_name: str,
    *,
    output_dir: str,
    agent: CodeAgent | None = None,
    stop: StopSignal | None = None,
) -> str:
    """Drive ``task_name`` to completion; return the loop's exit reason.

    ``agent`` is injectable (tests pass a scripted agent); by default it is built
    from ``config.agent``.
    """
    requested_runtime = _requested_runtime(config)
    if config.sandbox and resolve(requested_runtime) is SandboxRuntime.NONE:
        raise RuntimeError(
            "sandbox=true but no sandbox backend is usable on this host; "
            "enable one (bwrap on Linux, seatbelt on macOS) or set sandbox=false"
        )
    # Warm the game library NOW, in the background, so its heavy first import (gym/minigrid
    # over Lustre on a shared HPC node) overlaps agent boot instead of blocking the agent's
    # first make_env into an EnvClient ReadTimeout. Module imports are process-global, so the
    # env server thread finds it cached.
    threading.Thread(
        target=_warmup_problem, args=(problem,), daemon=True, name="env-warmup"
    ).start()

    controller = Controller.from_config(config.controller)
    features = build_features(config.features)
    if config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE and (
        controller.evaluates_on_env or any(feature.evaluates_on_env for feature in features)
    ):
        raise RuntimeError(
            "single-instance problem with an on-env evaluation (the always-on controller scores "
            "by rolling episodes; an evaluating feature may too): exploration and evaluation share "
            "the same env, so scores would reflect the session, not an isolated policy"
        )
    workdir = os.path.join(output_dir, "workdir")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(redacted_config_dict(config), handle, indent=2, default=str)

    server = _build_server(
        config, problem, task_name, features=features, workdir=workdir, output_dir=output_dir
    )
    in_process = config.agent.name is AgentName.SCRIPTED

    async with serve_env(server, task_name, in_process=in_process) as conn:
        with (
            TranscriptWriter(os.path.join(logs_dir, "transcript.jsonl")) as transcript,
            RunLogger(logs_dir, task=task_name) as logger,
        ):
            _bootstrap_workdir(
                config,
                problem,
                task_name,
                workdir=workdir,
                conn=conn,
                controller=controller,
                features=features,
            )

            experiment = ExperimentState(
                problem_name=problem.name,
                task_name=task_name,
                problem_kwargs=dict(config.problem.kwargs),
            )
            src_dir = _regact_src_dir()
            deny_read = _secret_module_paths(problem.secret_modules())
            # Hide regact's OWN problem wrappers from the sandbox. The agent needs
            # ``regact.envclient``/``regact.controllers`` to run, but never ``regact.problems``
            # (the game-side logic + obs format); reading it leaks the game. src_dir is on
            # allow_read so the rest of regact stays importable; this carves just the games out.
            problems_dir = os.path.realpath(os.path.join(src_dir, "regact", "problems"))
            if os.path.isdir(problems_dir):
                deny_read = [*deny_read, problems_dir]

            # Network isolation (a fresh net namespace + the loopback netbridge + the egress
            # proxy) is on by default under sandbox. ``sandbox_opts.network_isolation=false``
            # keeps the FILESYSTEM anti-cheat (deny_read: minigrid/problems) but drops the
            # network namespace, so the agent reaches the env server on host 127.0.0.1 directly
            # (no netbridge). SAFE ONLY where egress is impossible/acceptable (an offline HPC
            # node): it removes the egress block, so a routable node could exfiltrate/fetch.
            net_isolation = _network_isolation(config)

            mirror: LoopbackMirror | None = None
            env_port = _loopback_port(conn.base_url)
            if (
                net_isolation
                and not in_process
                and resolve(requested_runtime) is SandboxRuntime.BWRAP
            ):
                mirror = LoopbackMirror()

            eval_ports = [port for port in (env_port,) if port]
            if mirror is not None:
                for port in eval_ports:
                    await mirror.mirror(port)
            eval_wrap = (
                None
                if in_process
                else _bridged(
                    make_wrapper(
                        requested_runtime,
                        workdir=workdir,
                        allow_read=[src_dir],
                        deny_egress=net_isolation,
                        deny_read=deny_read,
                        allow_rw=_mirror_sockets(mirror, eval_ports),
                    ),
                    mirror,
                    eval_ports,
                )
            )
            deps = RunDeps(
                experiment=experiment,
                env_client=conn.client,
                lifecycle=config.problem.lifecycle,
                solution_path=os.path.join(workdir, "solution.py"),
                submissions_dir=os.path.join(workdir, "submissions"),
                compute_episode_metrics=problem.compute_episode_metrics,
                aggregate_episode_metrics=problem.aggregate_episode_metrics,
                sandbox_wrap=eval_wrap,
                render_frame=problem.render_frame,
                seed=config.problem.seed,
            )
            deps.feature_metrics = lambda: _collect_feature_metrics(features, deps, logger)
            # Core controller first, then each optional feature: one flat tool/hook surface.
            tool_specs = [*controller.tools(deps), *(t for f in features for t in f.tools(deps))]
            tools: list[Tool] = [LoggingTool(tool, logger) for tool in tool_specs]
            hooks = [*controller.hooks(deps), *(h for f in features for h in f.hooks(deps))]

            agent = agent or build_agent(config.agent)
            caps = agent.capabilities()
            # Every non-native protocol reaches the framework tools over the workdir control CLI, so
            # the channel MUST be bound for it (uses_control_cli is the shared predicate the prompt
            # builder keys off too - see its docstring). Only the in-process "native" backend takes
            # the tools as loop tools directly.
            if uses_control_cli(caps.tool_protocol):
                server.bind_control(task_name, tools, cwd=workdir)
                loop_tools: list[Tool] = []
            elif caps.executes_tools:
                loop_tools = []
            else:
                loop_tools = tools

            agent_tmp = os.path.join(workdir, "tmp")
            os.makedirs(agent_tmp, exist_ok=True)
            # workdir first so ``import framework`` (and code_library) resolves no matter
            # which directory the agent runs a script from - Python only puts the script's
            # own dir on sys.path, so ``python code_library/probe.py`` otherwise can't see
            # the root-level framework/ package.
            agent_env = {"PYTHONPATH": os.pathsep.join([workdir, src_dir]), "TMPDIR": agent_tmp}
            egress: EgressProxy | None = None
            egress_hosts = agent.host_egress_hosts()
            if net_isolation and egress_hosts:
                egress = EgressProxy(egress_hosts)
                proxy_url = f"http://127.0.0.1:{await egress.start()}"
                agent_env |= {
                    "HTTPS_PROXY": proxy_url,
                    "HTTP_PROXY": proxy_url,
                    "NO_PROXY": "127.0.0.1,localhost",
                }

            agent_ports: list[int] = []
            if mirror is not None:
                candidates = (
                    env_port,
                    egress.port if egress is not None else None,
                    _loopback_port(config.agent.base_url),
                )
                agent_ports = list(dict.fromkeys(port for port in candidates if port))
                for port in agent_ports:
                    await mirror.mirror(port)
            runtime_wrap = _bridged(
                make_wrapper(
                    requested_runtime,
                    workdir=workdir,
                    allow_read=[src_dir, *agent.host_read_paths()],
                    deny_egress=net_isolation,
                    deny_read=deny_read,
                    allow_write_prefixes=agent.host_write_prefixes(),
                    allow_rw=[*_mirror_sockets(mirror, agent_ports), *agent.host_rw_paths()],
                ),
                mirror,
                agent_ports,
            )

            builder = PromptBuilder()
            system_prompt = builder.build_system_prompt(
                problem,
                task_name,
                features,
                controller=controller,
                lifecycle=config.problem.lifecycle,
                info_mode=config.problem.info_mode,
                tool_protocol=caps.tool_protocol,
                tool_names=[tool.name for tool in tools],
                # Opt-in tier-2 verbalization hint (empty_response A/B); bench sets the env.
                verbalize_state=os.environ.get("REGACT_VERBALIZE_STATE", "").strip()
                not in ("", "0", "false", "False"),
            )
            try:
                await agent.start(
                    cwd=workdir,
                    model=config.agent.model,
                    base_url=config.agent.base_url,
                    api_key=config.agent.api_key,
                    system_prompt=system_prompt,
                    tools=tools,
                    env=agent_env,
                    runtime_wrap=runtime_wrap,
                )
                rendered_first_obs = None
                if config.first_obs_in_prompt:
                    rendered_first_obs = problem.render_obs_text(server.first_obs(task_name))
                first_message = builder.build_first_message(rendered_first_obs)

                if config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE:
                    logger.log(
                        LogComponent.ORCHESTRATOR,
                        "WARNING",
                        "single_instance_shared_env",
                        message=(
                            "single-instance: exploration and evaluation share the same env; "
                            "scores reflect the session, not an isolated policy"
                        ),
                    )
                reason = await run_session(
                    agent,
                    experiment=experiment,
                    first_message=first_message,
                    tools=loop_tools,
                    transcript=transcript,
                    logger=logger,
                    limits=config.limits,
                    state_path=os.path.join(logs_dir, "experiment_state.json"),
                    cwd=workdir,
                    system_prompt=system_prompt,
                    hooks=hooks,
                    move_count=lambda: server.total_action_count(task_name),
                    stop=stop,
                )
            finally:  # always release the agent subprocess + network plumbing, even on a crash
                await agent.close()
                if egress is not None:
                    await egress.close()
                if mirror is not None:
                    await mirror.close()
            return reason
