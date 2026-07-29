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
import os
from collections.abc import Sequence
from urllib.parse import urlparse

from regact.agent.base import CodeAgent, build_agent
from regact.config.schema import AgentName, Lifecycle, RunConfig, redacted_config_dict
from regact.env.lifecycle import EnvLifecyclePolicy, MultiInstancePolicy, SingleInstancePolicy
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.features.base import Feature, RunDeps, build_features
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
from regact.tools.base import Tool
from regact.workspace.bootstrap import Workspace


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


def _lifecycle_policy(lifecycle: Lifecycle) -> EnvLifecyclePolicy:
    if lifecycle is Lifecycle.SINGLE_INSTANCE:
        return SingleInstancePolicy()
    return MultiInstancePolicy()


def _build_server(config: RunConfig, problem: BaseProblem, task_name: str) -> EnvServer:
    """Register the task's :class:`EnvSession` (renderer + lifecycle + milestones)."""
    session = EnvSession(
        make_native=lambda: problem.make_env(task_name),
        key=task_name,
        renderer=problem.obs_renderer(task_name, mode=config.problem.obs_mode),
        lifecycle=_lifecycle_policy(config.problem.lifecycle),
        milestone_detector=problem.milestone_detector(task_name),
        step_budget=config.limits.env_step_budget,
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
    features: list[Feature],
) -> None:
    Workspace(workdir).bootstrap(
        features,
        problem_name=problem.name,
        task_name=task_name,
        env_base_url=conn.base_url,
        game_id=task_name,
        lifecycle=config.problem.lifecycle,
        helper_templates=problem.helper_templates(task_name),
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
    workdir = os.path.join(output_dir, "workdir")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(redacted_config_dict(config), handle, indent=2, default=str)

    server = _build_server(config, problem, task_name)
    in_process = config.agent.name is AgentName.SCRIPTED

    async with serve_env(server, task_name, in_process=in_process) as conn:
        features = build_features(config.features)
        _bootstrap_workdir(
            config, problem, task_name, workdir=workdir, conn=conn, features=features
        )

        experiment = ExperimentState(
            problem_name=problem.name, task_name=task_name, n_eval_episodes=1, n_videos=0
        )
        src_dir = _regact_src_dir()
        deny_read = _secret_module_paths(problem.secret_modules())

        mirror: LoopbackMirror | None = None
        env_port = _loopback_port(conn.base_url)
        if not in_process and resolve(config.security.sandbox) is SandboxRuntime.BWRAP:
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
                    config.security.sandbox,
                    workdir=workdir,
                    allow_read=[src_dir],
                    deny_egress=True,
                    deny_read=deny_read,
                    allow_rw=_mirror_sockets(mirror, eval_ports),
                    image=config.security.runtime_opts.get("image"),
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
            n_episodes=config.limits.n_episodes,
            max_moves=config.limits.max_moves,
            compute_episode_metrics=problem.compute_episode_metrics,
            aggregate_episode_metrics=problem.aggregate_episode_metrics,
            sandbox_wrap=eval_wrap,
            render_frame=problem.render_frame,
            record_video=config.record_video,
            seed=config.problem.seed,
            shadow_replay=config.shadow_replay,
        )
        tools = [tool for feature in features for tool in feature.tools(deps)]
        hooks = [hook for feature in features for hook in feature.hooks(deps)]

        agent = agent or build_agent(config.agent)
        caps = agent.capabilities()
        if caps.control_actions == "client_cli":
            server.bind_control(task_name, tools, cwd=workdir)
            loop_tools: list[Tool] = []
        elif caps.executes_tools:
            loop_tools = []
        else:
            loop_tools = tools

        agent_tmp = os.path.join(workdir, "tmp")
        os.makedirs(agent_tmp, exist_ok=True)
        agent_env = {"PYTHONPATH": src_dir, "TMPDIR": agent_tmp}
        egress: EgressProxy | None = None
        egress_hosts = agent.host_egress_hosts()
        if config.security.deny_egress and egress_hosts:
            egress = EgressProxy(egress_hosts)
            proxy_url = f"http://127.0.0.1:{await egress.start()}"
            agent_env |= {
                "HTTPS_PROXY": proxy_url,
                "HTTP_PROXY": proxy_url,
                "NO_PROXY": "127.0.0.1,localhost",
            }

        agent_ports: list[int] = []
        if mirror is not None and config.security.deny_egress:
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
                config.security.sandbox,
                workdir=workdir,
                allow_read=[src_dir, *agent.host_read_paths()],
                deny_egress=config.security.deny_egress,
                deny_read=deny_read,
                allow_write_prefixes=agent.host_write_prefixes(),
                allow_rw=[*_mirror_sockets(mirror, agent_ports), *agent.host_rw_paths()],
                image=config.security.runtime_opts.get("image"),
            ),
            mirror,
            agent_ports,
        )

        builder = PromptBuilder()
        system_prompt = builder.build_system_prompt(
            problem,
            task_name,
            features,
            lifecycle=config.problem.lifecycle,
            info_mode=config.problem.info_mode,
            control_actions=agent.capabilities().control_actions,
            tool_names=[tool.name for tool in tools],
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
            first_obs = server.first_obs(task_name)
            first_message = builder.build_first_message(problem.render_obs_text(first_obs))

            with (
                TranscriptWriter(os.path.join(logs_dir, "transcript.jsonl")) as transcript,
                RunLogger(logs_dir, task=task_name) as logger,
            ):
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
                    move_count=lambda: server.live_action_count(task_name),
                    stop=stop,
                )
        finally:  # always release the agent subprocess + the network plumbing, even on a crash
            await agent.close()
            if egress is not None:
                await egress.close()
            if mirror is not None:
                await mirror.close()
        return reason
