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

import os

from regact.agent.base import CodeAgent, build_agent
from regact.config.schema import AgentName, Lifecycle, RunConfig
from regact.env.lifecycle import EnvLifecyclePolicy, MultiInstancePolicy, SingleInstancePolicy
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.features.base import Feature, RunDeps, build_features
from regact.obs.logger import RunLogger
from regact.obs.transcript import TranscriptWriter
from regact.orchestration.env_transport import EnvConnection, serve_env
from regact.orchestration.loop import run_session
from regact.problems.base import BaseProblem
from regact.prompt.builder import PromptBuilder
from regact.security.runtime import make_wrapper
from regact.session.state import ExperimentState
from regact.tools.base import Tool
from regact.workspace.bootstrap import Workspace


def _regact_src_dir() -> str:
    """Absolute path of the dir containing the ``regact`` package (for subprocess imports)."""
    import regact

    return os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))


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
) -> str:
    """Drive ``task_name`` to completion; return the loop's exit reason.

    ``agent`` is injectable (tests pass a scripted agent); by default it is built
    from ``config.agent``.
    """
    workdir = os.path.join(output_dir, "workdir")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

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
        deps = RunDeps(
            experiment=experiment,
            env_client=conn.client,
            lifecycle=config.problem.lifecycle,
            solution_path=os.path.join(workdir, "solution.py"),
            submissions_dir=os.path.join(workdir, "submissions"),
            n_episodes=1,
            max_moves=config.limits.max_moves,
        )
        tools = [tool for feature in features for tool in feature.tools(deps)]
        hooks = [hook for feature in features for hook in feature.hooks(deps)]

        agent = agent or build_agent(config.agent)
        # Where framework tools are executed depends on the backend:
        #   native_tools (scripted/Alan): the loop executes them on ToolCall events.
        #   client_cli (Claude/codex): the control server executes them; the workdir
        #   CLI hits it, and the loop only observes the agent's stream.
        if agent.capabilities().control_actions == "client_cli":
            server.bind_control(task_name, tools, cwd=workdir)
            loop_tools: list[Tool] = []
        else:
            loop_tools = tools

        # Confine the agent's subprocess to its workdir and the regact source it imports;
        # paths outside (the rest of the repo) are absent from its filesystem view.
        # In-process agents ignore this; subprocess (CLI) agents run wrapped.
        src_dir = _regact_src_dir()
        runtime_wrap = make_wrapper(
            config.security.runtime,
            workdir=workdir,
            allow_read=[src_dir],
            forbid_read=[os.path.dirname(src_dir)],  # the repo: src + environnement + experiments
            deny_egress=config.security.deny_egress,
            image=config.security.runtime_opts.get("image"),
        )

        builder = PromptBuilder()
        await agent.start(
            cwd=workdir,
            model=config.agent.model,
            base_url=config.agent.base_url,
            api_key=config.agent.api_key,
            system_prompt=builder.build_system_prompt(),
            tools=tools,
            # The agent's subprocess scripts (cwd=workdir) must import regact; give
            # them the absolute src dir so it works whether or not regact is installed.
            env={"PYTHONPATH": src_dir},
            runtime_wrap=runtime_wrap,
        )
        first_message = builder.build_first_message(
            problem, task_name, features, info_mode=config.problem.info_mode
        )

        with (
            TranscriptWriter(os.path.join(logs_dir, "transcript.jsonl")) as transcript,
            RunLogger(logs_dir, task=task_name) as logger,
        ):
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
                hooks=hooks,
            )
        await agent.close()
        return reason
