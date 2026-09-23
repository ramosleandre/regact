"""Protocol selection, independent of the runner and of the optional feature registry."""

from collections.abc import Callable

from regact.config.schema import RunConfig
from regact.protocols.base import ExperimentProtocol

_REGISTRY: dict[str, Callable[[RunConfig], ExperimentProtocol]] = {}


def register_protocol(name: str, factory: Callable[[RunConfig], ExperimentProtocol]) -> None:
    """Register a task-local protocol factory; accidental replacement is an error."""
    if name in ("policy_search", "cwm") or name in _REGISTRY:
        raise ValueError(f"protocol {name!r} is already registered")
    _REGISTRY[name] = factory


def build_protocol(config: RunConfig) -> ExperimentProtocol:
    """Resolve the selected workflow. Factory exceptions retain their original meaning."""
    from regact.protocols.cwm.protocol import CwmProtocol
    from regact.protocols.policy_search import PolicySearchProtocol

    factories = {"policy_search": PolicySearchProtocol, "cwm": CwmProtocol, **_REGISTRY}
    if config.protocol.name not in factories:
        raise ValueError(
            f"unknown experiment protocol {config.protocol.name!r}; available: {sorted(factories)}"
        )
    return factories[config.protocol.name](config)
