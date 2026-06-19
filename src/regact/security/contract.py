"""The sandbox contract: the invariants the agent's sandbox must satisfy on every platform.

The contract is identical on every platform (macOS / Linux / Jean-Zay / Adastra /
Kaggle); only the enforcing mechanism differs (seatbelt / bwrap / landlock /
apptainer / intrinsic). The probe (:mod:`regact.security.probe`) checks these same
invariants on each platform, so "the same restriction everywhere" is verifiable
rather than asserted.

Pure data: no agent, environment, or feature types, so the security layer stays
agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Invariant(StrEnum):
    """The six invariants of the agent's room."""

    R1_WORKDIR = "R1"  # CAN read & write its own workdir + framework + venv (must work)
    R2_SECRET = "R2"  # CANNOT read the game files or climb out of the workdir
    R3_WRITE = "R3"  # can write ONLY its workdir (+ /tmp), not framework / venv / games
    R4_LOOPBACK = "R4"  # CAN reach localhost (the env server + a local LLM)
    R5_EGRESS = "R5"  # CANNOT reach the external internet (scored runs)
    R6_NO_ESCAPE = "R6"  # a child process inherits the restriction (no escape)


@dataclass(frozen=True)
class InvariantSpec:
    invariant: Invariant
    intent: str  # "allow" (the legitimate use must work) | "deny" (the attack must fail)
    summary: str


CONTRACT: tuple[InvariantSpec, ...] = (
    InvariantSpec(Invariant.R1_WORKDIR, "allow", "read & write its own workdir + framework + venv"),
    InvariantSpec(Invariant.R2_SECRET, "deny", "read game files or climb out of the workdir"),
    InvariantSpec(Invariant.R3_WRITE, "deny", "write outside its workdir (framework, venv, games)"),
    InvariantSpec(Invariant.R4_LOOPBACK, "allow", "reach the localhost env server + a local LLM"),
    InvariantSpec(Invariant.R5_EGRESS, "deny", "reach the external internet on a scored run"),
    InvariantSpec(Invariant.R6_NO_ESCAPE, "deny", "lift the restriction from a child process"),
)
