"""Public errors shared by the environment server and its sandboxed clients."""


class InvalidActionError(ValueError):
    """The controller supplied an action that violates the problem contract."""
