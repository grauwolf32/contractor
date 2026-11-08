"""Explicit tool-result diagnostics and fatal execution signals.

Only Runtime-authored text/details belong here; provider exception text must
never be used to construct a model diagnostic.
"""


class ToolResultDiagnostic:
    """Opt-in mixin for an exception with Runtime-authored model diagnostics."""

    diagnostic_message: str
    model_details: dict


class ToolExecutionError(RuntimeError):
    """Unwind the Worker instead of returning an ordinary model tool result."""

    retryable = False

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)
