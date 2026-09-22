"""Opt-in diagnostics for invalid model tool arguments."""


class ToolInputError(ValueError):
    """Subclasses must keep exception messages limited to Runtime-authored text.

    Model-facing repair details belong in the structured response, not in the
    diagnostic message. Metrics apply their normal bounds and secret redaction.
    """

    code = "tool_input_invalid"
    retryable = False

    def __init__(self, *args: object, code: str | None = None) -> None:
        super().__init__(*args)
        if code is not None:
            self.code = code

    @property
    def diagnostic_message(self) -> str:
        return str(self)
