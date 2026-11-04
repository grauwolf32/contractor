"""Authority checked before native connection and application I/O."""

import time


class RequestBoundary:
    def __init__(
        self,
        deadline: float,
        *,
        authority_deadline,
        monotonic=time.monotonic,
        require_session=lambda: None,
    ) -> None:
        self.deadline = deadline
        self.clock = monotonic
        self._authority_deadline = authority_deadline
        self.request_started = False
        self._require_session = require_session

    def require_time(self) -> None:
        self._require_session()
        authority = self._authority_deadline()
        if authority is None or self.clock() >= min(self.deadline, authority):
            raise TimeoutError
