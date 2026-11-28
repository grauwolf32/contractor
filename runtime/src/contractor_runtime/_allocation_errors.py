"""Safe shared errors for allocation clients and transport adapters."""

import grpc


class ArtifactClientError(Exception):
    """Bounded error safe to record in allocation metrics."""


class ArtifactTransportError(ArtifactClientError):
    pass


class ArtifactResponseLimitError(ArtifactTransportError):
    """The response exceeded the caller's byte limit."""


class ArtifactAPIError(ArtifactClientError):
    def __init__(
        self,
        status_code: grpc.StatusCode,
        code: str,
        retryable: bool,
        request_id: str | None = None,
    ) -> None:
        if not isinstance(status_code, grpc.StatusCode):
            raise TypeError("allocation errors require a native gRPC status")
        super().__init__(f"private allocation API rejected request ({code})")
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.request_id = request_id
        self.retry_after_seconds: float | None = None
        self.precondition_conflict = status_code == grpc.StatusCode.FAILED_PRECONDITION
        self.server_failure = status_code in {
            grpc.StatusCode.UNKNOWN,
            grpc.StatusCode.INTERNAL,
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DATA_LOSS,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.CANCELLED,
        }
        self.transient_failure = retryable and (
            self.server_failure
            or status_code in {grpc.StatusCode.ABORTED, grpc.StatusCode.RESOURCE_EXHAUSTED}
        )
