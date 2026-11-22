"""Model representations selected by http_request, before session mutation."""

from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.toolsets.common.argument_diagnostics import NormalizationNotices
from contractor_runtime.toolsets.common.arguments import (
    MAX_CONTAINER_DEPTH,
    MAX_CONTAINER_NODES,
    container_argument,
)
from contractor_runtime.toolsets.http.errors import HTTPArgumentError


def request_containers(headers, query, body_type, body):
    notices = NormalizationNotices()

    def object_argument(field, value):
        result, normalized = container_argument(
            value,
            field=field,
            kinds=(dict,),
            max_bytes=get_shared_config().http_request_body_max_bytes,
            max_depth=MAX_CONTAINER_DEPTH,
            max_nodes=MAX_CONTAINER_NODES,
            invalid=HTTPArgumentError,
        )
        notices.record(field, normalized, "object")
        return result

    headers = object_argument("headers", headers) if headers is not None else None
    query = object_argument("query", query) if query is not None else None
    if body_type == "form":
        body = object_argument("body", body)
    return headers, query, body, notices
