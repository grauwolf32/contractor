"""Model representations selected by http_request, before session mutation."""

from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.toolsets.common.argument_diagnostics import NormalizationNotices
from contractor_runtime.toolsets.common.arguments import (
    MAX_CONTAINER_DEPTH,
    MAX_CONTAINER_NODES,
    container_argument,
)
from contractor_runtime.toolsets.http.errors import HTTPArgumentError


def request_containers(headers, query, body_type, body, body_limit):
    notices = NormalizationNotices()

    def argument(field, value, kinds=(dict,)):
        def invalid(field, message, **details):
            if field == "body" and body_type == "json":
                message += (
                    " For scalar or exact JSON use body_type=text"
                    " and Content-Type=application/json."
                )
            return HTTPArgumentError(field, message, **details)

        result, normalized = container_argument(
            value,
            field=field,
            kinds=kinds,
            max_bytes=body_limit
            if field == "body"
            else get_shared_config().http_request_body_max_bytes,
            max_depth=MAX_CONTAINER_DEPTH,
            max_nodes=MAX_CONTAINER_NODES,
            invalid=invalid,
        )
        notices.record(field, normalized, "object" if type(result) is dict else "array")
        return result

    headers = argument("headers", headers) if headers is not None else None
    query = argument("query", query) if query is not None else None
    if body_type == "form":
        body = argument("body", body)
    elif body_type == "json":
        body = argument("body", body, (dict, list))
    return headers, query, body, notices
