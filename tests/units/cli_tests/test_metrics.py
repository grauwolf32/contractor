"""Regression tests for cli.metrics._event_to_record credential redaction.

The plugin masks only the ``arguments``/``result``/``tool_response`` keys, but the
trace plugin also emits the raw ADK ``event`` and session ``state`` payloads
(deep-serialized here via ``_jsonable``), which echo the same Authorization /
Set-Cookie / cookie-jar values. Redaction at this persistence boundary must cover
them all before anything is written to metrics.jsonl.
"""
from __future__ import annotations

from cli.metrics import _event_to_record
from contractor.runners.plugins.base import _REDACTED
from contractor.runners.task_runner import TaskRunnerEvent


def test_event_payload_headers_are_redacted():
    # Mirrors an `adk_event` record: the model's function_call args carry an
    # Authorization header that the plugin's key list never touches.
    event = TaskRunnerEvent(
        type="adk_event",
        task_name="trace",
        task_id=1,
        payload={
            "event": {
                "content": {
                    "parts": [
                        {
                            "function_call": {
                                "name": "http_request",
                                "args": {"headers": {"Authorization": "Bearer SECRET"}},
                            }
                        }
                    ]
                }
            }
        },
    )

    record = _event_to_record(event)

    headers = record["event"]["content"]["parts"][0]["function_call"]["args"]["headers"]
    assert headers["Authorization"] == _REDACTED
    # Envelope fields survive.
    assert record["type"] == "adk_event"
    assert record["task_name"] == "trace"


def test_state_snapshot_cookie_jar_is_redacted():
    event = TaskRunnerEvent(
        type="adk_tool_call",
        task_name="trace",
        task_id=1,
        payload={"state": {"cookies": {"sessionid": "SECRET"}}},
    )

    record = _event_to_record(event)

    assert record["state"]["cookies"] == _REDACTED
