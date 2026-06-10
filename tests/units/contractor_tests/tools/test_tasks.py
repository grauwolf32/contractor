import json
from unittest.mock import AsyncMock

import pytest
import yaml
from pydantic import ValidationError

import contractor.tools.tasks as m
from contractor.tools.tasks import (
    Subtask,
    SubtaskDecomposition,
    SubtaskExecutionResult,
    SubtaskFormatter,
)
from tests.units.contractor_tests.helpers import MockAgentTool, mk_tool_context


@pytest.fixture()
def subtask() -> Subtask:
    return Subtask(
        task_id="1.2",
        title="Do thing",
        description="Do the thing safely & quickly",
        status="new",
    )


@pytest.fixture()
def subtask_result() -> SubtaskExecutionResult:
    return SubtaskExecutionResult(
        task_id="3",
        status="done",
        output="Produced artifact <ok> & validated.",
        summary="All steps completed.",
    )


# ---------------------------
# Helpers
# ---------------------------


def _result_json(task_id: str, status: str, output: str, summary: str) -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "status": status,
            "output": output,
            "summary": summary,
        }
    )


def _attach_invocation_context(ctx):
    ctx._invocation_context = type("InvocationCtx", (), {"end_invocation": False})()
    return ctx


def _mk_tools(
    monkeypatch,
    *,
    worker,
    max_tasks=100,
    use_skip=False,
    use_summarization=False,
):
    from contractor.tools.tasks import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "AgentTool", MockAgentTool)
    monkeypatch.setattr(_tools_mod, "instrument_worker", lambda w, *a, **k: w)

    fmt = m.SubtaskFormatter(_format="json")
    tools = m.task_tools(
        name="tm",
        max_tasks=max_tasks,
        worker=worker,
        fmt=fmt,
        worker_instrumentation=False,
        use_input_schema=True,
        use_output_schema=False,
        use_type_hint=False,
        use_skip=use_skip,
        use_summarization=use_summarization,
    )
    return {fn.__name__: fn for fn in tools}


def _mk_worker():
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()
    worker.tools = []
    worker.model = "gpt-3.5-turbo"
    worker.instruction = ""
    # Uninstrumented workers must declare input_schema themselves (the
    # RouterWorkflow contract); task_tools validates this at assembly time.
    worker.input_schema = Subtask
    return worker


# ---------------------------
# FORMAT: JSON
# ---------------------------


def test_parse_subtask_result_json_valid(subtask_result: SubtaskExecutionResult):
    payload = subtask_result.model_dump()
    s = json.dumps(payload)
    parsed = SubtaskFormatter._parse_subtask_result_json(s)

    assert parsed is not None
    assert parsed.task_id == subtask_result.task_id
    assert parsed.status == subtask_result.status
    assert parsed.output == subtask_result.output
    assert parsed.summary == subtask_result.summary


@pytest.mark.parametrize("bad", ["", "   \n\t", "not-json", "{bad:}", "[]", "123"])
def test_parse_subtask_result_json_invalid_returns_none(bad: str):
    parsed = SubtaskFormatter._parse_subtask_result_json(bad)
    assert parsed is None


def test_parse_subtask_result_json_accepts_python_literal_dict():
    s = "{'task_id': '9', 'status': 'incomplete', 'output': 'x', 'summary': 'y'}"
    parsed = SubtaskFormatter._parse_subtask_result_json(s)
    assert parsed is not None
    assert parsed.task_id == "9"
    assert parsed.status == "incomplete"


def test_parse_subtask_result_json_rejects_wrong_shape():
    s = json.dumps({"task_id": "1", "status": "done", "output": "x"})
    parsed = SubtaskFormatter._parse_subtask_result_json(s)
    assert parsed is None


# ---------------------------
# FORMAT: YAML
# ---------------------------


def test_parse_subtask_result_yaml_valid_mapping_style():
    s = yaml.safe_dump(
        {"3": {"task_id": "3", "status": "done", "output": "o", "summary": "s"}},
        sort_keys=False,
    )

    parsed = SubtaskFormatter._parse_subtask_result_yaml(s)

    assert parsed is not None
    assert parsed.task_id == "3"
    assert parsed.status == "done"
    assert parsed.output == "o"
    assert parsed.summary == "s"


def test_parse_subtask_result_yaml_valid_direct_payload():
    s = yaml.safe_dump(
        {"task_id": "8", "status": "incomplete", "output": "o", "summary": "s"},
        sort_keys=False,
    )
    parsed = SubtaskFormatter._parse_subtask_result_yaml(s)
    assert parsed is not None
    assert parsed.task_id == "8"
    assert parsed.status == "incomplete"


@pytest.mark.parametrize("bad", ["", "[]", "x: [1,2", "!!!", "- a\n- b\n"])
def test_parse_subtask_result_yaml_invalid_returns_none(bad: str):
    parsed = SubtaskFormatter._parse_subtask_result_yaml(bad)
    assert parsed is None


# ---------------------------
# FORMAT: MARKDOWN
# ---------------------------


def test_parse_subtask_result_markdown_valid_single_line_fields():
    text = (
        "### RESULT [ID: 42]\n"
        "**Status**: done\n"
        "**Output**: ok\n"
        "**Summary**: fine\n"
        "---\n"
    )
    parsed = SubtaskFormatter._parse_subtask_result_markdown(text)

    assert parsed is not None
    assert parsed.task_id == "42"
    assert parsed.status == "done"
    assert parsed.output == "ok"
    assert parsed.summary == "fine"


def test_parse_subtask_result_markdown_multiline_output_and_summary():
    text = (
        "### RESULT [ID: 7]\n"
        "**Status**: incomplete\n"
        "**Output**: line1\n"
        "line2\n"
        "\n"
        "**Summary**: s1\n"
        "s2\n"
        "---\n"
    )
    parsed = SubtaskFormatter._parse_subtask_result_markdown(text)
    assert parsed is not None
    assert parsed.task_id == "7"
    assert parsed.status == "incomplete"
    assert parsed.output == "line1\nline2"
    assert parsed.summary == "s1\ns2"


def test_parse_subtask_result_markdown_accepts_bullet_fields():
    text = (
        "### RESULT [ID: 11]\n"
        "- **Status**: done\n"
        "- **Output**: artifact\n"
        "- **Summary**: ready\n"
    )
    parsed = SubtaskFormatter._parse_subtask_result_markdown(text)
    assert parsed is not None
    assert parsed.task_id == "11"
    assert parsed.status == "done"


def test_parse_subtask_result_markdown_invalid_returns_none():
    parsed = SubtaskFormatter._parse_subtask_result_markdown("### RESULT [ID: 1]\n")
    assert parsed is None


# ---------------------------
# FORMAT: XML
# ---------------------------


def test_parse_task_result_xml_valid():
    xml = (
        '<result task_id="10">\n'
        "  <status>done</status>\n"
        "  <output>o</output>\n"
        "  <summary>s</summary>\n"
        "</result>"
    )
    parsed = SubtaskFormatter._parse_subtask_result_xml(xml)
    assert parsed is not None
    assert parsed.task_id == "10"
    assert parsed.status == "done"
    assert parsed.output == "o"
    assert parsed.summary == "s"


def test_parse_task_result_xml_nested_in_wrapper():
    xml = (
        "<wrapper>\n"
        '  <result task_id="10">\n'
        "    <status>done</status>\n"
        "    <output>o</output>\n"
        "    <summary>s</summary>\n"
        "  </result>\n"
        "</wrapper>"
    )
    parsed = SubtaskFormatter._parse_subtask_result_xml(xml)
    assert parsed is not None
    assert parsed.task_id == "10"


@pytest.mark.parametrize("bad", ["", "<x></x>", "<result></result>"])
def test_parse_subtask_result_xml_invalid_returns_none(bad: str):
    parsed = SubtaskFormatter._parse_subtask_result_xml(bad)
    assert parsed is None


# ---------------------------
# type_hint behavior
# ---------------------------


def test_type_hint_wraps_only_when_enabled(subtask_result: SubtaskExecutionResult):
    fmt = SubtaskFormatter(_format="markdown")
    out_no_hint = fmt.format_subtask_result(subtask_result, type_hint=False)
    assert not out_no_hint.startswith("```")

    out_hint = fmt.format_subtask_result(subtask_result, type_hint=True)
    assert out_hint.startswith("```markdown\n")
    assert out_hint.endswith("\n```")


def test_subtask_format_description_xml():
    fmt = SubtaskFormatter(_format="xml")
    assert type(fmt.format_subtask_result_description()) is str


def test_parse_subtask_result_uses_fenced_block_and_fallback_task_id():
    fmt = SubtaskFormatter(_format="json")
    raw = """```json
{"task_id":"77","status":"done","output":"ok","summary":"fine"}
```"""
    parsed = fmt.parse_subtask_result(raw, fallback_task_id="0")
    assert parsed is not None
    assert parsed.task_id == "77"

    raw2 = """```json
{"task_id":"","status":"done","output":"ok","summary":"fine"}
```"""
    parsed2 = fmt.parse_subtask_result(raw2, fallback_task_id="0")
    assert parsed2 is not None
    assert parsed2.task_id == "0"


def test_sanitize_llm_output_removes_think_tags_and_outer_quotes():
    text = '"<think>hidden</think>{\\"task_id\\":\\"1\\"}"'
    out = SubtaskFormatter._sanitize_llm_output(text)
    assert "<think>" not in out
    assert "</think>" not in out
    assert out == 'hidden{\\"task_id\\":\\"1\\"}'


# ---------------------------
# Model / validation tests
# ---------------------------


def test_validate_status_transition_accepts_valid():
    assert m.validate_status_transition("new", "done") is True
    assert m.validate_status_transition("new", "malformed") is True


def test_validate_status_transition_rejects_invalid():
    with pytest.raises(m.InvalidStatusTransitionError):
        m.validate_status_transition("done", "done")


def test_subtask_decomposition_model_requires_at_least_one():
    with pytest.raises(ValidationError):
        SubtaskDecomposition.model_validate({"subtasks": []})


def test_subtask_decomposition_model_rejects_more_than_three():
    specs = [{"title": f"t{i}", "description": f"d{i}"} for i in range(4)]
    with pytest.raises(ValidationError):
        SubtaskDecomposition.model_validate({"subtasks": specs})


def test_task_limit_msg_instructs_resolving_subtasks_before_finish():
    msg = m.TASK_LIMIT_REACHED_MSG.format(max_tasks=5)
    # finish(status="done") refuses while 'new' subtasks remain, so the
    # message must not push the planner into an immediate finish call.
    assert "immediately" not in msg
    assert "Execute or skip" in msg


# ---------------------------
# Behavior tests
# ---------------------------


@pytest.mark.anyio
async def test_current_id_starts_at_0_execute_all_then_add_new_becomes_current(
    monkeypatch,
):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)

    async def _done_result(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output=f"completed {args['task_id']}",
            summary="ok",
        )

    worker.run_async.side_effect = _done_result
    ctx = mk_tool_context()

    for i in range(3):
        res = tool["add_subtask"](title=f"t{i}", description=f"d{i}", tool_context=ctx)
        assert "error" not in res

    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "0"

    for expected_id in ["0", "1", "2"]:
        cur = tool["get_current_subtask"](tool_context=ctx)["result"]
        assert cur["task_id"] == expected_id

        exec_res = await tool["execute_current_subtask"](tool_context=ctx)
        assert "error" not in exec_res
        assert exec_res["record"]["task_id"] == expected_id
        assert exec_res["record"]["status"] == "done"

    cur_after = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur_after["task_id"] == "2"
    assert cur_after["status"] == "done"

    tool["add_subtask"](title="t3", description="d3", tool_context=ctx)

    cur2 = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur2["task_id"] == "3"
    assert cur2["status"] == "new"


@pytest.mark.anyio
async def test_add_new_task_after_decompose(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)

    async def _incomplete_result(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output=f"completed {args['task_id']}",
            summary="ok",
        )

    worker.run_async.side_effect = _incomplete_result
    ctx = mk_tool_context()

    res = tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    assert "error" not in res

    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "0"

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert "error" not in exec_res
    assert exec_res["record"]["task_id"] == "0"
    assert exec_res["record"]["status"] == "incomplete"

    res = tool["decompose_subtask"](
        task_id="0",
        decomposition={
            "subtasks": [
                {"title": "sub.t1", "description": "sub.d1"},
                {"title": "sub.t2", "description": "sub.d2"},
            ],
        },
        tool_context=ctx,
    )

    assert "error" not in res

    cur_after = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur_after["task_id"] == "0.1"
    assert cur_after["status"] == "new"


@pytest.mark.anyio
async def test_add_new_task_after_decompose_with_multiple(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)

    async def _incomplete_result(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output=f"completed {args['task_id']}",
            summary="ok",
        )

    async def _done_result(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output=f"completed {args['task_id']}",
            summary="ok",
        )

    worker.run_async.side_effect = _done_result

    ctx = mk_tool_context()

    res = tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    assert "error" not in res

    res = tool["add_subtask"](title="t1", description="d1", tool_context=ctx)
    assert "error" not in res

    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "0"

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert "error" not in exec_res
    assert exec_res["record"]["task_id"] == "0"
    assert exec_res["record"]["status"] == "done"

    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "1"

    worker.run_async.side_effect = _incomplete_result
    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert "error" not in exec_res
    assert exec_res["record"]["task_id"] == "1"
    assert exec_res["record"]["status"] == "incomplete"

    res = tool["decompose_subtask"](
        task_id="1",
        decomposition={
            "subtasks": [
                {"title": "sub.t1", "description": "sub.d1"},
                {"title": "sub.t2", "description": "sub.d2"},
            ],
        },
        tool_context=ctx,
    )

    assert "error" not in res

    cur_after = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur_after["task_id"] == "1.1"
    assert cur_after["status"] == "new"


@pytest.mark.anyio
async def test_execute_malformed_worker_output_marks_malformed_and_sets_error(
    monkeypatch,
):
    worker = _mk_worker()

    async def _bad(*, args, tool_context):
        return "this is not a valid SubtaskExecutionResult"

    worker.run_async.side_effect = _bad

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)

    assert exec_res.get("error") == m.SUBTASK_RESULT_MALFORMED

    rec = exec_res["record"]
    assert rec["task_id"] == "0"
    assert rec["status"] == "malformed"
    assert rec["summary"] == m.SUBTASK_RESULT_MALFORMED
    assert "this is not a valid SubtaskExecutionResult" in rec["output"]


@pytest.mark.anyio
async def test_execute_mismatched_task_id_becomes_malformed(monkeypatch):
    worker = _mk_worker()

    async def _bad_id(*, args, tool_context):
        return _result_json(
            task_id="999",
            status="done",
            output="wrong task",
            summary="wrong task",
        )

    worker.run_async.side_effect = _bad_id

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)

    assert exec_res["error"] == m.SUBTASK_RESULT_MALFORMED
    assert exec_res["record"]["task_id"] == "0"
    assert exec_res["record"]["status"] == "malformed"
    assert "expected '0'" in exec_res["record"]["output"]


@pytest.mark.anyio
async def test_decompose_requires_current_task_id(monkeypatch):
    worker = _mk_worker()

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert exec_res["record"]["task_id"] == "0"
    assert exec_res["record"]["status"] == "incomplete"
    assert (
        m.SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(
            task_id="0",
            status="incomplete",
        )
        in exec_res["action"]
    )

    res = tool["decompose_subtask"](
        task_id="1",
        decomposition={"subtasks": [{"title": "x", "description": "y"}]},
        tool_context=ctx,
    )
    assert res["error"] == m.SUBTASK_NOT_CURRENT_MSG.format(task_id="1")

    res_ok = tool["decompose_subtask"](
        task_id="0",
        decomposition={"subtasks": [{"title": "x", "description": "y"}]},
        tool_context=ctx,
    )
    assert "error" not in res_ok
    assert res_ok["result"][0]["task_id"] == "0.1"


@pytest.mark.anyio
async def test_skip_validations_and_state_transition(monkeypatch):
    worker = _mk_worker()

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=True)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    res = tool["skip"](task_id="0", reason="   ", tool_context=ctx)
    assert res["error"] == m.SKIP_REASON_MUST_NOT_BE_EMPTY

    res = tool["skip"](task_id="1", reason="nope", tool_context=ctx)
    assert res["error"] == m.SUBTASK_NOT_CURRENT_MSG.format(task_id="1")

    res = tool["skip"](task_id="0", reason="redundant", tool_context=ctx)
    assert res["result"] == "ok"
    assert res["next-subtask"]["task_id"] == "1"

    all_tasks = tool["list_subtasks"](tool_context=ctx, view="all")["result"]
    t0 = next(t for t in all_tasks if t["task_id"] == "0")
    assert t0["status"] == "skipped"

    records = tool["get_records"](tool_context=ctx)["result"]
    assert isinstance(records, list)
    assert records[-1]["task_id"] == "0"
    assert records[-1]["status"] == "skipped"
    assert records[-1]["output"] == "redundant"


@pytest.mark.anyio
async def test_skip_incomplete_non_last_requires_decomposition(monkeypatch):
    worker = _mk_worker()

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=True)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    await tool["execute_current_subtask"](tool_context=ctx)

    res = tool["skip"](task_id="0", reason="cannot continue", tool_context=ctx)
    assert "error" in res
    assert "cannot be skipped unless it is the last remaining subtask" in res["error"]


@pytest.mark.anyio
async def test_skip_incomplete_last_is_allowed(monkeypatch):
    worker = _mk_worker()

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=True)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    await tool["execute_current_subtask"](tool_context=ctx)
    res = tool["skip"](task_id="0", reason="cannot continue", tool_context=ctx)

    assert res["result"] == m.NO_ACTIVE_SUBTASKS_MSG

    records = tool["get_records"](tool_context=ctx)["result"]
    assert records[-1]["task_id"] == "0"
    assert records[-1]["status"] == "skipped"


@pytest.mark.anyio
async def test_skip_resolved_current_subtask_returns_error(monkeypatch):
    worker = _mk_worker()

    async def _done(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output="ok",
            summary="ok",
        )

    worker.run_async.side_effect = _done

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=True)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    await tool["execute_current_subtask"](tool_context=ctx)

    # The last subtask is 'done' and still current — the rejected skip must
    # surface as an error naming the cause, not as the no-active-subtasks
    # success message.
    res = tool["skip"](task_id="0", reason="redundant", tool_context=ctx)
    assert res["error"] == m.SUBTASK_SKIP_NOT_SKIPPABLE.format(
        task_id="0", status="done"
    )

    # State untouched: subtask stays 'done', no skip record was appended.
    all_tasks = tool["list_subtasks"](tool_context=ctx, view="all")["result"]
    assert all_tasks[0]["status"] == "done"
    records = tool["get_records"](tool_context=ctx)["result"]
    assert all(r["status"] != "skipped" for r in records)


# ---------------------------
# task_tools assembly validation
# ---------------------------


def test_task_tools_uninstrumented_requires_worker_input_schema():
    worker = _mk_worker()
    worker.input_schema = None

    with pytest.raises(ValueError, match="input_schema"):
        m.task_tools(
            name="tm",
            max_tasks=10,
            worker=worker,
            fmt=m.SubtaskFormatter(_format="markdown"),
            worker_instrumentation=False,
            use_input_schema=True,
            use_summarization=False,
        )


def test_task_tools_uninstrumented_accepts_worker_with_input_schema(monkeypatch):
    worker = _mk_worker()  # sets input_schema = Subtask
    tool = _mk_tools(monkeypatch, worker=worker)
    assert "execute_current_subtask" in tool


def test_task_tools_uninstrumented_allows_missing_schema_when_disabled(monkeypatch):
    from contractor.tools.tasks import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "AgentTool", MockAgentTool)

    worker = _mk_worker()
    worker.input_schema = None
    tools = m.task_tools(
        name="tm",
        max_tasks=10,
        worker=worker,
        fmt=m.SubtaskFormatter(_format="markdown"),
        worker_instrumentation=False,
        use_input_schema=False,
        use_summarization=False,
    )
    assert any(fn.__name__ == "execute_current_subtask" for fn in tools)


@pytest.mark.anyio
async def test_records_accumulate_for_multiple_executes(monkeypatch):
    worker = _mk_worker()

    async def _done(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output=f"ok {args['task_id']}",
            summary="ok",
        )

    worker.run_async.side_effect = _done

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    await tool["execute_current_subtask"](tool_context=ctx)
    await tool["execute_current_subtask"](tool_context=ctx)

    records = tool["get_records"](tool_context=ctx)["result"]
    assert [r["task_id"] for r in records[-2:]] == ["0", "1"]
    assert all(r["status"] == "done" for r in records[-2:])


@pytest.mark.anyio
async def test_decompose_inserts_children_then_resumes_next_root(monkeypatch):
    worker = _mk_worker()

    async def _done_or_incomplete(*, args, tool_context):
        if args["task_id"] == "1":
            return _result_json(
                task_id="1",
                status="incomplete",
                output="blocked at 1",
                summary="need decompose",
            )
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output=f"ok {args['task_id']}",
            summary="ok",
        )

    worker.run_async.side_effect = _done_or_incomplete

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)
    tool["add_subtask"](title="t2", description="d2", tool_context=ctx)

    await tool["execute_current_subtask"](tool_context=ctx)
    assert tool["get_current_subtask"](tool_context=ctx)["result"]["task_id"] == "1"

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert exec_res["record"]["task_id"] == "1"
    assert exec_res["record"]["status"] == "incomplete"
    assert (
        m.SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(
            task_id="1",
            status="incomplete",
        )
        in exec_res["action"]
    )

    tool["decompose_subtask"](
        task_id="1",
        decomposition={
            "subtasks": [
                {"title": "s1", "description": "sd1"},
                {"title": "s2", "description": "sd2"},
            ]
        },
        tool_context=ctx,
    )
    assert tool["get_current_subtask"](tool_context=ctx)["result"]["task_id"] == "1.1"

    await tool["execute_current_subtask"](tool_context=ctx)
    assert tool["get_current_subtask"](tool_context=ctx)["result"]["task_id"] == "1.2"

    await tool["execute_current_subtask"](tool_context=ctx)
    assert tool["get_current_subtask"](tool_context=ctx)["result"]["task_id"] == "2"


@pytest.mark.anyio
async def test_execute_current_subtask_retries_until_non_empty(monkeypatch):
    worker = _mk_worker()
    worker.run_async.side_effect = [
        "",
        "   ",
        _result_json("0", "done", "ok", "fine"),
    ]

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()
    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    res = await tool["execute_current_subtask"](tool_context=ctx)
    assert "error" not in res
    assert res["record"]["task_id"] == "0"
    assert worker.run_async.await_count == 3


@pytest.mark.anyio
async def test_execute_current_subtask_blocks_when_current_is_incomplete(monkeypatch):
    worker = _mk_worker()

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()
    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    first = await tool["execute_current_subtask"](tool_context=ctx)
    assert first["record"]["status"] == "incomplete"

    second = await tool["execute_current_subtask"](tool_context=ctx)
    assert second["error"] == m.SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(
        task_id="0",
        status="incomplete",
    )


@pytest.mark.anyio
async def test_decompose_over_capacity_suggests_fewer_children(monkeypatch):
    worker = _mk_worker()

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete

    tool = _mk_tools(monkeypatch, worker=worker, max_tasks=3, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)
    await tool["execute_current_subtask"](tool_context=ctx)

    # 2 subtasks exist, limit is 3 → only 1 more fits, but 3 are requested.
    res = tool["decompose_subtask"](
        task_id="0",
        decomposition={
            "subtasks": [
                {"title": f"s{i}", "description": f"sd{i}"} for i in range(3)
            ]
        },
        tool_context=ctx,
    )
    assert res["error"] == m.SUBTASK_DECOMPOSE_OVER_CAPACITY.format(
        requested=3, max_tasks=3, remaining=1
    )

    # A single child still fits.
    res_ok = tool["decompose_subtask"](
        task_id="0",
        decomposition={"subtasks": [{"title": "s", "description": "sd"}]},
        tool_context=ctx,
    )
    assert "error" not in res_ok


@pytest.mark.anyio
async def test_decompose_at_full_limit_returns_limit_reached(monkeypatch):
    worker = _mk_worker()

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete

    tool = _mk_tools(monkeypatch, worker=worker, max_tasks=2, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)
    await tool["execute_current_subtask"](tool_context=ctx)

    # Limit fully spent → no number of children would fit.
    res = tool["decompose_subtask"](
        task_id="0",
        decomposition={"subtasks": [{"title": "s", "description": "sd"}]},
        tool_context=ctx,
    )
    assert res["error"] == m.TASK_LIMIT_REACHED_MSG.format(max_tasks=2)


@pytest.mark.anyio
async def test_decompose_subtask_rejects_string_input(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete
    await tool["execute_current_subtask"](tool_context=ctx)

    res = tool["decompose_subtask"](
        task_id="0",
        decomposition="not-a-structured-object",
        tool_context=ctx,
    )
    assert "error" in res
    assert "TypeError" in res["error"]
    assert "SubtaskDecomposition" in res["error"]


@pytest.mark.anyio
async def test_decompose_subtask_rejects_invalid_dict(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    async def _incomplete(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="incomplete",
            output="blocked",
            summary="need more steps",
        )

    worker.run_async.side_effect = _incomplete
    await tool["execute_current_subtask"](tool_context=ctx)

    res = tool["decompose_subtask"](
        task_id="0",
        decomposition={"subtasks": []},
        tool_context=ctx,
    )
    assert "error" in res
    assert "Validation error in decomposition" in res["error"]


@pytest.mark.anyio
async def test_list_subtasks_remaining_and_all(monkeypatch):
    worker = _mk_worker()

    async def _done(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output="ok",
            summary="ok",
        )

    worker.run_async.side_effect = _done

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)
    tool["add_subtask"](title="t2", description="d2", tool_context=ctx)

    await tool["execute_current_subtask"](tool_context=ctx)

    remaining = tool["list_subtasks"](tool_context=ctx, view="remaining")["result"]
    assert [t["task_id"] for t in remaining] == ["1", "2"]

    all_tasks = tool["list_subtasks"](tool_context=ctx, view="all")["result"]
    assert [t["task_id"] for t in all_tasks] == ["0", "1", "2"]
    assert all_tasks[0]["status"] == "done"


@pytest.mark.anyio
async def test_list_subtasks_remaining_returns_no_remaining_message_when_last_done(
    monkeypatch,
):
    worker = _mk_worker()

    async def _done(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output="ok",
            summary="ok",
        )

    worker.run_async.side_effect = _done

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()
    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    await tool["execute_current_subtask"](tool_context=ctx)

    res = tool["list_subtasks"](tool_context=ctx, view="remaining")
    assert res["result"] == m.NO_REMAINING_SUBTASKS_MSG


@pytest.mark.anyio
async def test_add_subtask_respects_max_tasks(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, max_tasks=1, use_skip=False)
    ctx = mk_tool_context()

    first = tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    second = tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    assert "error" not in first
    assert second["error"] == m.TASK_LIMIT_REACHED_MSG.format(max_tasks=1)


@pytest.mark.anyio
async def test_execute_current_subtask_returns_no_active_when_none_exist(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    res = await tool["execute_current_subtask"](tool_context=ctx)
    assert res["error"] == m.NO_ACTIVE_SUBTASKS_MSG


@pytest.mark.anyio
async def test_get_current_subtask_returns_no_subtasks_when_none_exist(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    res = tool["get_current_subtask"](tool_context=ctx)
    assert res["error"] == m.NO_SUBTASKS_EXIST_MSG


@pytest.mark.anyio
async def test_finish_done_rejects_when_new_subtasks_remain(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(
        monkeypatch, worker=worker, use_skip=False, use_summarization=False
    )
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    res = await tool["finish"](
        status="done",
        result="everything complete",
        tool_context=ctx,
    )
    assert res["error"] == m.DO_NOT_FINISH_WITH_NO_TASKS_DONE


@pytest.mark.anyio
async def test_finish_failed_succeeds_without_completed_subtasks(monkeypatch):
    worker = _mk_worker()
    tool = _mk_tools(
        monkeypatch, worker=worker, use_skip=False, use_summarization=False
    )
    ctx = mk_tool_context()
    ctx = _attach_invocation_context(mk_tool_context())

    res = await tool["finish"](
        status="failed",
        result="blocked immediately",
        tool_context=ctx,
    )

    assert res["result"] == "ok"
    assert ctx._invocation_context.end_invocation is True

    status_key = "task::0::status"
    result_key = "task::0::result"
    assert ctx.state[status_key] == "failed"
    assert ctx.state[result_key] == "blocked immediately"


@pytest.mark.anyio
async def test_finish_done_succeeds_after_all_tasks_resolved(monkeypatch):
    worker = _mk_worker()

    async def _done(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output="ok",
            summary="ok",
        )

    worker.run_async.side_effect = _done

    tool = _mk_tools(
        monkeypatch, worker=worker, use_skip=False, use_summarization=False
    )
    ctx = mk_tool_context()
    ctx = _attach_invocation_context(mk_tool_context())

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    await tool["execute_current_subtask"](tool_context=ctx)

    res = await tool["finish"](
        status="done",
        result="completed successfully",
        tool_context=ctx,
    )

    assert res["result"] == "ok"
    assert ctx._invocation_context.end_invocation is True
    assert ctx.state["task::0::status"] == "done"
    assert ctx.state["task::0::result"] == "completed successfully"


# ---------------------------
# Summarizer construction + record truncation
# ---------------------------


@pytest.mark.anyio
async def test_finish_summarizer_has_no_tools_and_caps_records(monkeypatch):
    from contractor.tools.tasks import tools as _tools_mod

    captured: dict = {}

    class StubLlmAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

        async def run_async(self, *, args, tool_context):
            captured["request"] = args["request"]
            return "summary-text"

    worker = _mk_worker()
    worker.tools = [lambda: None]  # the summarizer must NOT inherit these

    monkeypatch.setattr(_tools_mod, "LlmAgent", StubLlmAgent)
    monkeypatch.setattr(_tools_mod, "_get_agent_ref", lambda w: worker)

    async def _done(*, args, tool_context):
        return _result_json(
            task_id=args["task_id"],
            status="done",
            output="ok",
            summary="ok",
        )

    worker.run_async.side_effect = _done

    tool = _mk_tools(
        monkeypatch, worker=worker, use_skip=False, use_summarization=True
    )
    ctx = _attach_invocation_context(mk_tool_context())

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    await tool["execute_current_subtask"](tool_context=ctx)

    # Seed the records pool well past max_records (default 20), including one
    # oversized record that must be truncated for the summarizer payload.
    pool_key = m.StreamlineManager._task_keys(ctx).pool
    giant = "z" * (_tools_mod._MAX_RECORD_FIELD_LEN + 5_000)
    ctx.state[pool_key] = ctx.state[pool_key] + [f"rec-{i}" for i in range(30)] + [
        {"task_id": "x", "status": "done", "output": giant, "summary": "big"}
    ]

    res = await tool["finish"](
        status="done",
        result="completed successfully",
        tool_context=ctx,
    )
    assert res["result"] == "ok"

    # The summarizer agent was built with an empty toolset.
    assert captured["agent_kwargs"]["tools"] == []

    payload = json.loads(captured["request"])
    # Only the most recent max_records (20) records are passed on.
    assert len(payload["records"]) == 20
    # The oversized record's output field was truncated with a marker.
    big_rec = payload["records"][-1]
    assert big_rec["output"].endswith(_tools_mod._TRUNCATION_MARKER)
    assert len(big_rec["output"]) <= _tools_mod._MAX_RECORD_FIELD_LEN + len(
        _tools_mod._TRUNCATION_MARKER
    )
    # The summarizer output landed in the task summary slot.
    assert ctx.state["task::0::summary"] == "summary-text"


@pytest.mark.anyio
async def test_execute_malformed_raw_output_is_truncated_in_record(monkeypatch):
    from contractor.tools.tasks import tools as _tools_mod

    worker = _mk_worker()
    giant = "not-parseable " + "z" * (_tools_mod._MAX_RECORD_FIELD_LEN + 5_000)

    async def _bad(*, args, tool_context):
        return giant

    worker.run_async.side_effect = _bad

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    exec_res = await tool["execute_current_subtask"](tool_context=ctx)

    rec = exec_res["record"]
    assert rec["status"] == "malformed"
    assert rec["output"].endswith(_tools_mod._TRUNCATION_MARKER)
    assert len(rec["output"]) <= _tools_mod._MAX_RECORD_FIELD_LEN + len(
        _tools_mod._TRUNCATION_MARKER
    )
