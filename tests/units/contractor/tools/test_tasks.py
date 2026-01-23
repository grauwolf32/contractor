import json
import re

import pytest
import yaml

from contractor.tools.tasks import TaskFormat, Subtask, TaskExecutionResult
from tests.units.contractor.helpers import MockAgentTool, mk_tool_context
from unittest.mock import AsyncMock

import contractor.tools.tasks as m

@pytest.fixture()
def subtask() -> Subtask:
    return Subtask(
        subtask_id="1.2",
        title="Do thing",
        description="Do the thing safely & quickly",
        status="new",
    )


@pytest.fixture()
def task_result() -> TaskExecutionResult:
    return TaskExecutionResult(
        task_id="3",
        status="done",
        output="Produced artifact <ok> & validated.",
        summary="All steps completed.",
    )


# ---------------------------
# FORMAT: JSON
# ---------------------------


def test_parse_task_result_json_valid(task_result: TaskExecutionResult):
    payload = task_result.model_dump()
    s = json.dumps(payload)
    parsed = TaskFormat._parse_task_result_json(s)

    # Сейчас этот тест, вероятно, УПАДЕТ из-за ошибок в _parse_task_result_json
    assert parsed is not None
    assert parsed.task_id == task_result.task_id
    assert parsed.status == task_result.status
    assert parsed.output == task_result.output
    assert parsed.summary == task_result.summary


@pytest.mark.parametrize("bad", ["", "   \n\t", "not-json", "{bad:}", "[]", "123"])
def test_parse_task_result_json_invalid_returns_none(bad: str):
    parsed = TaskFormat._parse_task_result_json(bad)
    assert parsed is None


def test_parse_task_result_json_accepts_python_literal_dict():
    # ast.literal_eval ветка
    s = "{'task_id': '9', 'status': 'incomplete', 'output': 'x', 'summary': 'y'}"
    parsed = TaskFormat._parse_task_result_json(s)
    assert parsed is not None
    assert parsed.task_id == "9"
    assert parsed.status == "incomplete"


# ---------------------------
# FORMAT: YAML
# ---------------------------


def test_parse_task_result_yaml_valid_mapping_style():
    # ожидаемый формат в формате TaskFormat._task_result_to_yaml:
    # 3:
    #   status: done
    #   output: ...
    #   summary: ...
    s = yaml.safe_dump(
        {"3": {"task_id": "3", "status": "done", "output": "o", "summary": "s"}},
        sort_keys=False,
    )

    parsed = TaskFormat._parse_task_result_yaml(s)

    # Сейчас этот тест, вероятно, УПАДЕТ из-за ошибок в _parse_task_result_yaml
    assert parsed is not None
    assert parsed.task_id == "3"
    assert parsed.status == "done"
    assert parsed.output == "o"
    assert parsed.summary == "s"


@pytest.mark.parametrize("bad", ["", "[]", "x: [1,2", "!!!", "- a\n- b\n"])
def test_parse_task_result_yaml_invalid_returns_none(bad: str):
    parsed = TaskFormat._parse_task_result_yaml(bad)
    assert parsed is None


# ---------------------------
# FORMAT: MARKDOWN
# ---------------------------


def test_parse_task_result_markdown_valid_single_line_fields():
    text = (
        "### RESULT [ID: 42]\n"
        "**Status**: done\n"
        "**Output**: ok\n"
        "**Summary**: fine\n"
        "---\n"
    )
    parsed = TaskFormat._parse_task_result_markdown(text)

    # Сейчас этот тест, вероятно, УПАДЕТ из-за ошибок (text vs output переменная)
    assert parsed is not None
    assert parsed.task_id == "42"
    assert parsed.status == "done"
    assert parsed.output == "ok"
    assert parsed.summary == "fine"


def test_parse_task_result_markdown_multiline_output_and_summary():
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
    parsed = TaskFormat._parse_task_result_markdown(text)
    assert parsed is not None
    assert parsed.task_id == "7"
    assert parsed.status == "incomplete"
    assert parsed.output == "line1\nline2"
    assert parsed.summary == "s1\ns2"


# ---------------------------
# FORMAT: XML
# ---------------------------


def test_parse_task_result_xml_valid():
    xml = (
        '<task_result task_id="10">\n'
        "  <status>done</status>\n"
        "  <output>o</output>\n"
        "  <summary>s</summary>\n"
        "</task_result>"
    )
    parsed = TaskFormat._parse_task_result_xml(xml)
    assert parsed is not None
    assert parsed.task_id == "10"
    assert parsed.status == "done"
    assert parsed.output == "o"
    assert parsed.summary == "s"


@pytest.mark.parametrize("bad", ["", "<x></x>", "<task_result></task_result>"])
def test_parse_task_result_xml_invalid_returns_none(bad: str):
    parsed = TaskFormat._parse_task_result_xml(bad)
    assert parsed is None


# ---------------------------
# type_hint behavior
# ---------------------------


def test_type_hint_wraps_only_when_enabled(task_result: TaskExecutionResult):
    fmt = TaskFormat(_format="markdown")
    out_no_hint = fmt.format_task_result(task_result, type_hint=False)
    assert not out_no_hint.startswith("```")

    out_hint = fmt.format_task_result(task_result, type_hint=True)
    assert out_hint.startswith("```markdown\n")
    assert out_hint.endswith("\n```")


def test_subtask_format_description_xml():
    fmt = TaskFormat(_format="xml")
    assert type(fmt.format_task_result_description()) is str

# ---------------------------
# Behavior tests
# ---------------------------

def _mk_tools(monkeypatch, *, worker, max_tasks=100, use_skip=False):
    monkeypatch.setattr(m, "AgentTool", MockAgentTool)
    monkeypatch.setattr(m, "instrument_worker", lambda w, *a, **k: w)

    fmt = m.TaskFormat(_format="json")
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
    )
    return {fn.__name__: fn for fn in tools}


@pytest.mark.anyio
async def test_current_id_starts_at_0_execute_all_then_add_new_becomes_current(monkeypatch):
    # ---- Create async-mocked worker ----
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)

    # Side-effect: always return a "done" result for the current task_id passed in args
    async def _done_result(*, args, tool_context):
        return {
            "task_id": args["task_id"],
            "status": "done",
            "output": f"completed {args['task_id']}",
            "summary": "ok",
        }

    worker.run_async.side_effect = _done_result

    ctx = mk_tool_context()

    # ---- Add several tasks ----
    for i in range(3):
        res = tool["add_subtask"](
            title=f"t{i}", description=f"d{i}", tool_context=ctx
        )
        assert "error" not in res

    # ---- Current task should be 0 ----
    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "0"

    # ---- Execute all tasks in order ----
    # After each execute (done), manager should advance except for last task
    for expected_id in ["0", "1", "2"]:
        cur = tool["get_current_subtask"](tool_context=ctx)["result"]
        assert cur["task_id"] == expected_id

        exec_res = await tool["execute_current_subtask"](tool_context=ctx)
        assert "error" not in exec_res  # no malformed result
        # record is either dict (json format) or str; in json it’s dict
        assert exec_res["record"]["task_id"] == expected_id
        assert exec_res["record"]["status"] == "done"

    # At this point, current task index typically still points at last task ("2")
    cur_after = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur_after["task_id"] == "2"
    assert cur_after["status"] == "done"

    # ---- Add another one; expected behavior: current becomes the last (new) task ----
    tool["add_subtask"](title="t3", description="d3", tool_context=ctx)

    cur2 = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur2["task_id"] == "3"
    assert cur2["status"] == "new"


@pytest.mark.anyio
async def test_add_new_task_after_decompose(monkeypatch):
    # ---- Create async-mocked worker ----
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)

    # Side-effect: always return a "done" result for the current task_id passed in args
    async def _incomplete_result(*, args, tool_context):
        return {
            "task_id": args["task_id"],
            "status": "incomplete",
            "output": f"completed {args['task_id']}",
            "summary": "ok",
        }

    worker.run_async.side_effect = _incomplete_result

    ctx = mk_tool_context()

    res = tool["add_subtask"](
        title=f"t0", description=f"d0", tool_context=ctx
    )
    assert "error" not in res

    # ---- Current task should be 0 ----
    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "0"

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert "error" not in exec_res  # no malformed result

    assert exec_res["record"]["task_id"] == "0"
    assert exec_res["record"]["status"] == "incomplete"

    # At this point, current task index typically still points at last task ("2")
    res = tool["decompose_subtask"](
        task_id="0",
        decomposition={
            "subtasks": [
                {
                    "title": "sub.t1",
                    "description":"sub.d1"
                },
                {
                    "title": "sub.t2",
                    "description":"sub.d2"
                }
            ],
        },
        tool_context=ctx
    )

    assert "error" not in res

    cur_after = tool["get_current_subtask"](tool_context=ctx)["result"]

    assert cur_after["task_id"] == "0.1"
    assert cur_after["status"] == "new"

@pytest.mark.anyio
async def test_add_new_task_after_decompose_with_multiple(monkeypatch):
    # ---- Create async-mocked worker ----
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()
    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)


    # Side-effect: always return a "done" result for the current task_id passed in args
    async def _incomplete_result(*, args, tool_context):
        return {
            "task_id": args["task_id"],
            "status": "incomplete",
            "output": f"completed {args['task_id']}",
            "summary": "ok",
        }
    
    async def _done_result(*, args, tool_context):
        return {
            "task_id": args["task_id"],
            "status": "done",
            "output": f"completed {args['task_id']}",
            "summary": "ok",
        }

    worker.run_async.side_effect = _done_result

    ctx = mk_tool_context()

    res = tool["add_subtask"](
        title=f"t0", description=f"d0", tool_context=ctx
    )
    assert "error" not in res

    res = tool["add_subtask"](
        title=f"t1", description=f"d1", tool_context=ctx
    )
    assert "error" not in res

    # ---- Current task should be 0 ----
    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "0"

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert "error" not in exec_res  # no malformed result

    assert exec_res["record"]["task_id"] == "0"
    assert exec_res["record"]["status"] == "done"

    cur = tool["get_current_subtask"](tool_context=ctx)["result"]
    assert cur["task_id"] == "1"

    worker.run_async.side_effect = _incomplete_result
    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert "error" not in exec_res  # no malformed result

    assert exec_res["record"]["task_id"] == "1"
    assert exec_res["record"]["status"] == "incomplete"

    # At this point, current task index typically still points at last task ("2")
    res = tool["decompose_subtask"](
        task_id="1",
        decomposition={
            "subtasks": [
                {
                    "title": "sub.t1",
                    "description":"sub.d1"
                },
                {
                    "title": "sub.t2",
                    "description":"sub.d2"
                }
            ],
        },
        tool_context=ctx
    )

    assert "error" not in res

    cur_after = tool["get_current_subtask"](tool_context=ctx)["result"]

    assert cur_after["task_id"] == "1.1"
    assert cur_after["status"] == "new"


@pytest.mark.anyio
async def test_execute_malformed_worker_output_marks_incomplete_and_sets_error(monkeypatch):
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()

    async def _bad(*, args, tool_context):
        return "this is not a valid TaskExecutionResult"

    worker.run_async.side_effect = _bad

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)

    exec_res = await tool["execute_current_subtask"](tool_context=ctx)

    # Tool should flag malformed result
    assert exec_res.get("error") == m.TASK_RESULT_MALFORMED

    # Record is stored as dict in json mode
    rec = exec_res["record"]
    assert rec["task_id"] == "0"
    assert rec["status"] == "incomplete"
    assert rec["summary"] == m.TASK_RESULT_MALFORMED
    assert "this is not a valid TaskExecutionResult" in rec["output"]

@pytest.mark.anyio
async def test_decompose_requires_current_task_id(monkeypatch):
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()

    async def _incomplete(*, args, tool_context):
        return {
            "task_id": args["task_id"],
            "status": "incomplete",
            "output": "blocked",
            "summary": "need more steps",
        }

    worker.run_async.side_effect = _incomplete

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    # Execute current (0) -> incomplete, so current remains 0
    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert exec_res["record"]["task_id"] == "0"
    assert exec_res["record"]["status"] == "incomplete"
    assert m.TASK_REQUIRES_DECOMPOSITION_MSG.format(task_id="0") in exec_res["action"]

    # Try decomposing with wrong id
    res = tool["decompose_subtask"](
        task_id="1",
        decomposition={"subtasks": [{"title": "x", "description": "y"}]},
        tool_context=ctx,
    )
    assert res["error"] == m.TASK_NOT_CURRENT_MSG.format(task_id="1")

    # Correct id works
    res_ok = tool["decompose_subtask"](
        task_id="0",
        decomposition={"subtasks": [{"title": "x", "description": "y"}]},
        tool_context=ctx,
    )
    assert "error" not in res_ok
    assert res_ok["result"][0]["task_id"] == "0.1"


@pytest.mark.anyio
async def test_skip_validations_and_state_transition(monkeypatch):
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=True)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    # Empty reason rejected
    res = tool["skip"](task_id="0", reason="   ", tool_context=ctx)
    assert res["error"] == m.SKIP_REASON_MUST_NOT_BE_EMPTY

    # Wrong task_id rejected (current is 0)
    res = tool["skip"](task_id="1", reason="nope", tool_context=ctx)
    assert res["error"] == m.TASK_NOT_CURRENT_MSG.format(task_id="1")

    # Valid skip moves to next task and marks 0 skipped
    res = tool["skip"](task_id="0", reason="redundant", tool_context=ctx)
    assert res["result"]["task_id"] == "1"

    all_tasks = tool["list_subtasks"](tool_context=ctx)["result"]
    t0 = next(t for t in all_tasks if t["task_id"] == "0")
    assert t0["status"] == "skipped"

    # Record exists
    records = tool["get_records"](tool_context=ctx)["result"]
    assert isinstance(records, list)
    assert records[-1]["task_id"] == "0"
    assert records[-1]["status"] == "skipped"
    assert records[-1]["output"] == "redundant"


@pytest.mark.anyio
async def test_records_accumulate_for_multiple_executes(monkeypatch):
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()

    async def _done(*, args, tool_context):
        return {
            "task_id": args["task_id"],
            "status": "done",
            "output": f"ok {args['task_id']}",
            "summary": "ok",
        }

    worker.run_async.side_effect = _done

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)

    await tool["execute_current_subtask"](tool_context=ctx)  # 0
    await tool["execute_current_subtask"](tool_context=ctx)  # 1

    records = tool["get_records"](tool_context=ctx)["result"]
    assert [r["task_id"] for r in records[-2:]] == ["0", "1"]
    assert all(r["status"] == "done" for r in records[-2:])


@pytest.mark.anyio
async def test_decompose_inserts_children_then_resumes_next_root(monkeypatch):
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()

    async def _done_or_incomplete(*, args, tool_context):
        # Make task 1 incomplete; everything else done
        if args["task_id"] == "1":
            return {
                "task_id": "1",
                "status": "incomplete",
                "output": "blocked at 1",
                "summary": "need decompose",
            }
        return {
            "task_id": args["task_id"],
            "status": "done",
            "output": f"ok {args['task_id']}",
            "summary": "ok",
        }

    worker.run_async.side_effect = _done_or_incomplete

    tool = _mk_tools(monkeypatch, worker=worker, use_skip=False)
    ctx = mk_tool_context()

    tool["add_subtask"](title="t0", description="d0", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d1", tool_context=ctx)
    tool["add_subtask"](title="t2", description="d2", tool_context=ctx)

    # 0 done -> current becomes 1
    await tool["execute_current_subtask"](tool_context=ctx)
    assert tool["get_current_subtask"](tool_context=ctx)["result"]["task_id"] == "1"

    # 1 incomplete -> stays current and demands decomposition
    exec_res = await tool["execute_current_subtask"](tool_context=ctx)
    assert exec_res["record"]["task_id"] == "1"
    assert exec_res["record"]["status"] == "incomplete"
    assert m.TASK_REQUIRES_DECOMPOSITION_MSG.format(task_id="1") in exec_res["action"]

    # Decompose 1 into 1.1 and 1.2 -> current becomes 1.1
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

    # Execute children done -> current becomes next root (2)
    await tool["execute_current_subtask"](tool_context=ctx)  # 1.1
    assert tool["get_current_subtask"](tool_context=ctx)["result"]["task_id"] == "1.2"

    await tool["execute_current_subtask"](tool_context=ctx)  # 1.2
    assert tool["get_current_subtask"](tool_context=ctx)["result"]["task_id"] == "2"