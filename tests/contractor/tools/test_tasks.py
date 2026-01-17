import json
import re

import pytest
import yaml

from contractor.tools.tasks import TaskFormat, Subtask, TaskExecutionResult
from tests.contractor.helpers import MockAgentTool, mk_tool_context
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

@pytest.mark.anyio
async def test_current_id_starts_at_0_execute_all_then_add_new_becomes_current(monkeypatch):
    # ---- Patch out google.adk wrappers/instrumentation so this is a pure unit test ----
    monkeypatch.setattr(m, "AgentTool", MockAgentTool)
    monkeypatch.setattr(m, "instrument_worker", lambda worker, *a, **k: worker)

    # ---- Create async-mocked worker ----
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()

    # Side-effect: always return a "done" result for the current task_id passed in args
    async def _done_result(*, args, tool_context):
        return {
            "task_id": args["task_id"],
            "status": "done",
            "output": f"completed {args['task_id']}",
            "summary": "ok",
        }

    worker.run_async.side_effect = _done_result

    fmt = m.TaskFormat(_format="json")
    tools = m.task_tools(
        name="tm",
        max_tasks=100,
        worker=worker,
        fmt=fmt,
        # keep it simple for unit test:
        worker_instrumentation=False,
        use_input_schema=True,
        use_output_schema=False,  # we'll return dicts validated by code anyway
        use_type_hint=False,
        use_skip=False,
    )
    tool = {fn.__name__: fn for fn in tools}

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