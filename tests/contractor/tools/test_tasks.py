import json
import re

import pytest
import yaml

from contractor.tools.tasks import Format, Subtask, TaskExecutionResult


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
    parsed = Format._parse_task_result_json(s)

    # Сейчас этот тест, вероятно, УПАДЕТ из-за ошибок в _parse_task_result_json
    assert parsed is not None
    assert parsed.task_id == task_result.task_id
    assert parsed.status == task_result.status
    assert parsed.output == task_result.output
    assert parsed.summary == task_result.summary


@pytest.mark.parametrize("bad", ["", "   \n\t", "not-json", "{bad:}", "[]", "123"])
def test_parse_task_result_json_invalid_returns_none(bad: str):
    parsed = Format._parse_task_result_json(bad)
    assert parsed is None


def test_parse_task_result_json_accepts_python_literal_dict():
    # ast.literal_eval ветка
    s = "{'task_id': '9', 'status': 'incomplete', 'output': 'x', 'summary': 'y'}"
    parsed = Format._parse_task_result_json(s)
    assert parsed is not None
    assert parsed.task_id == "9"
    assert parsed.status == "incomplete"


# ---------------------------
# FORMAT: YAML
# ---------------------------


def test_parse_task_result_yaml_valid_mapping_style():
    # ожидаемый формат в формате Format._task_result_to_yaml:
    # 3:
    #   status: done
    #   output: ...
    #   summary: ...
    s = yaml.safe_dump(
        {"3": {"task_id":"3","status": "done", "output": "o", "summary": "s"}},
        sort_keys=False,
    )

    parsed = Format._parse_task_result_yaml(s)

    # Сейчас этот тест, вероятно, УПАДЕТ из-за ошибок в _parse_task_result_yaml
    assert parsed is not None
    assert parsed.task_id == "3"
    assert parsed.status == "done"
    assert parsed.output == "o"
    assert parsed.summary == "s"


@pytest.mark.parametrize("bad", ["", "[]", "x: [1,2", "!!!", "- a\n- b\n"])
def test_parse_task_result_yaml_invalid_returns_none(bad: str):
    parsed = Format._parse_task_result_yaml(bad)
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
    parsed = Format._parse_task_result_markdown(text)

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
    parsed = Format._parse_task_result_markdown(text)
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
    parsed = Format._parse_task_result_xml(xml)
    assert parsed is not None
    assert parsed.task_id == "10"
    assert parsed.status == "done"
    assert parsed.output == "o"
    assert parsed.summary == "s"


@pytest.mark.parametrize("bad", ["", "<x></x>", "<task_result></task_result>"])
def test_parse_task_result_xml_invalid_returns_none(bad: str):
    parsed = Format._parse_task_result_xml(bad)
    assert parsed is None

# ---------------------------
# type_hint behavior
# ---------------------------

def test_type_hint_wraps_only_when_enabled(task_result: TaskExecutionResult):
    fmt = Format(_format="markdown")
    out_no_hint = fmt.format_task_result(task_result, type_hint=False)
    assert not out_no_hint.startswith("```")

    out_hint = fmt.format_task_result(task_result, type_hint=True)
    assert out_hint.startswith("```markdown\n")
    assert out_hint.endswith("\n```")


def test_subtask_format_description_xml():
    fmt = Format(_format="xml")
    assert type(fmt.format_task_result_description()) is str