import json
import sys
from textwrap import indent
from typing import Any


from cli.helpers import (
    _normalize_event,
    _fmt_tool_call_event,
    _fmt_tool_result_event,
    _fmt_tool_error_event,
    _fmt_tool_data,
    _try_parse_json_like,
    _fmt_dict,
    _fmt_list,
    _fmt_value,
    _fmt_blob,
    _clamp_lines,
    _wrap_text
)

IGNORED_TOOL_RESULTS = {
    "add_subtask",
    "decompose_subtask",
}

LOW_SIGNAL_TOOL_RESULTS = {
    "get_current_subtask",
    "list_subtasks",
    "get_records",
}

FS_TOOLS = {
    "ls",
    "glob",
    "read_file",
    "grep",
    "interaction_stats",
    "list_touched_files",
    "list_untouched_files",
}

MEMORY_TOOLS = {
    "write_memory",
    "append_memory",
    "read_memory",
    "search_memory",
    "list_tags",
    "list_memories",
}

OPENAPI_TOOLS = {
    "list_paths",
    "list_components",
    "list_servers",
    "get_info",
    "get_path",
    "get_component",
    "set_info",
    "add_server",
    "upsert_path",
    "upsert_component",
    "remove_server",
    "remove_path",
    "remove_component",
    "get_full_openapi_schema",
}

TASK_TOOLS = {
    "add_subtask",
    "get_current_subtask",
    "list_subtasks",
    "get_records",
    "decompose_subtask",
    "skip",
    "finish",
    "execute_current_subtask",
}


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [make_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(make_jsonable(v) for v in value)
    return value


class C:
    ENABLED = sys.stdout.isatty()

    RESET = "\033[0m" if ENABLED else ""

    BOLD = "\033[1m" if ENABLED else ""
    DIM = "\033[2m" if ENABLED else ""
    UNDERLINE = "\033[4m" if ENABLED else ""

    BLACK = "\033[30m" if ENABLED else ""
    RED = "\033[31m" if ENABLED else ""
    GREEN = "\033[32m" if ENABLED else ""
    YELLOW = "\033[33m" if ENABLED else ""
    BLUE = "\033[34m" if ENABLED else ""
    MAGENTA = "\033[35m" if ENABLED else ""
    CYAN = "\033[36m" if ENABLED else ""
    WHITE = "\033[37m" if ENABLED else ""
    GRAY = "\033[90m" if ENABLED else ""

    @classmethod
    def wrap(cls, text: str, *styles: str) -> str:
        if not cls.ENABLED:
            return text
        return f"{''.join(styles)}{text}{cls.RESET}"


def _hr(char: str = "─", width: int = 80) -> str:
    return char * width


def _j(data: Any) -> str:
    return json.dumps(make_jsonable(data), ensure_ascii=False, indent=2)


def _short(text: Any, limit: int = 160) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _short_block(text: Any, limit: int = 1200) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _first_nonempty_line(text: Any) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _render_subtasks(subtasks: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for idx, subtask in enumerate(subtasks, start=1):
        title = subtask.get("title", "Untitled")
        description = subtask.get("description", "")
        task_id = subtask.get("task_id")
        status = subtask.get("status")

        head = f"      {C.wrap(f'{idx}.', C.MAGENTA)} {C.wrap(title, C.BOLD)}"

        meta: list[str] = []
        if task_id is not None:
            meta.append(f"id={task_id}")
        if status:
            meta.append(f"status={status}")
        if meta:
            head += f" {C.wrap('(' + ', '.join(meta) + ')', C.DIM)}"

        lines.append(head)

        if description:
            lines.append(C.wrap(indent(description, "         "), C.DIM))

    return "\n".join(lines)


def _render_kv_lines(items: list[tuple[str, Any]]) -> str:
    lines: list[str] = []

    for key, value in items:
        if value in (None, "", [], {}, ()):
            continue
        lines.append(f"    {C.wrap(f'{key}:', C.DIM)} {value}")

    return "\n".join(lines)


def _fmt_fs_paging(result: dict[str, Any]) -> str | None:
    total = result.get("total_items")
    offset = result.get("offset")
    limit = result.get("limit")
    interaction = result.get("interaction")

    parts: list[str] = []
    if total is not None:
        parts.append(f"total={total}")
    if offset is not None:
        parts.append(f"offset={offset}")
    if limit is not None:
        parts.append(f"limit={limit}")
    if interaction:
        parts.append(f"interaction={interaction}")

    if not parts:
        return None

    return f"    {C.wrap('meta:', C.DIM)} " + ", ".join(parts)


def _fmt_tool_args(tool_name: str, args: dict[str, Any] | None) -> str:
    if not args or type(args) is not dict:
        return ""

    if tool_name == "add_subtask":
        title = args.get("title", "Untitled")
        description = args.get("description", "")
        lines = [f"    {C.wrap('•', C.CYAN)} {C.wrap(title, C.BOLD)}"]
        if description:
            lines.append(C.wrap(indent(description, "      "), C.DIM))
        return "\n".join(lines)

    if tool_name == "decompose_subtask":
        task_id = args.get("task_id")
        decomposition = args.get("decomposition") or {}
        if type(decomposition) is str:
            return ""

        subtasks = decomposition.get("subtasks") or []
        lines = [
            f"    {C.wrap('↳', C.MAGENTA)} "
            f"{C.wrap(f'Subtask decomposition {task_id}', C.BOLD)} "
            f"{C.wrap(f'({len(subtasks)} pcs.)', C.DIM)}"
        ]
        body = _render_subtasks(subtasks)
        if body:
            lines.append(body)
        return "\n".join(lines)

    if tool_name == "skip":
        task_id = args.get("task_id")
        reason = args.get("reason", "")
        lines = [
            f"    {C.wrap('⤼', C.YELLOW)} {C.wrap(f'Skipping subtask {task_id}', C.BOLD)}"
        ]
        if reason:
            lines.append(f"    {C.wrap('reason:', C.DIM)} {reason}")
        return "\n".join(lines)

    if tool_name == "finish":
        status = args.get("status", "unknown")
        result = args.get("result", "")
        lines = [
            f"    {C.wrap('🏁', C.GREEN)} {C.wrap('Finalizing global subtask', C.BOLD)}",
            f"    {C.wrap('status:', C.DIM)} {status}",
        ]
        if result:
            lines.append(f"    {C.wrap('result:', C.DIM)} {_short(result)}")
        return "\n".join(lines)

    if tool_name == "write_memory":
        lines = [
            f"    {C.wrap('🧠', C.CYAN)} {C.wrap(args.get('name', 'Untitled'), C.BOLD)}"
        ]
        if args.get("description"):
            lines.append(f"    {C.wrap('description:', C.DIM)} {args['description']}")
        if args.get("tags"):
            lines.append(
                f"    {C.wrap('tags:', C.DIM)} {', '.join(map(str, args['tags']))}"
            )
        if args.get("memory"):
            lines.append(f"    {C.wrap('memory:', C.DIM)} {_short(args['memory'])}")
        return "\n".join(lines)

    if tool_name == "append_memory":
        name = args.get("name")
        text = args.get("text", "")
        return (
            f"    {C.wrap('➕', C.CYAN)} "
            f"{C.wrap(f'Appending to memory {name}', C.BOLD)}\n"
            f"    {C.wrap('text:', C.DIM)} {_short(text)}"
        )

    if tool_name == "read_memory":
        name = args.get("name")
        return f"    {C.wrap('📖', C.CYAN)} {C.wrap(f'Reading memory {name}', C.BOLD)}"

    if tool_name == "search_memory":
        tags = args.get("tags") or []
        return (
            f"    {C.wrap('🔎', C.CYAN)} {C.wrap('Memory search', C.BOLD)}\n"
            f"    {C.wrap('tags:', C.DIM)} {', '.join(map(str, tags)) if tags else '—'}"
        )

    if tool_name == "list_tags":
        return f"    {C.wrap('🏷', C.CYAN)} {C.wrap('List memory tags', C.BOLD)}"

    if tool_name == "list_memories":
        return f"    {C.wrap('🗃', C.CYAN)} {C.wrap('List memories', C.BOLD)}"

    if tool_name == "ls":
        return (
            f"    {C.wrap('📁', C.CYAN)} {C.wrap('List directory', C.BOLD)}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '/')}"
        )

    if tool_name == "glob":
        return (
            f"    {C.wrap('🧭', C.CYAN)} {C.wrap('Glob search', C.BOLD)}\n"
            f"    {C.wrap('pattern:', C.DIM)} {args.get('pattern', '')}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '/')}\n"
            f"    {C.wrap('offset:', C.DIM)} {args.get('offset', 0)}"
        )

    if tool_name == "read_file":
        lines = [
            f"    {C.wrap('📄', C.CYAN)} {C.wrap('Read file', C.BOLD)}",
            f"    {C.wrap('file:', C.DIM)} {args.get('file', '')}",
        ]
        if args.get("offset") is not None:
            lines.append(f"    {C.wrap('offset:', C.DIM)} {args['offset']}")
        if args.get("limit") is not None:
            lines.append(f"    {C.wrap('limit:', C.DIM)} {args['limit']}")
        return "\n".join(lines)

    if tool_name == "grep":
        return (
            f"    {C.wrap('🔎', C.CYAN)} {C.wrap('Regex search', C.BOLD)}\n"
            f"    {C.wrap('pattern:', C.DIM)} {args.get('pattern', '')}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '/')}\n"
            f"    {C.wrap('offset:', C.DIM)} {args.get('offset', 0)}"
        )

    if tool_name == "interaction_stats":
        return (
            f"    {C.wrap('📊', C.CYAN)} {C.wrap('Interaction stats', C.BOLD)}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '/')}\n"
            f"    {C.wrap('pattern:', C.DIM)} {args.get('pattern', '**/*')}"
        )

    if tool_name == "list_touched_files":
        return (
            f"    {C.wrap('✅', C.CYAN)} {C.wrap('Covered files', C.BOLD)}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '/')}\n"
            f"    {C.wrap('pattern:', C.DIM)} {args.get('pattern', '**/*')}\n"
            f"    {C.wrap('interaction:', C.DIM)} {args.get('interaction', 'any')}\n"
            f"    {C.wrap('offset:', C.DIM)} {args.get('offset', 0)}"
        )

    if tool_name == "list_untouched_files":
        return (
            f"    {C.wrap('🫥', C.CYAN)} {C.wrap('Untouched files', C.BOLD)}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '/')}\n"
            f"    {C.wrap('pattern:', C.DIM)} {args.get('pattern', '**/*')}\n"
            f"    {C.wrap('offset:', C.DIM)} {args.get('offset', 0)}"
        )

    if tool_name == "code_execution_tool":
        return (
            f"    {C.wrap('🐚', C.CYAN)} {C.wrap('Execute command in container', C.BOLD)}\n"
            f"    {C.wrap('command:', C.DIM)} {_short(args.get('command', ''))}"
        )

    if tool_name == "list_paths":
        return f"    {C.wrap('🛣', C.CYAN)} {C.wrap('List OpenAPI paths', C.BOLD)}"

    if tool_name == "list_components":
        return (
            f"    {C.wrap('🧩', C.CYAN)} {C.wrap('List OpenAPI components', C.BOLD)}\n"
            f"    {C.wrap('key:', C.DIM)} {args.get('key', '')}"
        )

    if tool_name == "list_servers":
        return f"    {C.wrap('🌐', C.CYAN)} {C.wrap('List OpenAPI servers', C.BOLD)}"

    if tool_name == "get_info":
        return f"    {C.wrap('ℹ', C.CYAN)} {C.wrap('Read OpenAPI info', C.BOLD)}"

    if tool_name == "get_path":
        return (
            f"    {C.wrap('🛣', C.CYAN)} {C.wrap('Read path definition', C.BOLD)}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '')}"
        )

    if tool_name == "get_component":
        return (
            f"    {C.wrap('🧩', C.CYAN)} {C.wrap('Read component', C.BOLD)}\n"
            f"    {C.wrap('key:', C.DIM)} {args.get('key', '')}\n"
            f"    {C.wrap('component:', C.DIM)} {args.get('component_name', '')}"
        )

    if tool_name == "set_info":
        lines = [
            f"    {C.wrap('✍', C.CYAN)} {C.wrap('Update OpenAPI info', C.BOLD)}",
        ]
        if args.get("title"):
            lines.append(f"    {C.wrap('title:', C.DIM)} {args['title']}")
        if args.get("version"):
            lines.append(f"    {C.wrap('version:', C.DIM)} {args['version']}")
        if args.get("description"):
            lines.append(
                f"    {C.wrap('description:', C.DIM)} {_short(args['description'])}"
            )
        if args.get("code_language"):
            lines.append(f"    {C.wrap('language:', C.DIM)} {args['code_language']}")
        return "\n".join(lines)

    if tool_name == "add_server":
        return (
            f"    {C.wrap('🌐', C.CYAN)} {C.wrap('Add server', C.BOLD)}\n"
            f"    {C.wrap('url:', C.DIM)} {args.get('url', '')}\n"
            f"    {C.wrap('description:', C.DIM)} {_short(args.get('description', '')) or '—'}"
        )

    if tool_name == "remove_server":
        return (
            f"    {C.wrap('🗑', C.CYAN)} {C.wrap('Remove server', C.BOLD)}\n"
            f"    {C.wrap('url:', C.DIM)} {args.get('url', '')}"
        )

    if tool_name == "upsert_path":
        return (
            f"    {C.wrap('🛠', C.CYAN)} {C.wrap('Upsert path', C.BOLD)}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '')}"
        )

    if tool_name == "remove_path":
        return (
            f"    {C.wrap('🗑', C.CYAN)} {C.wrap('Remove path', C.BOLD)}\n"
            f"    {C.wrap('path:', C.DIM)} {args.get('path', '')}"
        )

    if tool_name == "upsert_component":
        return (
            f"    {C.wrap('🧩', C.CYAN)} {C.wrap('Upsert component', C.BOLD)}\n"
            f"    {C.wrap('key:', C.DIM)} {args.get('key', '')}\n"
            f"    {C.wrap('component:', C.DIM)} {args.get('component_name', '')}"
        )

    if tool_name == "remove_component":
        return (
            f"    {C.wrap('🗑', C.CYAN)} {C.wrap('Remove component', C.BOLD)}\n"
            f"    {C.wrap('key:', C.DIM)} {args.get('key', '')}\n"
            f"    {C.wrap('component:', C.DIM)} {args.get('component_name', '')}"
        )

    if tool_name == "get_full_openapi_schema":
        return f"    {C.wrap('📚', C.CYAN)} {C.wrap('Read full OpenAPI schema', C.BOLD)}"

    if tool_name == "execute_current_subtask":
        return ""

    return indent(_j(args), "    ")


def _fmt_tool_result(tool_name: str, result: dict[str, Any] | None) -> str | None:
    if not result:
        return None

    if result.get("error") not in (None, "", [], {}):
        return f"    {C.wrap('error:', C.DIM, C.RED)} {result['error']}"

    if result.get("error_message") not in (None, "", [], {}):
        return f"    {C.wrap('error:', C.DIM, C.RED)} {result['error_message']}"

    if result.get("errors") not in (None, "", [], {}):
        return f"    {C.wrap('errors:', C.DIM, C.RED)} {_short(result['errors'])}"

    if tool_name == "execute_current_subtask":
        lines: list[str] = []
        if result.get("record"):
            lines.append(f"    {C.wrap('record:', C.DIM)}")
            lines.append(indent(str(result["record"]), "      "))
        if result.get("action"):
            lines.append(f"    {C.wrap('action:', C.DIM)} {result['action']}")
        return "\n".join(lines) if lines else None

    if tool_name == "finish":
        lines: list[str] = []
        if "result" in result:
            lines.append(f"    {C.wrap('result:', C.DIM)} {_short(result['result'])}")
        if result.get("summary"):
            lines.append(f"    {C.wrap('summary:', C.DIM)}")
            lines.append(indent(_short_block(result["summary"]), "      "))
        return "\n".join(lines) if lines else None

    if tool_name == "read_file":
        payload = result.get("result")
        if payload is None:
            return None
        lines: list[str] = []
        meta = _fmt_fs_paging(result)
        if meta:
            lines.append(meta)
        lines.append(f"    {C.wrap('content:', C.DIM)}")
        lines.append(indent(_short_block(payload), "      "))
        return "\n".join(lines)

    if tool_name in {"ls", "glob", "grep"}:
        lines: list[str] = []
        meta = _fmt_fs_paging(result)
        if meta:
            lines.append(meta)
        payload = result.get("result")
        if payload not in (None, "", [], {}):
            lines.append(f"    {C.wrap('result:', C.DIM)}")
            if isinstance(payload, (dict, list)):
                lines.append(indent(_j(payload), "      "))
            else:
                lines.append(indent(_short_block(payload, 1200), "      "))
        return "\n".join(lines) if lines else None

    if tool_name == "interaction_stats":
        payload = result.get("result")
        if isinstance(payload, dict):
            return _render_kv_lines(
                [
                    ("path", payload.get("path")),
                    ("pattern", payload.get("pattern")),
                    ("total_files", payload.get("total_files")),
                    ("touched_files_count", payload.get("touched_files_count")),
                    ("untouched_files_count", payload.get("untouched_files_count")),
                    ("interaction_percent", f"{payload.get('interaction_percent')}%"),
                ]
            )
        return indent(_j(result), "    ")

    if tool_name in {"list_touched_files", "uncovered"}:
        lines: list[str] = []
        meta = _fmt_fs_paging(result)
        if meta:
            lines.append(meta)
        payload = result.get("result")
        if payload not in (None, "", [], {}):
            lines.append(f"    {C.wrap('files:', C.DIM)}")
            if isinstance(payload, (dict, list)):
                lines.append(indent(_j(payload), "      "))
            else:
                lines.append(indent(_short_block(payload, 1200), "      "))
        return "\n".join(lines) if lines else None

    if tool_name in {"list_tags", "list_memories", "search_memory", "read_memory"}:
        payload = result.get("result")
        if payload is None:
            return None
        if isinstance(payload, (dict, list)):
            return indent(_j(payload), "    ")
        return indent(_short_block(payload, 1200), "    ")

    if tool_name == "write_memory":
        return f"    {C.wrap('status:', C.DIM)} ok"

    if tool_name == "append_memory":
        payload = result.get("result")
        if payload is None:
            return f"    {C.wrap('status:', C.DIM)} ok"
        if isinstance(payload, (dict, list)):
            return indent(_j(payload), "    ")
        return indent(_short_block(payload, 1200), "    ")

    if tool_name == "code_execution_tool":
        lines: list[str] = []
        stdout = result.get("result")
        stderr = result.get("error")
        if stdout:
            lines.append(f"    {C.wrap('stdout:', C.DIM)}")
            lines.append(indent(_short_block(stdout, 1200), "      "))
        if stderr:
            lines.append(f"    {C.wrap('stderr:', C.DIM, C.RED)}")
            lines.append(indent(_short_block(stderr, 1200), "      "))
        return "\n".join(lines) if lines else None

    if tool_name in {
        "list_paths",
        "list_components",
        "list_servers",
        "get_info",
        "get_path",
        "get_component",
        "get_full_openapi_schema",
    }:
        payload = result.get("result")
        if payload is None:
            return None
        if isinstance(payload, list):
            return f"    {C.wrap('items:', C.DIM)} {len(payload)}\n" + indent(
                _j(payload), "    "
            )
        if isinstance(payload, dict):
            return indent(_j(payload), "    ")
        return indent(_short_block(payload, 1200), "    ")

    if tool_name in {
        "set_info",
        "add_server",
        "remove_server",
        "upsert_path",
        "remove_path",
        "upsert_component",
        "remove_component",
    }:
        payload = result.get("result")
        if payload is None:
            return f"    {C.wrap('status:', C.DIM)} ok"
        if isinstance(payload, dict):
            return f"    {C.wrap('diff:', C.DIM)}\n" + indent(_j(payload), "      ")
        return indent(_short_block(payload, 1200), "    ")

    if "result" in result:
        payload = result["result"]
        if isinstance(payload, (dict, list)):
            return indent(_j(payload), "    ")
        return indent(_short_block(payload, 1200), "    ")

    return indent(_j(result), "    ")


def _render_run_started(event: Any) -> str:
    total_tasks = event.payload.get("total_tasks")
    extra = f"\n{C.wrap(f'total tasks: {total_tasks}', C.DIM)}" if total_tasks else ""

    return (
        f"\n{C.wrap(_hr('═'), C.BLUE)}\n"
        f"{C.wrap('▶ Runner launch', C.BOLD, C.BLUE)}"
        f"{extra}\n"
        f"{C.wrap(_hr('═'), C.BLUE)}"
    )


def _render_task_started(event: Any) -> str:
    iterations = event.payload.get("iterations")
    max_attempts = event.payload.get("max_attempts")

    meta: list[str] = []
    if iterations is not None:
        meta.append(f"runs: {iterations}")
    if max_attempts is not None:
        meta.append(f"attempts: {max_attempts}")

    extra = f" {C.wrap('(' + ', '.join(meta) + ')', C.DIM)}" if meta else ""

    return (
        f"\n{C.wrap(_hr('═'), C.BLUE)}\n"
        f"{C.wrap(f'▶ Task #{event.task_id}: {event.task_name}', C.BOLD, C.BLUE)}{extra}\n"
        f"{C.wrap(_hr('═'), C.BLUE)}"
    )


def _render_iteration_started(event: Any) -> str:
    iteration = event.payload.get("iteration")
    objective = event.payload.get("objective", "")

    return (
        f"\n  {C.wrap(f'🔁 Iteration {iteration}', C.BOLD, C.YELLOW)}\n"
        f"    {C.wrap('objective:', C.DIM)} {_short(objective, 200)}"
    )


def _render_tool_call(event: Any) -> str:
    tool_name = event.payload["tool_name"]
    title = f"  {C.wrap('🛠', C.CYAN)} {C.wrap(tool_name, C.CYAN)}"
    body = _fmt_tool_args(tool_name, event.payload.get("tool_args"))
    return f"{title}\n{body}" if body else title


def _render_tool_result(event: Any) -> str | None:
    tool_name = event.payload["tool_name"]

    if tool_name in IGNORED_TOOL_RESULTS:
        return None

    if tool_name in LOW_SIGNAL_TOOL_RESULTS:
        return f"  {C.wrap('↩', C.GREEN)} {C.wrap(f'{tool_name} ok', C.GREEN)}"

    title = f"  {C.wrap('↩', C.GREEN)} {C.wrap(f'{tool_name} result', C.GREEN)}"
    body = _fmt_tool_result(tool_name, event.payload.get("result"))
    return f"{title}\n{body}" if body else title


def _render_tool_error(event: Any) -> str:
    return (
        f"\n  {C.wrap('✖ Tool error', C.BOLD, C.RED)}\n"
        f"    {C.wrap('tool:', C.DIM)} {event.payload.get('tool_name')}\n"
        f"    {C.wrap('error:', C.DIM)} {event.payload.get('error')}"
    )


def _render_final_text(event: Any) -> str | None:
    text = (event.payload.get("text") or "").strip()
    if not text:
        return None

    return (
        f"\n  {C.wrap('✅ Final answer', C.BOLD, C.GREEN)}\n"
        f"{indent(text, '     ')}"
    )


def _render_iteration_result(event: Any) -> str:
    iteration = event.payload.get("iteration")

    return (
        f"\n  {C.wrap(f'📌 Iteration result: {iteration}', C.BOLD, C.YELLOW)}\n"
        f"    {C.wrap('status:', C.DIM)} {event.payload.get('status')}\n"
        f"    {C.wrap('completed:', C.DIM)} {'yes' if event.payload.get('completed') else 'no'}\n"
        f"    {C.wrap('summary:', C.DIM)} {event.payload.get('summary') or '—'}"
    )


def _render_global_task_finished(event: Any) -> str:
    return (
        f"\n{C.wrap(_hr(), C.GREEN)}\n"
        f"{C.wrap(f'✓ Task finished: {event.task_name}', C.BOLD, C.GREEN)}\n"
        f"{C.wrap('  ' + (event.payload.get('summary') or '—'), C.DIM)}\n"
        f"{C.wrap(_hr(), C.GREEN)}"
    )


def _render_task_failed(event: Any) -> str:
    last_result = event.payload.get("last_result")
    lines = [
        f"\n{C.wrap(_hr('!'), C.RED)}",
        f"{C.wrap(f'✖ Task error #{event.task_id}: {event.task_name}', C.BOLD, C.RED)}",
    ]

    if last_result:
        lines.append(C.wrap(indent(_j(last_result), "  "), C.DIM))

    lines.append(C.wrap(_hr("!"), C.RED))
    return "\n".join(lines)


def _render_event(event: Any) -> str | None:
    if event.type == "run_started":
        return _render_run_started(event)

    if event.type == "task_started":
        return _render_task_started(event)

    if event.type == "iteration_started":
        return _render_iteration_started(event)

    if event.type == "tool_call":
        return _render_tool_call(event)

    if event.type == "tool_result":
        return _render_tool_result(event)

    if event.type == "tool_error":
        return _render_tool_error(event)

    if event.type == "final_text":
        return _render_final_text(event)

    if event.type == "iteration_result":
        return _render_iteration_result(event)

    if event.type == "global_task_finished":
        return _render_global_task_finished(event)

    if event.type == "task_failed":
        return _render_task_failed(event)

    return None


async def render_event(event: Any) -> None:
    rendered = _render_event(event)
    if rendered:
        print(rendered)