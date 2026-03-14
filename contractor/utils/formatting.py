import json
import sys
from textwrap import indent
from typing import Any


IGNORED_TOOLS = (
    "glob",
    "read_file",
    "grep",
    "ls",
    "add_subtask",
    "decompose_subtask",
    "skip",
)


def make_jsonable(value):
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


def _render_subtasks(subtasks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, subtask in enumerate(subtasks, start=1):
        title = subtask.get("title", "Без названия")
        description = subtask.get("description", "")
        task_id = subtask.get("task_id")
        status = subtask.get("status")

        head = f"      {C.wrap(f'{idx}.', C.MAGENTA)} {C.wrap(title, C.BOLD)}"
        meta = []
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


def _fmt_tool_args(tool_name: str, args: dict[str, Any] | None) -> str:
    if not args:
        return ""

    if tool_name == "add_subtask":
        title = args.get("title", "Без названия")
        description = args.get("description", "")
        lines = [f"    {C.wrap('•', C.CYAN)} {C.wrap(title, C.BOLD)}"]
        if description:
            lines.append(C.wrap(indent(description, "      "), C.DIM))
        return "\n".join(lines)

    if tool_name == "decompose_subtask":
        task_id = args.get("task_id")
        subtasks = (args.get("decomposition") or {}).get("subtasks") or []
        lines = [
            f"    {C.wrap('↳', C.MAGENTA)} "
            f"{C.wrap(f'Декомпозиция подзадачи {task_id}', C.BOLD)} "
            f"{C.wrap(f'({len(subtasks)} шт.)', C.DIM)}"
        ]
        body = _render_subtasks(subtasks)
        if body:
            lines.append(body)
        return "\n".join(lines)

    if tool_name == "skip":
        task_id = args.get("task_id")
        reason = args.get("reason", "")
        lines = [
            f"    {C.wrap('⤼', C.YELLOW)} {C.wrap(f'Пропуск подзадачи {task_id}', C.BOLD)}"
        ]
        if reason:
            lines.append(f"    {C.wrap('reason:', C.DIM)} {reason}")
        return "\n".join(lines)

    if tool_name == "finish":
        status = args.get("status", "unknown")
        result = args.get("result", "")
        lines = [
            f"    {C.wrap('🏁', C.GREEN)} {C.wrap('Завершение глобальной задачи', C.BOLD)}",
            f"    {C.wrap('status:', C.DIM)} {status}",
        ]
        if result:
            lines.append(f"    {C.wrap('result:', C.DIM)} {_short(result)}")
        return "\n".join(lines)

    if tool_name == "write_memory":
        lines = [
            f"    {C.wrap('🧠', C.CYAN)} {C.wrap(args.get('name', 'Без названия'), C.BOLD)}"
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
            f"{C.wrap(f'Дополнить memory {name}', C.BOLD)}\n"
            f"    {C.wrap('text:', C.DIM)} {_short(text)}"
        )

    if tool_name == "read_memory":
        name = args.get("name")
        return (
            f"    {C.wrap('📖', C.CYAN)} {C.wrap(f'Прочитать memory {name}', C.BOLD)}"
        )

    if tool_name == "search_memory":
        tags = args.get("tags") or []
        return f"    {C.wrap('🔎', C.CYAN)} {C.wrap('Поиск memory', C.BOLD)}\n    {C.wrap('tags:', C.DIM)} {', '.join(map(str, tags)) if tags else '—'}"

    if tool_name == "list_tags":
        return f"    {C.wrap('🏷', C.CYAN)} {C.wrap('Показать все теги', C.BOLD)}"

    if tool_name == "list_memories":
        return f"    {C.wrap('🗃', C.CYAN)} {C.wrap('Показать все memories', C.BOLD)}"

    if tool_name == "ls":
        return f"    {C.wrap('🔎', C.CYAN)} {C.wrap('Список файлов в директории:', C.BOLD)} {args.get('path', '/')}\n"

    if tool_name == "glob":
        return f"    {C.wrap('🔎', C.CYAN)} {C.wrap('Поиск паттернов:', C.BOLD)} {args.get('pattern', '')} в {args.get('path', '/')}\n"

    if tool_name == "read_file":
        return f"    {C.wrap('🔎', C.CYAN)} {C.wrap('Чтение файла:', C.BOLD)} {args.get('file', '')}\n"

    if tool_name in {
        "execute_current_subtask",
        "get_current_subtask",
        "list_subtasks",
        "get_records",
    }:
        return ""

    return indent(_j(args), "    ")


def _fmt_tool_result(tool_name: str, result: dict[str, Any] | None) -> str | None:
    if not result:
        return None

    if "error" in result:
        return f"    {C.wrap('error:', C.DIM, C.RED)} {result['error']}"

    if tool_name == "execute_current_subtask":
        lines: list[str] = []
        if result.get("record"):
            lines.append(f"    {C.wrap('record:', C.DIM)}")
            lines.append(indent(str(result["record"]), "      "))
        if result.get("action"):
            lines.append(f"    {C.wrap('action:', C.DIM)} {result['action']}")
        if result.get("error"):
            lines.append(f"    {C.wrap('error:', C.DIM, C.RED)} {result['error']}")
        return "\n".join(lines) if lines else None

    if tool_name == "finish":
        lines: list[str] = []
        if "result" in result:
            lines.append(f"    {C.wrap('result:', C.DIM)} {result['result']}")
        if result.get("summary"):
            lines.append(f"    {C.wrap('summary:', C.DIM)}")
            lines.append(indent(str(result["summary"]), "      "))
        return "\n".join(lines) if lines else None

    if "result" in result:
        payload = result["result"]
        if isinstance(payload, (dict, list)):
            return indent(_j(payload), "    ")
        return indent(str(payload), "    ")

    return indent(_j(result), "    ")


def _render_event(event) -> str | None:
    if event.type == "run_started":
        return (
            f"\n{C.wrap(_hr('═'), C.BLUE)}\n"
            f"{C.wrap('▶ Запуск runner', C.BOLD, C.BLUE)}\n"
            f"{C.wrap(_hr('═'), C.BLUE)}"
        )

    if event.type == "task_started":
        max_iterations = event.payload.get("max_iterations")
        extra = (
            f" {C.wrap(f'(итераций: {max_iterations})', C.DIM)}"
            if max_iterations
            else ""
        )
        return (
            f"\n{C.wrap(_hr('═'), C.BLUE)}\n"
            f"{C.wrap(f'▶ Задача #{event.task_id}: {event.task_name}', C.BOLD, C.BLUE)}{extra}\n"
            f"{C.wrap(_hr('═'), C.BLUE)}"
        )

    if event.type == "iteration_started":
        iteration = event.payload.get("iteration")
        objective = event.payload.get("objective", "")
        return (
            f"\n  {C.wrap(f'🔁 Итерация {iteration}', C.BOLD, C.YELLOW)}\n"
            f"    {C.wrap('objective:', C.DIM)} {_short(objective, 200)}"
        )

    if event.type == "tool_call":
        tool_name = event.payload["tool_name"]
        title = f"  {C.wrap('🛠', C.CYAN)} {C.wrap(tool_name, C.CYAN)}"
        body = _fmt_tool_args(tool_name, event.payload.get("tool_args"))
        return f"{title}\n{body}" if body else title

    if event.type == "tool_result":
        tool_name = event.payload["tool_name"]
        if tool_name in IGNORED_TOOLS:
            return

        title = f"  {C.wrap('↩', C.GREEN)} {C.wrap(f'{tool_name} result', C.GREEN)}"
        body = _fmt_tool_result(tool_name, event.payload.get("result"))
        return f"{title}\n{body}" if body else title

    if event.type == "tool_error":
        return (
            f"\n  {C.wrap('✖ Ошибка tool', C.BOLD, C.RED)}\n"
            f"    {C.wrap('tool:', C.DIM)} {event.payload.get('tool_name')}\n"
            f"    {C.wrap('error:', C.DIM)} {event.payload.get('error')}"
        )

    if event.type == "final_text":
        text = (event.payload.get("text") or "").strip()
        if not text:
            return None
        return (
            f"\n  {C.wrap('✅ Финальный ответ', C.BOLD, C.GREEN)}\n"
            f"{indent(text, '     ')}"
        )

    if event.type == "iteration_result":
        it = event.payload.get("iteration")
        return (
            f"\n  {C.wrap(f'📌 Итог итерации {it}', C.BOLD, C.YELLOW)}\n"
            f"    {C.wrap('status:', C.DIM)} {event.payload.get('status')}\n"
            f"    {C.wrap('completed:', C.DIM)} {'yes' if event.payload.get('completed') else 'no'}\n"
            f"    {C.wrap('summary:', C.DIM)} {event.payload.get('summary') or '—'}"
        )

    if event.type == "global_task_finished":
        return (
            f"\n{C.wrap(_hr(), C.GREEN)}\n"
            f"{C.wrap(f'✓ Завершена задача: {event.task_name}', C.BOLD, C.GREEN)}\n"
            f"{C.wrap('  ' + (event.payload.get('summary') or '—'), C.DIM)}\n"
            f"{C.wrap(_hr(), C.GREEN)}"
        )

    if event.type == "task_failed":
        last_result = event.payload.get("last_result")
        lines = [
            f"\n{C.wrap(_hr('!'), C.RED)}",
            f"{C.wrap(f'✖ Ошибка в задаче #{event.task_id}: {event.task_name}', C.BOLD, C.RED)}",
        ]
        if last_result:
            lines.append(C.wrap(indent(_j(last_result), "  "), C.DIM))
        lines.append(C.wrap(_hr("!"), C.RED))
        return "\n".join(lines)

    return None


async def handle_event(event) -> None:
    rendered = _render_event(event)
    if rendered:
        print(rendered)
