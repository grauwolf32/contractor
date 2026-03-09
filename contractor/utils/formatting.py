import sys
import json
from typing import Any
from textwrap import indent


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


def _fmt_tool_args(tool_name: str, args: dict[str, Any] | None) -> str:
    if not args:
        return ""

    if tool_name == "add_subtask":
        title = args.get("title", "Без названия")
        description = args.get("description", "")
        lines = [
            f"    {C.wrap('•', C.CYAN)} {C.wrap(title, C.BOLD)}",
        ]
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
        for idx, subtask in enumerate(subtasks, start=1):
            title = subtask.get("title", "Без названия")
            description = subtask.get("description", "")
            lines.append(f"      {C.wrap(f'{idx}.', C.MAGENTA)} {title}")
            if description:
                lines.append(C.wrap(indent(description, "         "), C.DIM))
        return "\n".join(lines)

    if tool_name == "read_memory":
        name = args.get("name")
        if name:
            return f"    {C.wrap('memory:', C.DIM)} {name}"
        return ""

    if tool_name in {
        "execute_current_subtask",
        "get_current_subtask",
        "list_subtasks",
        "list_memories",
    }:
        return ""

    return indent(_j(args), "    ")


def _render_event(event) -> str | None:
    if event.type == "task_started":
        return (
            f"\n{C.wrap(_hr('═'), C.BLUE)}\n"
            f"{C.wrap(f'▶ Задача #{event.task_id}: {event.task_name}', C.BOLD, C.BLUE)}\n"
            f"{C.wrap(_hr('═'), C.BLUE)}"
        )

    if event.type == "tool_call":
        tool_name = event.payload["tool_name"]
        body = _fmt_tool_args(tool_name, event.payload.get("tool_args"))
        title = f"  {C.wrap('🛠', C.CYAN)} {C.wrap(tool_name, C.CYAN)}"
        return f"{title}\n{body}" if body else title

    if event.type == "final_text":
        text = (event.payload.get("text") or "").strip()
        if not text:
            return None
        return (
            f"\n  {C.wrap('✅ Финальный ответ', C.BOLD, C.GREEN)}\n"
            f"{indent(text, '     ')}"
        )

    if event.type == "iteration_result":
        summary = event.payload.get("summary") or "—"
        status = event.payload.get("status") or "unknown"
        completed = "yes" if event.payload.get("completed") else "no"
        iteration = event.payload.get("iteration")
        return (
            f"\n  {C.wrap(f'📌 Итерация {iteration}', C.BOLD, C.YELLOW)}\n"
            f"    {C.wrap('status:', C.DIM)} {status}\n"
            f"    {C.wrap('completed:', C.DIM)} {completed}\n"
            f"    {C.wrap('summary:', C.DIM)} {summary}"
        )

    if event.type == "global_task_finished":
        summary = event.payload.get("summary") or "—"
        return (
            f"\n{C.wrap(_hr(), C.GREEN)}\n"
            f"{C.wrap(f'✓ Завершена задача: {event.task_name}', C.BOLD, C.GREEN)}\n"
            f"{C.wrap('  ' + summary, C.DIM)}\n"
            f"{C.wrap(_hr(), C.GREEN)}"
        )

    if event.type == "task_failed":
        return (
            f"\n{C.wrap(_hr('!'), C.RED)}\n"
            f"{C.wrap(f'✖ Ошибка в задаче #{event.task_id}: {event.task_name}', C.BOLD, C.RED)}\n"
            f"{C.wrap(_hr('!'), C.RED)}"
        )

    return None


async def handle_event(event) -> None:
    rendered = _render_event(event)
    if rendered:
        print(rendered)
