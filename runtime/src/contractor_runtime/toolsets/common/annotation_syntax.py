"""Canonical taint comment syntax shared by insertion and trace validation."""

import re
from types import MappingProxyType

from contractor_runtime.toolsets.common.source_languages import Language

MAX_TOKEN_CHARS = 128
MAX_LIST_ENTRIES = 32
MAX_ANNOTATION_BYTES = 4096


class AnnotationSyntaxError(ValueError):
    def __init__(self, *, capacity: bool = False):
        super().__init__("invalid annotation syntax")
        self.capacity = capacity


_ARG_STATES = frozenset({"tainted", "validated", "clean", "derived"})
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.$:/{}\[\]*?+\-<>]+\Z")
_ARGUMENT_PATTERN = re.compile(r"[A-Za-z0-9_.$\[\]*?+\-<>]+\Z")
_COMMENT_MARKERS = MappingProxyType(
    {
        Language.PYTHON: "#",
        Language.RUBY: "#",
        Language.BASH: "#",
        Language.ELIXIR: "#",
        Language.HASKELL: "--",
        Language.LUA: "--",
    }
)


def annotation_token(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TOKEN_CHARS
        or not value.isascii()
        or _TOKEN_PATTERN.fullmatch(value) is None
    ):
        raise AnnotationSyntaxError()
    return value


def argument_token(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TOKEN_CHARS
        or not value.isascii()
        or _ARGUMENT_PATTERN.fullmatch(value) is None
    ):
        raise AnnotationSyntaxError()
    return value


def annotation_args(value: str) -> str:
    if not isinstance(value, str):
        raise AnnotationSyntaxError()
    if not value:
        return ""
    result: list[str] = []
    names: set[str] = set()
    chunks = value.split(",")
    if len(chunks) > MAX_LIST_ENTRIES:
        raise AnnotationSyntaxError(capacity=True)
    for chunk in chunks:
        parts = chunk.strip().split(":")
        if len(parts) != 2:
            raise AnnotationSyntaxError()
        name = argument_token(parts[0].strip())
        resources = parts[1].strip()
        if resources not in _ARG_STATES or name in names:
            raise AnnotationSyntaxError()
        names.add(name)
        result.append(f"{name}:{resources}")
    return ",".join(result)


def annotation_calls(value: str) -> str:
    if not isinstance(value, str):
        raise AnnotationSyntaxError()
    if not value:
        return ""
    chunks = value.split(",")
    if len(chunks) > MAX_LIST_ENTRIES:
        raise AnnotationSyntaxError(capacity=True)
    result: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        item = annotation_token(chunk.strip())
        if item in seen:
            raise AnnotationSyntaxError()
        seen.add(item)
        result.append(item)
    return ",".join(result)


def comment_marker(language: Language) -> str:
    return _COMMENT_MARKERS.get(language, "//")


def canonical_annotation(value: str, indent: str, language: Language, line: int) -> bool:
    prefix = f"{indent}{comment_marker(language)} @"
    if not value.startswith(prefix) or len(value.encode()) > MAX_ANNOTATION_BYTES:
        return False
    if language is Language.PHP and "?>" in value:
        return False
    if (
        language is Language.PYTHON
        and line <= 2
        and re.search(r"coding[:=][ \t]*[-\w.]+", value, flags=re.ASCII)
    ):
        return False
    body = value[len(prefix) :]
    try:
        if match := re.fullmatch(
            r"trace target=([^ ]+)(?: args=([^ ]+))?(?: calls=([^ ]+))?", body
        ):
            target, args, calls = match.groups()
            return (
                annotation_token(target) == target
                and (args is None or annotation_args(args) == args)
                and (calls is None or annotation_calls(calls) == calls)
            )
        if match := re.fullmatch(r"validate arg=([^ ]+) kind=([^ ]+)", body):
            return bool(argument_token(match[1]) and annotation_token(match[2]))
        if match := re.fullmatch(r"sink kind=([^ ]+) arg=([^ ]+)", body):
            return bool(annotation_token(match[1]) and argument_token(match[2]))
    except AnnotationSyntaxError:
        return False
    return False
