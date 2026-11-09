"""Shared source language detection and offline Tree-sitter parser selection."""

from collections.abc import Iterable
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType

from tree_sitter import Parser
from tree_sitter_language_pack import get_parser


class Language(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    KOTLIN = "kotlin"
    C = "c"
    CPP = "cpp"
    C_SHARP = "c_sharp"
    RUBY = "ruby"
    PHP = "php"
    SCALA = "scala"
    SWIFT = "swift"
    LUA = "lua"
    ELIXIR = "elixir"
    HASKELL = "haskell"
    BASH = "bash"


PARSER_NAMES = MappingProxyType(
    {
        language: "csharp" if language is Language.C_SHARP else language.value
        for language in Language
    }
)

EXTENSION_LANGUAGES = MappingProxyType(
    {
        ".py": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".cjs": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TSX,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".java": Language.JAVA,
        ".kt": Language.KOTLIN,
        ".kts": Language.KOTLIN,
        ".c": Language.C,
        ".h": Language.C,
        ".cpp": Language.CPP,
        ".cc": Language.CPP,
        ".cxx": Language.CPP,
        ".hpp": Language.CPP,
        ".hh": Language.CPP,
        ".hxx": Language.CPP,
        ".cs": Language.C_SHARP,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".scala": Language.SCALA,
        ".sc": Language.SCALA,
        ".swift": Language.SWIFT,
        ".lua": Language.LUA,
        ".ex": Language.ELIXIR,
        ".exs": Language.ELIXIR,
        ".hs": Language.HASKELL,
        ".lhs": Language.HASKELL,
        ".sh": Language.BASH,
        ".bash": Language.BASH,
    }
)

# A ".h" header is valid C and C++ alike; header_language() picks one grammar
# for all of a snapshot's headers from its unambiguous C and C++ files. Those
# are derived from the shallow mapping, so every extension that votes for a
# grammar is also parsed with it.
HEADER_EXTENSION = ".h"
C_SOURCE_EXTENSIONS = frozenset(
    extension
    for extension, language in EXTENSION_LANGUAGES.items()
    if language is Language.C and extension != HEADER_EXTENSION
)
CPP_SOURCE_EXTENSIONS = frozenset(
    extension for extension, language in EXTENSION_LANGUAGES.items() if language is Language.CPP
)


def detect_language(path: str, *, header: Language) -> Language | None:
    """Return the shallow language of ``path``; a ``.h`` header uses ``header``.

    Suffixes match case-insensitively, so ``Tool.PY`` is Python source.
    """

    suffix = PurePosixPath(path).suffix.lower()
    if suffix == HEADER_EXTENSION:
        return header
    return EXTENSION_LANGUAGES.get(suffix)


def is_header(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() == HEADER_EXTENSION


def header_language(paths: Iterable[str]) -> Language:
    """Return the one grammar for every ``.h`` header of a snapshot.

    Shallow analysis, graph admission and taint annotation all call this with
    the snapshot's managed text paths, so they agree on every header. Headers
    are C++ when the snapshot has at least as many unambiguous C++ files as C
    sources, so a C project with a vendored C++ file keeps C. Ties favor C++:
    its grammar parses nearly every C header, while the C grammar rejects
    every class, namespace, template and reference. Without either kind of
    file, headers are C.
    """

    c_files = 0
    cpp_files = 0
    for path in paths:
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in C_SOURCE_EXTENSIONS:
            c_files += 1
        elif suffix in CPP_SOURCE_EXTENSIONS:
            cpp_files += 1
    return Language.CPP if cpp_files and cpp_files >= c_files else Language.C


def load_parser(language: Language) -> Parser:
    return get_parser(PARSER_NAMES[language])


def probe_all_parsers() -> bool:
    try:
        for language in Language:
            tree = load_parser(language).parse(b"")
            if tree.root_node is None:
                return False
        return True
    except Exception:
        return False
