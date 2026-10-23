"""Download every tree-sitter parser the code-analysis Toolsets load.

tree-sitter-language-pack keeps parsers in the invoking user's cache, so run
this as the account that runs the Runtime (make runtime-venv for development,
deploy/ansible for a Runtime host). Without the cache, code-analysis@1 and
taint-annotations@1 fail their startup probes on an offline host.
"""

from __future__ import annotations

from tree_sitter_language_pack import prefetch

from contractor_runtime.toolsets.code_analysis.languages import PARSER_NAMES


def main() -> None:
    prefetch(sorted(set(PARSER_NAMES.values())))


if __name__ == "__main__":
    main()
