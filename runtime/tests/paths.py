"""Where the tests are and where the repository is, resolved once.

Test modules used to spell these as index arithmetic on ``__file__``, which
silently tracked how deep the file sat: a test one directory further down
made ``parents[2]`` point at ``runtime/`` instead of the repository root.
Importing these constants keeps that arithmetic in one place, so tests can
be grouped into directories without every path breaking.
"""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
"""The runtime/tests directory, which holds fakes/ and fixtures/."""

REPOSITORY_ROOT = TESTS_ROOT.parents[1]
"""The repository root, which holds api/, internal/, testdata/ and deploy/."""
