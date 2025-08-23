import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any


def ensure_vacuum() -> str | None:
    if shutil.which("vacuum") is None:
        if shutil.which("go") is None:
            return "golang needs to be installed to use this tool"

        return "vacuum needs to be installed to use this tool"


def vacuum_lint_tool(file_path: str) -> dict:
    """Lint OpenAPI Schema
    Args:
        file_path: str
            Absolute path to OpenAPI schema
    Returns:
        dict: A dictionary containing linting results in 'results' key with discovered proplems if successful, or an 'error_message' if an error occurred.
    """

    if err := ensure_vacuum() is not None:
        return {"error": err}

    if not os.path.exists(file_path):
        return {"error": f"path {file_path} does not exists"}

    output = subprocess.run(
        ["vacuum", "spectral-report", "-q", "-o", file_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if err := output.stderr.encode("utf-8") != "":
        return {"error": err}

    return {"results": output.encode("utf-8")}


def vacuum_security_checks(file_path: str) -> dict:
    return dict()
