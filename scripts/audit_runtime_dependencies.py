"""Audit the frozen production Runtime dependency graph."""

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
SCANNER = "pip-audit==2.10.1"


def main() -> None:
    # Keep the scanner outside the shipped Runtime venv and keep the complete
    # export out of CI logs. The context removes the temporary requirements file
    # on both success and failure.
    with tempfile.NamedTemporaryFile() as requirements:
        subprocess.run(
            [
                "uv",
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                requirements.name,
            ],
            cwd=RUNTIME,
            stdout=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            [
                "uvx",
                "--from",
                SCANNER,
                "pip-audit",
                "--require-hashes",
                "--disable-pip",
                "-r",
                requirements.name,
            ],
            cwd=RUNTIME,
            check=True,
        )


if __name__ == "__main__":
    main()
