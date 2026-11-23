"""Audit the locked production Runtime dependency graph for published advisories.

This queries a live advisory service, so it runs in CI's advisory job rather
than in the deterministic release gate.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
# pip-audit and its whole dependency closure, pinned with hashes.
SCANNER_REQUIREMENTS = ROOT / "scripts/pip-audit-requirements.txt"
# A pin with long-published advisories (PYSEC-2021-108 among others). The
# scanner must report it before a clean result for the Runtime is trusted.
KNOWN_VULNERABLE_PIN = "urllib3==1.26.4"
KNOWN_VULNERABLE = (
    f"{KNOWN_VULNERABLE_PIN} \\\n"
    "    --hash=sha256:2f4da4594db7e1e110a944bb1b551fdf4e6c136ad42e4234131391e21eb5b0df \\\n"
    "    --hash=sha256:e7b021f7241115872f92f43c6508082facffbd1c048e3c6e2bb9c2a157e28937\n"
)


def install_scanner(directory: Path) -> list[str]:
    """Install the hashed scanner outside the shipped Runtime environment."""
    environment = directory / "scanner"
    subprocess.run(["uv", "venv", "--quiet", "--python", sys.executable, str(environment)], check=True)
    subprocess.run(
        [
            "uv", "pip", "install", "--quiet", "--require-hashes",
            "--python", str(environment / "bin" / "python"),
            "-r", str(SCANNER_REQUIREMENTS),
        ],
        check=True,
    )
    return [str(environment / "bin" / "pip-audit")]


def export_runtime_lock(path: Path) -> None:
    # --locked fails when uv.lock no longer matches runtime/pyproject.toml.
    # The export stays out of CI logs; only advisories are printed.
    subprocess.run(
        ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--output-file", str(path)],
        cwd=RUNTIME,
        stdout=subprocess.DEVNULL,
        check=True,
    )


def scan(scanner: list[str], requirements: Path) -> dict[str, list[str]]:
    """Return the advisory IDs pip-audit reports per pinned dependency."""
    result = subprocess.run(
        [
            *scanner, "--require-hashes", "--disable-pip", "--strict",
            "--progress-spinner", "off", "--format", "json", "-r", str(requirements),
        ],
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(result.stdout)
    except ValueError:
        raise SystemExit(f"pip-audit failed (exit {result.returncode}):\n{result.stderr}") from None
    found = {
        f"{dependency['name']}=={dependency['version']}": list(
            dict.fromkeys(vulnerability["id"] for vulnerability in dependency["vulns"])
        )
        for dependency in report.get("dependencies", [])
        if dependency.get("vulns")
    }
    if result.returncode != (1 if found else 0):
        raise SystemExit(f"pip-audit exit {result.returncode} disagrees with its report:\n{result.stderr}")
    return found


def audit(scanner: list[str], requirements: Path, directory: Path) -> None:
    canary = directory / "known-vulnerable.txt"
    canary.write_text(KNOWN_VULNERABLE)
    if KNOWN_VULNERABLE_PIN not in scan(scanner, canary):
        raise SystemExit(f"pip-audit reported no advisory for {KNOWN_VULNERABLE_PIN}; its results cannot be trusted")
    found = scan(scanner, requirements)
    for dependency, identifiers in sorted(found.items()):
        print(f"{dependency}: {', '.join(identifiers)}", file=sys.stderr)
    if found:
        raise SystemExit(f"{len(found)} production Runtime dependencies have published advisories")
    print("The production Runtime lock has no published advisories.")


def main() -> None:
    # The context removes the scanner and the exported lock on any outcome.
    with tempfile.TemporaryDirectory(prefix="runtime-audit-") as name:
        directory = Path(name)
        scanner = install_scanner(directory)
        requirements = directory / "runtime-requirements.txt"
        export_runtime_lock(requirements)
        audit(scanner, requirements, directory)


if __name__ == "__main__":
    main()
