"""Prepare keep-id image layers without executing code or mounting host data.

Only this invocation's unstarted provisioning container is removed. A timeout
fails deployment and retains its unique name for operator reconciliation; no
Runtime owner, container or bind is inspected or changed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import uuid

from contractor_runtime.sandbox.podman.io import local_engine_environment
from contractor_runtime.sandbox.podman.settings import PodmanSettings


def warm(image: str) -> None:
    PodmanSettings(image=image)  # Validate with the deployed Runtime's grammar.
    name = "contractor-provision-" + uuid.uuid4().hex
    command = ["/usr/bin/podman", "--remote=false"]
    environment = local_engine_environment()
    try:
        created = subprocess.run(
            [
                *command,
                "create",
                "--name=" + name,
                "--label=io.contractor.provisioning=keep-id-image",
                "--pull=never",
                "--userns=keep-id",
                "--network=none",
                "--read-only",
                "--cap-drop=all",
                "--security-opt=no-new-privileges",
                "--image-volume=ignore",
                "--log-driver=none",
                "--restart=no",
                "--http-proxy=false",
                "--env-host=false",
                "--unsetenv-all",
                "--entrypoint=/usr/local/bin/python",
                image,
                "-I",
                "-c",
                "pass",
            ],
            env=environment,
            cwd="/",
            capture_output=True,
            timeout=300,
            check=True,
        )
        identity = created.stdout.decode("ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise ValueError("unexpected provisioning container identity")
        subprocess.run(
            [*command, "rm", identity],
            env=environment,
            cwd="/",
            capture_output=True,
            timeout=30,
            check=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RuntimeError(
            f"keep-id image preparation failed; reconcile unstarted container {name} "
            "before retrying"
        ) from None


if __name__ == "__main__":
    warm(sys.argv[1])
    print("keep-id image prepared; provisioning container removed")
