import os
import shutil
import subprocess
from typing import Optional, Callable


class PodmanNotFoundException(Exception):
    def __init__(self) -> None:
        super().__init__("podman not found!")


class PodmanMountException(Exception):
    def __init__(self, mount: str) -> None:
        super().__init__(f"mounted directory {mount} does not exist or is not a directory")


class PodmanImageNotFoundException(Exception):
    def __init__(self, image: str) -> None:
        super().__init__(f"image {image} not found in registry (pull failed)")


class PodmanContainer:
    def __init__(
        self,
        image: str,
        mounts: list[str],
        commands: Optional[list[str]] = None,
        ro_mode: bool = False,
        workdir: str = "/workdir",
    ):
        """
        Args:
            image: container image from registry (e.g. "docker.io/library/alpine:latest")
            mounts: host directories to mount into the container under /workdir/<basename>
            commands: allow-list of commands executable inside the container; None means allow all
            ro_mode: enables read-only container mode (and mounts are mounted read-only)
            workdir: container working directory (default: /workdir)
        """
        self.image: str = image
        self.mounts: list[str] = mounts
        self.commands: Optional[list[str]] = commands
        self.ro_mode: bool = ro_mode
        self.workdir: str = workdir
        self.container_id: Optional[str] = None

    def _ensure_podman(self) -> None:
        """Checks that podman is available."""
        if not shutil.which("podman"):
            raise PodmanNotFoundException()

    def _ensure_image(self) -> None:
        """
        Checks if image exists locally. Pull image from registry otherwise.
        Raises PodmanImageNotFoundException if pulling fails.
        """
        exists = subprocess.run(
            ["podman", "image", "exists", self.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if exists.returncode == 0:
            return

        pull = subprocess.run(
            ["podman", "pull", self.image],
            capture_output=True,
            text=True,
        )
        if pull.returncode != 0:
            raise PodmanImageNotFoundException(self.image)

    def _ensure_mounts(self) -> None:
        """Checks if all mounts are directories and exist."""
        for m in self.mounts:
            if not os.path.isdir(m):
                raise PodmanMountException(m)

    def _run_container(self, ro_mode: bool) -> str:
        """
        Run container without persistence (--rm) in detached mode.
        Sets workdir to /workdir and mounts all directories under /workdir/<basename>.
        """
        cmd: list[str] = [
            "podman",
            "run",
            "-d",
            "--rm",
            "--workdir",
            self.workdir,
        ]

        if ro_mode:
            cmd.append("--read-only")

        for host_path in self.mounts:
            host_abs = os.path.abspath(host_path)
            mount_name = os.path.basename(host_abs)
            container_path = f"{self.workdir.rstrip('/')}/{mount_name}"

            volume = f"{host_abs}:{container_path}"
            if ro_mode:
                volume += ":ro"

            cmd += ["-v", volume]

        # Keep container alive so `podman exec` can run commands later
        cmd += [self.image, "sleep", "30m"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"failed to run container: {result.stderr.strip()}")

        return result.stdout.strip()

    def _check_command_available(self, command: str) -> bool:
        """
        If commands allow-list is provided, only allow executing those commands (by first token).
        """
        if not command or not command.strip():
            return False

        if self.commands is None:
            return True

        first = command.strip().split()[0]
        return first in self.commands

    def start(self) -> None:
        self._ensure_podman()
        self._ensure_image()
        self._ensure_mounts()
        self.container_id = self._run_container(self.ro_mode)

    def execute(self, command: str) -> tuple[str, str]:
        if self.container_id is None:
            self.start()

        if not self._check_command_available(command):
            return "", f"command {command} is not available to call"

        args = command.strip().split()
        result = subprocess.run(
            ["podman", "exec", self.container_id[:12], *args],
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr

    def tools(self) -> list[Callable[[str], dict[str, str]]]:
        if self.container_id is None:
            self.start()

        def code_execution_tool(command: str) -> dict[str, str]:
            """
            code_execution_tool: tool to execute arbitraty system command
            
            Args:
                command: arbitrary system comand (i.e. "ls -la")
            Returns:
                execution results
            """

            if not self._check_command_available(command):
                return {"result": "", "error": f"command {command} is not available to call"}

            out, err = self.execute(command)
            return {"result": out, "error": err}

        return [code_execution_tool]

    def stop(self) -> None:
        """Stops the running container (if any)."""
        if not self.container_id:
            return
        subprocess.run(
            ["podman", "stop", self.container_id[:12]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.container_id = None
