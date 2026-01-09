import os
import shutil
import subprocess
from typing import Optional, Callable


class PodmanNotFoundException(Exception):
    def __init__(self) -> None:
        super().__init__("podman not found!")


class PodmanMountException(Exception):
    def __init__(self, mount: str) -> None:
        super().__init__(
            f"mounted directory {mount} does not exist or is not a directory"
        )


class PodmanImageNotFoundException(Exception):
    def __init__(self, image: str) -> None:
        super().__init__(f"image {image} not found in registry (pull failed)")


class PodmanContainer:
    def __init__(
        self,
        name: str,
        image: str,
        mounts: list[str],
        commands: Optional[list[str]] = None,
        ro_mode: bool = False,
        workdir: str = "/mnt",
    ):
        """
        Args:
            name: container unique name (podman --name)
            image: container image from registry (e.g. "docker.io/library/alpine:latest")
            mounts: host directories to mount into the container under /mnt/<basename>
            commands: allow-list of commands executable inside the container; None means allow all
            ro_mode: enables read-only container mode (and mounts are mounted read-only)
            workdir: container working directory (default: /mnt)
        """
        self.name: str = name
        self.image: str = image
        self.mounts: list[str] = mounts
        self.commands: Optional[list[str]] = commands
        self.ro_mode: bool = ro_mode
        self.workdir: str = workdir
        self.container_id: Optional[str] = None

    # ----------------- checks / ensures -----------------

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
            ["podman", "pull", self.image], capture_output=True, text=True
        )
        if pull.returncode != 0:
            raise PodmanImageNotFoundException(self.image)

    def _ensure_mounts(self) -> None:
        """Checks if all mounts are directories and exist."""
        for m in self.mounts:
            if not os.path.isdir(m):
                raise PodmanMountException(m)

    # ----------------- container lifecycle helpers -----------------

    def _remove_container_by_name_if_exists(self) -> None:
        """
        Remove a container by name if it exists (running or not). This is a safety net:
        - prevents 'name already in use' issues
        - also cleans up containers created without --rm
        """
        subprocess.run(
            ["podman", "rm", "-f", self.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _get_running_container_id(self) -> Optional[str]:
        """Returns the ID of a *running* container that matches self.name, or None."""
        r = subprocess.run(
            ["podman", "ps", "--filter", f"name={self.name}", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return None

        ids = [line.strip() for line in r.stdout.splitlines() if line.strip()]
        return ids[0] if ids else None

    def _check_container_running(self) -> bool:
        """Checks whether self.container_id exists and is in Running state."""
        if self.container_id is None:
            return False

        r = subprocess.run(
            ["podman", "inspect", "-f", "{{.State.Running}}", self.container_id[:12]],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return False

        return r.stdout.strip().lower() == "true"

    def _run_container(self, ro_mode: bool) -> str:
        """
        Run container without persistence (--rm) in detached mode.
        Sets workdir to self.workdir and mounts all directories under self.workdir/<basename>.
        Keeps container alive for 30 minutes.
        Returns:
            container_id
        """
        cmd: list[str] = [
            "podman",
            "run",
            "-d",
            "--rm",
            "--name",
            self.name,
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

        # Keep container alive so `podman exec` can run commands later (30 minutes)
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

    def _ensure_container_running(self) -> None:
        """
        Ensures there is a running container and self.container_id is set.
        Order:
          1) if current self.container_id is running -> ok
          2) else if a container with self.name is already running -> reuse its id
          3) else remove any leftover container with same name and start a new one
        """
        if self._check_container_running():
            return

        if running_id := self._get_running_container_id():
            self.container_id = running_id
            return

        # container not running -> start fresh (avoid name conflicts)
        self._remove_container_by_name_if_exists()
        self.container_id = self._run_container(self.ro_mode)

    # ----------------- public API -----------------

    def start(self) -> None:
        self._ensure_podman()
        self._ensure_image()
        self._ensure_mounts()
        self._ensure_container_running()

    def execute(self, command: str) -> tuple[str, str]:
        """
        Execute a command inside the container.
        If the container is dead/expired, it will be (re)started automatically.
        """
        self._ensure_podman()

        if not self._check_command_available(command):
            return "", f"command {command} is not available to call"

        # If container is dead or missing, bring it back before exec
        if self.container_id is None or not self._check_container_running():
            # Ensure deps are ready before (re)run
            self._ensure_image()
            self._ensure_mounts()
            self._ensure_container_running()

        args = command.strip().split()
        result = subprocess.run(
            ["podman", "exec", self.container_id[:12], *args],
            capture_output=True,
            text=True,
        )

        # If exec failed because container died between checks, retry once.
        if result.returncode != 0 and (
            "no such container" in (result.stderr or "").lower()
        ):
            self.container_id = None
            self._ensure_image()
            self._ensure_mounts()
            self._ensure_container_running()
            result = subprocess.run(
                ["podman", "exec", self.container_id[:12], *args],
                capture_output=True,
                text=True,
            )

        return result.stdout, result.stderr

    def tools(self) -> list[Callable[[str], dict[str, str]]]:
        """
        Returns a list of callable tools.
        Ensures container is running; if it dies later, execution will restart it.
        """
        self.start()

        def code_execution_tool(command: str) -> dict[str, str]:
            """
            code_execution_tool: tool to execute arbitrary system command inside the container

            Args:
                command: arbitrary system command (i.e. "ls -la")
            Returns:
                {"result": <stdout>, "error": <stderr or validation error>}
            """
            if not self._check_command_available(command):
                return {
                    "result": "",
                    "error": f"command {command} is not available to call",
                }

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
