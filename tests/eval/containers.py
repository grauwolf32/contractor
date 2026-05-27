"""Podman Compose lifecycle for eval tests that need live targets."""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PodmanService:
    """Manages a podman-compose project for eval fixtures."""

    compose_file: Path
    project_name: str
    service_ports: dict[str, int] = field(default_factory=dict)

    init_urls: list[str] = field(default_factory=list)

    def up(self, *, build: bool = True, timeout: float = 60.0, quiet: bool = False) -> None:
        cmd = [
            "podman-compose",
            "-f", str(self.compose_file),
            "-p", self.project_name,
            "up", "-d",
        ]
        if build:
            cmd.append("--build")

        logger.info("starting containers: %s", " ".join(cmd))
        run_kwargs: dict = {"check": True, "text": True}
        if quiet:
            run_kwargs["capture_output"] = True
        subprocess.run(cmd, **run_kwargs)

        for service, port in self.service_ports.items():
            url = f"http://localhost:{port}/"
            logger.info("waiting for %s at %s", service, url)
            self.wait_healthy(url, timeout=timeout)

        for url in self.init_urls:
            logger.info("init request: %s", url)
            try:
                resp = httpx.get(url, timeout=10.0)
                logger.info("init %s -> %d", url, resp.status_code)
            except httpx.HTTPError as exc:
                logger.warning("init %s failed: %s", url, exc)

    def down(self) -> None:
        cmd = [
            "podman-compose",
            "-f", str(self.compose_file),
            "-p", self.project_name,
            "down", "-t", "5",
        ]
        logger.info("stopping containers: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            logger.warning("podman-compose down failed: %s", exc.stderr)

    def base_url(self, service: str) -> str:
        port = self.service_ports[service]
        return f"http://localhost:{port}"

    @staticmethod
    def wait_healthy(
        url: str,
        *,
        timeout: float = 30.0,
        interval: float = 2.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=5.0, follow_redirects=True)
                if resp.status_code < 500:
                    logger.info("health check passed: %s -> %d", url, resp.status_code)
                    return
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
            time.sleep(interval)

        raise TimeoutError(
            f"service at {url} not healthy after {timeout}s"
            + (f": {last_error}" if last_error else "")
        )
