"""Adapter for the XBOW validation-benchmarks (CTF-style flag-capture).

Each benchmark under ``validation-benchmarks/benchmarks/XBEN-NNN-24/`` is a
docker-compose app with a ``FLAG{...}`` injected at build time and hidden
behind a vulnerability; ``benchmark.json`` carries a name, a one-line hint,
a difficulty ``level`` and vuln-class ``tags``. The benchmark ships its own
source, so the code-aware pipelines can run against it with the live
container as the dynamic-testing target (same shape as the crAPI/VAmPI
fixtures), and success is measured by whether the agent recovers the flag.

This module provides:
  - ``discover_benchmarks`` — parse all benchmarks into ``XbowBenchmark``.
  - ``XbowService`` — bring one up via podman-compose, resolve the (possibly
    dynamic) host port, health-check, expose ``base_url``, tear down.

Only podman is available in this environment, so we drive ``podman-compose``
directly against each benchmark's ``docker-compose.yml`` rather than the
upstream ``make`` targets (which hardcode ``docker compose``).
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml

# Service names / images that are datastores, not the web entrypoint.
_DATASTORE_HINTS = ("mongo", "mysql", "postgres", "mariadb", "redis", "db", "memcached")


@dataclass
class XbowBenchmark:
    """One XBOW benchmark, parsed from its dir + benchmark.json + compose."""

    id: str  # e.g. "XBEN-001-24"
    path: Path
    name: str
    description: str
    level: int
    tags: list[str]
    win_condition: str
    flag: str
    compose_file: Path
    service: str  # the web entrypoint service
    container_port: int
    host_port: Optional[int]  # None => dynamic, resolve after `up`

    @property
    def primary_tag(self) -> str:
        return self.tags[0] if self.tags else "unknown"


def flag_captured(flag: str, *texts: str) -> bool:
    """True if the benchmark's flag (full string or its inner token) appears
    in any of *texts* (agent result, verdict evidence, collected HTTP chain)."""
    if not flag:
        return False
    blob = "\n".join(t for t in texts if t)
    if not blob:
        return False
    candidates = {flag}
    m = re.search(r"\{([^}]+)\}", flag)
    if m:
        candidates.add(m.group(1))  # the inner uuid, robust to flag{}/FLAG{} casing
    return any(c and c in blob for c in candidates)


def _read_flag(env_file: Path) -> str:
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("FLAG="):
            val = line.split("=", 1)[1].strip()
            return val.strip('"').strip("'")
    return ""


def _parse_port(entry: object) -> tuple[int, Optional[int]]:
    """Return (container_port, host_port) from a compose ports entry.

    "80" / 80            -> (80, None)   dynamic host port
    "8000:80"            -> (80, 8000)
    "127.0.0.1:8000:80"  -> (80, 8000)
    """
    parts = str(entry).split(":")
    if len(parts) == 1:
        return int(parts[0]), None
    if len(parts) == 2:
        return int(parts[1]), int(parts[0])
    return int(parts[2]), int(parts[1])


def _pick_web_service(services: dict) -> Optional[str]:
    """The service exposing an HTTP port that isn't a datastore."""
    candidates = []
    for name, spec in services.items():
        if not isinstance(spec, dict) or "ports" not in spec:
            continue
        lname = name.lower()
        image = str(spec.get("image", "")).lower()
        if any(h in lname or h in image for h in _DATASTORE_HINTS):
            continue
        candidates.append(name)
    if candidates:
        return candidates[0]
    # Fallback: any service with a ports mapping.
    for name, spec in services.items():
        if isinstance(spec, dict) and "ports" in spec:
            return name
    return None


def discover_benchmarks(benchmarks_root: Path) -> list[XbowBenchmark]:
    """Parse every ``XBEN-*`` benchmark under *benchmarks_root*."""
    found: list[XbowBenchmark] = []
    for bdir in sorted(benchmarks_root.iterdir()):
        meta_file = bdir / "benchmark.json"
        compose_file = bdir / "docker-compose.yml"
        if not meta_file.is_file() or not compose_file.is_file():
            continue
        try:
            meta = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
            compose = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        services = compose.get("services") or {}
        service = _pick_web_service(services)
        if not service:
            continue
        ports = services[service].get("ports") or []
        if not ports:
            continue
        container_port, host_port = _parse_port(ports[0])
        try:
            level = int(meta.get("level", 0))
        except (TypeError, ValueError):
            level = 0
        found.append(
            XbowBenchmark(
                id=bdir.name,
                path=bdir,
                name=str(meta.get("name", bdir.name)),
                description=str(meta.get("description", "")),
                level=level,
                tags=list(meta.get("tags", []) or []),
                win_condition=str(meta.get("win_condition", "flag")),
                flag=_read_flag(bdir / ".env"),
                compose_file=compose_file,
                service=service,
                container_port=container_port,
                host_port=host_port,
            )
        )
    return found


@dataclass
class XbowService:
    """Bring one XBOW benchmark up via podman-compose for dynamic testing."""

    benchmark: XbowBenchmark
    project_name: str = ""
    _resolved_port: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.project_name:
            self.project_name = f"xbow-{self.benchmark.id.lower()}"

    def _compose(self, *args: str) -> list[str]:
        return [
            "podman-compose",
            "-f", str(self.benchmark.compose_file),
            "-p", self.project_name,
            *args,
        ]

    def up(self, *, timeout: float = 120.0, quiet: bool = True) -> None:
        import os

        env = dict(os.environ)
        if self.benchmark.flag:
            env["FLAG"] = self.benchmark.flag  # build-arg `args: - FLAG`
        out = subprocess.DEVNULL if quiet else None
        subprocess.run(
            self._compose("up", "-d", "--build"),
            check=True, env=env, stdout=out, stderr=out,
            cwd=str(self.benchmark.path),
        )
        self._resolved_port = self._resolve_host_port()
        self._wait_healthy(self.base_url() + "/", timeout=timeout)

    def _resolve_host_port(self) -> int:
        if self.benchmark.host_port is not None:
            return self.benchmark.host_port
        cport = self.benchmark.container_port
        # podman-compose names containers <project>_<service>_1; fall back to a
        # label scan if that default ever changes.
        names = [
            f"{self.project_name}_{self.benchmark.service}_1",
            f"{self.project_name}-{self.benchmark.service}-1",
        ]
        for name in names:
            mapped = self._podman_port(name, cport)
            if mapped:
                return mapped
        # Last resort: scan running containers for one publishing the port.
        scan = subprocess.run(
            ["podman", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        for name in scan.stdout.split():
            if self.project_name.replace("-", "") in name.replace("-", "").replace("_", ""):
                mapped = self._podman_port(name, cport)
                if mapped:
                    return mapped
        raise RuntimeError(
            f"could not resolve host port for {self.benchmark.id} "
            f"service={self.benchmark.service} cport={cport}"
        )

    @staticmethod
    def _podman_port(container: str, cport: int) -> Optional[int]:
        res = subprocess.run(
            ["podman", "port", container, f"{cport}/tcp"],
            capture_output=True, text=True,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        # e.g. "0.0.0.0:49153" (possibly multiple lines)
        m = re.search(r":(\d+)\s*$", res.stdout.strip().splitlines()[0])
        return int(m.group(1)) if m else None

    def base_url(self) -> str:
        if self._resolved_port is None:
            raise RuntimeError("service not up — call up() first")
        return f"http://localhost:{self._resolved_port}"

    @staticmethod
    def _wait_healthy(url: str, *, timeout: float, interval: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        last: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                r = httpx.get(url, timeout=5.0)
                if r.status_code < 500:
                    return
                last = f"status {r.status_code}"
            except Exception as exc:  # connection refused while booting
                last = str(exc)
            time.sleep(interval)
        raise RuntimeError(f"{url} not healthy within {timeout}s (last: {last})")

    def down(self, *, rmi: bool = True, quiet: bool = True) -> None:
        """Clean step: tear down this benchmark's containers, pod, and images.

        Strictly scoped to this benchmark's compose project — never touches any
        container outside ``self.project_name`` (lesson learned: a broad
        ``podman rm -f`` once nuked the LiteLLM proxy + langfuse). With
        ``rmi=True`` the locally-built ``localhost/<project>_*`` images are
        removed too, so a 104-benchmark sweep does not accumulate disk.
        """
        out = subprocess.DEVNULL if quiet else None
        subprocess.run(
            self._compose("down", "-t", "5"),
            check=False, stdout=out, stderr=out, cwd=str(self.benchmark.path),
        )
        # podman-compose may leave the project pod behind; remove it by name.
        subprocess.run(
            ["podman", "pod", "rm", "-f", f"pod_{self.project_name}"],
            check=False, stdout=out, stderr=out,
        )
        if rmi:
            self._remove_project_images(out)

    def _remove_project_images(self, out) -> None:
        """Remove only images this project built (localhost/<project>_<svc>)."""
        res = subprocess.run(
            ["podman", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True,
        )
        prefixes = (
            f"localhost/{self.project_name}_",
            f"localhost/{self.project_name}-",
        )
        imgs = [
            line for line in res.stdout.split()
            if any(line.startswith(p) for p in prefixes)
        ]
        if imgs:
            subprocess.run(
                ["podman", "rmi", "-f", *imgs],
                check=False, stdout=out, stderr=out,
            )
