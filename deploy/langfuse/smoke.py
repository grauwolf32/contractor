"""Prove generic Go/Runtime OTLP export and native projection in Langfuse v4."""

import base64
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
IMAGES = {
    "web": "docker.langfuse.com/langfuse/langfuse:4.67.0@sha256:8e2b58b5086bfc955452a12c72eb2306bf3cb2ec59de8115454514c4c8e36e33",
    "worker": "docker.langfuse.com/langfuse/langfuse-worker:4.67.0@sha256:a0ec08090953ad8d710565a23cda1b24e84dc889c9122e88df2c5b8ada4cfe1f",
    "postgres": "docker.io/library/postgres@sha256:18cfe3ef5e6815560c98237d6216d1e5119702fb0f3894c8785dd58b8bbe5d73",
    "clickhouse": "docker.io/clickhouse/clickhouse-server:25.12@sha256:8a790dd3468db22b1d4e7b18a176f378ff5ff6053b9c48dd4ea1fa71a24c5ba6",
    "s3": "docker.io/chrislusf/seaweedfs:4.48@sha256:f1f303474940e9e088edbd2c1658f7ac38cb0dd90f91a2b0a51916d74b8575a1",
    "redis": "docker.io/library/redis:7@sha256:44960b3100c48159a66d28c18b0588c193492e25f7551d60e51dd464fc3eca5b",
}


class Probe:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.pod = "contractor-langfuse-smoke-" + secrets.token_hex(4)
        self.containers = []
        self.sensitive = []
        self.env = os.environ.copy()

    def secret(self):
        value = secrets.token_hex(32)
        self.sensitive.append(value)
        return value

    def redact(self, value):
        for secret in self.sensitive:
            value = value.replace(secret, "[redacted]")
        return value

    def command(self, *args, env=None, timeout=60):
        result = subprocess.run(
            args,
            check=False,
            env=env or self.env,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(self.redact(result.stderr[-4000:]) or "Probe command failed")
        return result.stdout.strip()

    def podman(self, *args, **kwargs):
        return self.command("podman", *args, **kwargs)

    def start(self, service, env=None, arguments=(), memory="768m", mounts=()):
        name = self.pod + "-" + service
        self.containers.append(name)
        args = [
            "run",
            "--detach",
            "--pod",
            self.pod,
            "--name",
            name,
            "--memory",
            memory,
        ]
        for source, destination in mounts:
            args += ["--volume", f"{source}:{destination}:Z"]
        for key in env or {}:
            args += ["--env", key]
        self.podman(*args, IMAGES[service], *arguments, env={**self.env, **(env or {})})
        return name

    def wait_command(self, *args):
        for _ in range(60):
            result = subprocess.run(args, check=False, env=self.env, capture_output=True, timeout=5)
            if result.returncode == 0:
                return
            time.sleep(1)
        raise RuntimeError("Dependency did not become ready within its probe budget")

    def get(self, path, *, auth=False):
        headers = {"Authorization": self.authorization} if auth else {}
        with urlopen(Request(self.base + path, headers=headers), timeout=5) as response:
            return json.load(response)

    def run(self):
        for service, image in IMAGES.items():
            exists = subprocess.run(
                ["podman", "image", "exists", image],
                check=False,
                env=self.env,
                capture_output=True,
                timeout=10,
            )
            if exists.returncode:
                print(f"Pulling the pinned {service} image", flush=True)
                self.podman("pull", image, timeout=600)
        self.podman(
            "pod",
            "create",
            "--name",
            self.pod,
            "--infra-name",
            self.pod + "-infra",
            "--publish",
            "127.0.0.1::3000",
        )
        password = self.secret()
        public_key, secret_key = "pk-lf-" + self.secret(), "sk-lf-" + self.secret()
        self.authorization = (
            "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        )
        self.sensitive.append(self.authorization)
        common = {
            "DATABASE_URL": f"postgresql://postgres:{password}@127.0.0.1:5432/langfuse",
            "NEXTAUTH_URL": "http://127.0.0.1:3000",
            "NEXTAUTH_SECRET": self.secret(),
            "SALT": self.secret(),
            "ENCRYPTION_KEY": self.secret(),
            "TELEMETRY_ENABLED": "false",
            "LANGFUSE_IN_APP_AGENT_ENABLED": "false",
            "CLICKHOUSE_MIGRATION_URL": "clickhouse://127.0.0.1:9000",
            "CLICKHOUSE_URL": "http://127.0.0.1:8123",
            "CLICKHOUSE_USER": "telemetry",
            "CLICKHOUSE_PASSWORD": password,
            "CLICKHOUSE_CLUSTER_ENABLED": "false",
            "REDIS_HOST": "127.0.0.1",
            "REDIS_PORT": "6379",
            "LANGFUSE_MIGRATION_V4_WRITE_MODE": "events_only",
            "LANGFUSE_MIGRATION_V4_NATIVE_OTEL_BEHAVIOUR": "direct",
            "LANGFUSE_MIGRATION_V4_ALLOW_PREVIEW_OPT_IN": "true",
            "LANGFUSE_INGESTION_QUEUE_DELAY_MS": "0",
            "LANGFUSE_INGESTION_CLICKHOUSE_WRITE_INTERVAL_MS": "1000",
        }
        for kind in ("EVENT", "MEDIA"):
            for key, value in {
                "BUCKET": "langfuse",
                "REGION": "us-east-1",
                "ACCESS_KEY_ID": "telemetry",
                "SECRET_ACCESS_KEY": password,
                "ENDPOINT": "http://127.0.0.1:8333",
                "FORCE_PATH_STYLE": "true",
            }.items():
                common[f"LANGFUSE_S3_{kind}_UPLOAD_{key}"] = value
        directories = {}
        for service in ("postgres", "clickhouse", "s3", "redis"):
            directories[service] = self.directory / service
            directories[service].mkdir()
        pg = self.start(
            "postgres",
            {
                "POSTGRES_PASSWORD": password,
                "POSTGRES_DB": "langfuse",
            },
            mounts=[(directories["postgres"], "/var/lib/postgresql/data")],
        )
        ch_config = self.directory / "clickhouse.xml"
        ch_config.write_text("""<clickhouse>
<max_server_memory_usage>536870912</max_server_memory_usage>
<background_schedule_pool_size>16</background_schedule_pool_size>
<background_message_broker_schedule_pool_size>4</background_message_broker_schedule_pool_size>
<background_distributed_schedule_pool_size>4</background_distributed_schedule_pool_size>
</clickhouse>""")
        ch = self.start(
            "clickhouse",
            {
                "CLICKHOUSE_USER": "telemetry",
                "CLICKHOUSE_PASSWORD": password,
            },
            mounts=[
                (directories["clickhouse"], "/var/lib/clickhouse"),
                (ch_config, "/etc/clickhouse-server/config.d/smoke.xml"),
            ],
        )
        self.start("redis", mounts=[(directories["redis"], "/data")], memory="128m")
        self.start(
            "s3",
            {
                "AWS_ACCESS_KEY_ID": "telemetry",
                "AWS_SECRET_ACCESS_KEY": password,
                "S3_BUCKET": "langfuse",
            },
            mounts=[(directories["s3"], "/data")],
            memory="512m",
        )
        self.wait_command("podman", "exec", pg, "pg_isready", "-U", "postgres")
        self.wait_command(
            "podman",
            "exec",
            ch,
            "sh",
            "-c",
            'clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT 1"',
        )
        # Podman allocates dynamic published ports when the pod first starts.
        address = self.podman("port", self.pod + "-infra", "3000/tcp")
        if not address:
            raise RuntimeError("Podman did not assign a loopback HTTP port")
        self.base = "http://" + address
        self.start(
            "web",
            {
                **common,
                "LANGFUSE_INIT_ORG_ID": "telemetry-smoke-org",
                "LANGFUSE_INIT_ORG_NAME": "Disposable telemetry test",
                "LANGFUSE_INIT_PROJECT_ID": "telemetry-smoke-project",
                "LANGFUSE_INIT_PROJECT_NAME": "OTLP interoperability",
                "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": public_key,
                "LANGFUSE_INIT_PROJECT_SECRET_KEY": secret_key,
                "LANGFUSE_INIT_USER_EMAIL": "telemetry@example.invalid",
                "LANGFUSE_INIT_USER_PASSWORD": self.secret(),
            },
            memory="1280m",
        )
        self.start("worker", common, memory="1024m")
        print("Waiting for disposable Langfuse 4.67.0 readiness", flush=True)
        for _ in range(180):
            states = self.podman("inspect", "--format", "{{.State.Running}}", *self.containers)
            if any(state != "true" for state in states.splitlines()):
                raise RuntimeError("A disposable backend process exited during startup")
            try:
                self.get("/api/public/health")
                self.get("/api/public/projects", auth=True)
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Langfuse did not become ready within its probe budget")
        headers = json.dumps(
            {"Authorization": self.authorization, "x-langfuse-ingestion-version": "4"}
        )
        exporter_env = {
            **self.env,
            "CONTRACTOR_TEST_OTLP_ENDPOINT": self.base + "/api/public/otel/v1/traces",
            "CONTRACTOR_TEST_OTLP_HEADERS": headers,
        }
        self.command(
            "go",
            "test",
            "-count=1",
            "./internal/telemetry",
            "-run",
            "^TestPlannerOTLPCollectorProjection$",
            env=exporter_env,
            timeout=180,
        )
        self.command(
            sys.executable,
            "deploy/telemetry/emit.py",
            exporter_env["CONTRACTOR_TEST_OTLP_ENDPOINT"],
            env=exporter_env,
        )
        self.check()

    def check(self):
        expected = {
            "contractor.planner.model": ("Go", 2, 3, 1),
            "contractor.worker.model": ("Runtime", 13, 17, 3),
        }
        for _ in range(90):
            observations = self.get(
                "/api/public/v2/observations?limit=10&fields=core,basic,io,model,usage",
                auth=True,
            )["data"]
            actual = {item["name"]: item for item in observations}
            if set(expected) <= actual.keys():
                for name, (source, inputs, outputs, cached) in expected.items():
                    item = actual[name]
                    if item["type"] != "GENERATION" or item["model"] != "smoke-model":
                        raise RuntimeError(f"{name}: model/generation projection mismatch")
                    messages = (
                        json.loads(item["input"])
                        if isinstance(item["input"], str)
                        else item["input"]
                    )
                    output = (
                        json.loads(item["output"])
                        if isinstance(item["output"], str)
                        else item["output"]
                    )
                    if messages[0] != {
                        "role": "system",
                        "content": f"{source} system instruction",
                    }:
                        raise RuntimeError(f"{name}: system instruction projection mismatch")
                    if messages[1] != {
                        "role": "user",
                        "parts": [{"type": "text", "content": f"synthetic {source} input"}],
                    }:
                        raise RuntimeError(f"{name}: input projection mismatch")
                    if output != [
                        {
                            "role": "assistant",
                            "parts": [
                                {
                                    "type": "text",
                                    "content": f"synthetic {source} output",
                                }
                            ],
                            "finish_reason": "unknown",
                        }
                    ]:
                        raise RuntimeError(f"{name}: output projection mismatch")
                    usage = item["usageDetails"]
                    checks = {
                        "input": inputs - cached,
                        "output": outputs,
                        "input_cached_tokens": cached,
                    }
                    if any(usage.get(key) != value for key, value in checks.items()):
                        raise RuntimeError(f"{name}: native usage projection mismatch: {usage}")
                print(
                    "Langfuse: both exporters acknowledged; native model, prompts, replies and token projections verified",
                    flush=True,
                )
                return
            time.sleep(1)
        raise RuntimeError("Langfuse observations did not appear within the probe budget")

    def cleanup(self):
        self.podman("pod", "rm", "--force", "--ignore", self.pod)

    def diagnostics(self):
        for name in self.containers:
            result = subprocess.run(
                ["podman", "logs", "--tail", "15", name],
                check=False,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(
                name,
                self.redact((result.stdout + result.stderr)[-4000:]),
                file=sys.stderr,
            )


def main():
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    local = REPO / ".local"
    local.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="langfuse-smoke.", dir=local) as directory:
        probe = Probe(directory)
        try:
            probe.run()
        except BaseException:
            probe.diagnostics()
            raise
        finally:
            probe.cleanup()


if __name__ == "__main__":
    main()
