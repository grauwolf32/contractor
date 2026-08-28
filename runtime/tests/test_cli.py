from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request


def unused_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_runtime_process_handles_sigterm() -> None:
    port = unused_tcp_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "contractor_runtime", "--listen", f"127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"runtime exited before readiness: {process.returncode}\n{stdout}\n{stderr}"
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/healthz", timeout=0.25
                ) as response:
                    assert response.status == 200
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    message = "runtime did not become ready within five seconds"
                    raise AssertionError(message) from None
                time.sleep(0.05)

        process.terminate()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
