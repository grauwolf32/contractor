"""Generate the private protobuf bindings with pinned Go/Python compilers."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import grpc_tools

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".local/grpc-tools"
PYTHON_TARGET = ROOT / "runtime/src/contractor_runtime/contracts/grpc_wire"
GO_TARGET = ROOT / "api/v1alpha1/runtimepb"
GENERATORS = {
    "protoc-gen-go": ("google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.12", "v1.36.12"),
    "protoc-gen-go-grpc": (
        "google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.2",
        "1.6.2",
    ),
}


def generate(*, check: bool) -> None:
    if importlib.metadata.version("grpcio-tools") != "1.81.2":
        raise RuntimeError("generation requires locked grpcio-tools 1.81.2")
    TOOLS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GOBIN": str(TOOLS), "PATH": str(TOOLS) + os.pathsep + os.environ["PATH"]}
    for name, (module, version) in GENERATORS.items():
        executable = TOOLS / name
        if not executable.exists():
            subprocess.run(["go", "install", module], env=env, cwd=ROOT, check=True)
        actual = subprocess.check_output([str(executable), "--version"], text=True).strip()
        if actual != f"{name} {version}":
            raise RuntimeError(f"{name} has an unexpected version; install {module}")
    with tempfile.TemporaryDirectory(prefix="contractor-wire-") as directory:
        temporary = Path(directory)
        python = temporary / "python"
        go = temporary / "go"
        python.mkdir()
        go.mkdir()
        module = "github.com/grauwolf32/contractor"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{ROOT / 'api/v1alpha1'}",
                f"-I{Path(grpc_tools.__file__).parent / '_proto'}",
                f"--python_out={python}",
                f"--pyi_out={python}",
                f"--grpc_python_out={python}",
                f"--go_out={go}",
                f"--go_opt=module={module}",
                f"--go-grpc_out={go}",
                f"--go-grpc_opt=module={module}",
                "runtime.proto",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        outputs = {
            PYTHON_TARGET / file.name: file.read_bytes() for file in sorted(python.iterdir())
        }
        outputs.update(
            {
                GO_TARGET / file.name: file.read_bytes()
                for file in (go / "api/v1alpha1/runtimepb").iterdir()
            }
        )
        # protoc's Python generator emits a top-level sibling import. Keep
        # generated files in their owning package with a deterministic rewrite.
        target = PYTHON_TARGET / "runtime_pb2_grpc.py"
        outputs[target] = outputs[target].replace(
            b"import runtime_pb2 as runtime__pb2",
            b"from . import runtime_pb2 as runtime__pb2",
        )
        stale = []
        for path, content in outputs.items():
            if check:
                if not path.exists() or path.read_bytes() != content:
                    stale.append(str(path.relative_to(ROOT)))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        if stale:
            raise RuntimeError(
                "stale runtime bindings; run make generate-runtime-wire: " + ", ".join(stale)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    generate(check=parser.parse_args().check)
