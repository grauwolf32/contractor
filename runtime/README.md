# Contractor Runtime Agent

The Runtime Agent is the Python, single-slot execution process for Contractor
v2. During the bootstrap task it exposes only health endpoints; allocation,
ADK, A2A, and artifact behavior are added by later tasks.

```shell
uv sync
uv run contractor-runtime --listen 127.0.0.1:9080
```
