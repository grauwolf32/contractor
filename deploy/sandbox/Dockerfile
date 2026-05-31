# Sandbox image for the exploit agents' run_python / execute_bash tools.
# Ephemeral, per-agent-run container: target source mounted read-only at
# /project, scripts write to the writable /work scratch dir (captured as
# artifacts), host network. See contractor/tools/podman.py.
FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Base python + common offensive CLI tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        curl wget jq ca-certificates \
        nmap sqlmap netcat-traditional gobuster \
    && rm -rf /var/lib/apt/lists/*

# Python libraries useful for HTTP exploitation scripting.
RUN pip3 install --no-cache-dir \
        requests httpx beautifulsoup4 pyjwt pwntools

# Writable scratch workdir; target source is bind-mounted read-only at /project.
RUN mkdir -p /work
WORKDIR /work

# Default command is overridden by the sandbox runner (kept alive via `sleep`).
CMD ["sleep", "infinity"]
