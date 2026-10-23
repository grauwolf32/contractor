"""Derive a shared systemd scope policy with the deployed Runtime's parser.

This opt-in policy is for a dedicated account whose Podman instances use the
same resource bounds. Never silently constrain an instance to another's limits.
No engine calls, environment reads or default limit copies belong here.
"""

from __future__ import annotations

import json
import sys

from contractor_runtime.settings import parse_settings


def scope_policy(arguments: list[list[str]]) -> dict[str, str | int]:
    policies = []
    for argv in arguments:
        settings = parse_settings(argv, {}).podman
        if not settings.enabled:
            continue
        policy = {
            "cpu_quota": format(settings.cpus * 100, ".12g"),
            "memory_bytes": settings.memory_bytes,
            "pids": settings.pids,
        }
        if policies and policy != policies[0]:
            raise ValueError(
                "shared Podman scope policy requires identical CPU, memory and PID limits"
            )
        policies.append(policy)
    if not policies:
        raise ValueError("shared Podman scope policy requires an enabled Podman instance")
    return policies[0]


if __name__ == "__main__":
    print(json.dumps(scope_policy([argv[1:] for argv in json.load(sys.stdin)]), sort_keys=True))
