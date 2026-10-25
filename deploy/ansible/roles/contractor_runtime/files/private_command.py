"""Transfer configurable environment through protected stdin, never shell prefixes.

Ansible's connection plugins print task.environment in EXEC even with no_log.
The caller must also use no_log to protect the module payload and child output.
exec preserves the command's exit status, signals and lifecycle without a child
that could outlive its Ansible command owner.
"""

import json
import os
import sys


def main():
    try:
        payload = json.load(sys.stdin)
        argv = payload["argv"]
        environment = os.environ.copy()
        environment.update(
            {key: str(value) for key, value in payload["environment"].items()}
        )
        os.execvpe(argv[0], argv, environment)
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        print(
            "Private deployment command could not start; check executable and environment configuration.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
