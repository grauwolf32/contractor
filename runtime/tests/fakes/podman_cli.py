"""Test-only CLI child: no engine access or container creation.

fake_executable in test_podman_engine.py installs it behind a shebang naming the
test interpreter.
"""

import json
import os
import signal
import sys
import time

command = sys.argv[4]
if command == "echo":
    print(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ), "pid": os.getpid()}))
elif command in {"flood", "slow-flood"}:
    if command == "slow-flood":
        time.sleep(0.25)
    for _ in range(512):
        os.write(1, b"x" * 8192)
        os.write(2, b"secret" * 1024)
elif command == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    time.sleep(30)
elif command == "inherited-pipe":
    # A detached fake helper holds stdout after its parent exits. The transport
    # must close its read pipe by the deadline, not wait for this helper's EOF.
    if os.fork() == 0:
        os.setsid()
        time.sleep(0.4)
        os._exit(0)
else:
    print("recognizable private failure", file=sys.stderr)
    sys.exit(125)
