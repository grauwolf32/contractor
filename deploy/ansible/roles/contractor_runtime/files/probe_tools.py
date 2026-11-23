"""Print the tools that the Runtime's executable-backed Toolsets would advertise.

Runs the Runtime's own startup probes for openapi@1, likec4@1 and scan@1 in
the current environment and prints {"<toolset ref>": [<tool>, ...]} as JSON.
The probes only run version commands in private temporary directories, so
this is safe beside a running Runtime. deploy/ansible runs it with the release
interpreter and the instance's environment before a restart.
"""

from __future__ import annotations

import asyncio
import json

from contractor_runtime.capabilities import DEFAULT_FACTORY_PROBE_TIMEOUT_SECONDS
from contractor_runtime.toolsets.likec4.tools import LikeC4ToolsetFactory
from contractor_runtime.toolsets.openapi.tools import OpenAPIToolsetFactory
from contractor_runtime.toolsets.scan.tools import ScanToolsetFactory


async def probe() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for factory in (OpenAPIToolsetFactory(), LikeC4ToolsetFactory(), ScanToolsetFactory()):
        try:
            tools = await asyncio.wait_for(
                factory.probe(), timeout=DEFAULT_FACTORY_PROBE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            tools = frozenset()
        result[factory.ref] = sorted(tools)
    return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(probe()), sort_keys=True))
