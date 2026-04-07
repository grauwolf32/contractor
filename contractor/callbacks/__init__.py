from typing import Any


def default_tool(meta: dict[str, Any]) -> dict:
    """
    default_tool: You must not use this tool. This is safeguard mechanism.
        Args:
            meta: meta information about failed tool call
        Returns:
            instructions
    """

    return {"error": f"tool {meta.get('func_name')} is not available!"}
