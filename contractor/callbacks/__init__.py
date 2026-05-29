from typing import Any


def default_tool(meta: Any = None) -> dict:
    """
    default_tool: You must not use this tool. This is safeguard mechanism.
        Args:
            meta: meta information about failed tool call
        Returns:
            instructions
    """

    func_name = meta.get("func_name") if isinstance(meta, dict) else meta
    return {"error": f"tool {func_name} is not available!"}
