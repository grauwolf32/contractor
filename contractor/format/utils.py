from typing import Any


def _safe_to_string(v: Any) -> str:
    try:
        v = str(v)
        return v
    except TypeError:
        ...
    return "<?>"


def flatten(data: Any, prefix="") -> str:
    result: str = ""
    if type(data) is dict:
        for k, v in data.items():
            result += flatten(v, prefix=f"{prefix}.{k}")
        return result
    elif type(data) is list:
        for i, v in enumerate(data):
            result += flatten(v, prefix=f"{prefix}.{i}")
        return result

    v = _safe_to_string(data)
    return f"{prefix}={v}\n"
