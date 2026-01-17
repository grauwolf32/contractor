import copy
import collections.abc
from dataclasses import dataclass, field
from typing import Any

def deep_merge(target: dict, diff: dict) -> dict:
    """
    Deep merge dictionaries. Lists are replaced (not concatenated) to avoid
    duplications and preserve OpenAPI determinism.
    """

    for key, value in diff.items():
        if isinstance(value, collections.abc.Mapping) and isinstance(
            target.get(key), collections.abc.Mapping
        ):
            target[key] = deep_merge(dict(target[key]), value)
        else:
            target[key] = copy.deepcopy(value)
    return target


@dataclass
class DictDiff:
    added: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)
    changed: dict[str, Any] = field(default_factory=dict)


def dict_diff(old: dict, new: dict) -> dict:
    """Structural diff without external dependencies."""

    diff: DictDiff = DictDiff()

    old_keys = set(old.keys())
    new_keys = set(new.keys())

    for k in new_keys - old_keys:
        diff.added[k] = new[k]

    for k in old_keys - new_keys:
        diff.removed[k] = old[k]

    for k in old_keys & new_keys:
        ov, nv = old[k], new[k]
        if isinstance(ov, dict) and isinstance(nv, dict):
            nested = dict_diff(ov, nv)
            if any(nested.values()):
                diff.changed[k] = nested
        elif ov != nv:
            diff.changed[k] = {"from": ov, "to": nv}
    return diff
