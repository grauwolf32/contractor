"""Bounded YAML and ZIP syntax shared by local scanner rule adapters."""

from __future__ import annotations

import struct
import time

import yaml
from yaml.tokens import AliasToken, AnchorToken, TagToken

MAX_FILE_BYTES = 1024 * 1024
MAX_YAML_TOKENS = 100_000
MAX_YAML_DEPTH = 64


class InvalidRuleYAML(ValueError):
    pass


class _UniqueLoader(yaml.CSafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise InvalidRuleYAML
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_rule_yaml(data: bytes, deadline: float) -> dict:
    """Reject aliases, duplicate keys and unbounded syntax before constructing values."""
    if not data or len(data) > MAX_FILE_BYTES:
        raise InvalidRuleYAML
    try:
        text = data.decode("utf-8", errors="strict")
        depth = 0
        for count, token in enumerate(yaml.scan(text, Loader=yaml.CSafeLoader), 1):
            if time.monotonic() >= deadline:
                raise TimeoutError
            if count > MAX_YAML_TOKENS or isinstance(token, (AliasToken, AnchorToken, TagToken)):
                raise InvalidRuleYAML
            if isinstance(
                token,
                (
                    yaml.tokens.BlockMappingStartToken,
                    yaml.tokens.BlockSequenceStartToken,
                    yaml.tokens.FlowMappingStartToken,
                    yaml.tokens.FlowSequenceStartToken,
                ),
            ):
                depth += 1
                if depth > MAX_YAML_DEPTH:
                    raise InvalidRuleYAML
            elif isinstance(
                token,
                (
                    yaml.tokens.BlockEndToken,
                    yaml.tokens.FlowMappingEndToken,
                    yaml.tokens.FlowSequenceEndToken,
                ),
            ):
                depth -= 1
        value = yaml.load(text, Loader=_UniqueLoader)
    except (UnicodeError, yaml.YAMLError, ValueError, RecursionError):
        raise InvalidRuleYAML from None
    if time.monotonic() >= deadline:
        raise TimeoutError
    if not isinstance(value, dict):
        raise InvalidRuleYAML
    return value


def preflight_rule_zip(data: bytes, deadline: float, max_entries: int) -> None:
    """Bound central-directory work before ZipFile allocates member objects.

    Rulepacks use ordinary single-disk ZIP, without ZIP64, appended bytes or SFX.
    The owning adapter separately bounds encoded bytes and expanded content.
    """
    if not data.startswith(b"PK\x03\x04"):
        raise ValueError
    position = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
    if position < 0 or position + 22 > len(data):
        raise ValueError
    _, disk, directory_disk, on_disk, total, size, offset, comment = struct.unpack_from(
        "<4s4H2LH", data, position
    )
    if (
        disk != 0
        or directory_disk != 0
        or on_disk != total
        or not 1 <= total <= max_entries
        or offset + size != position
        or position + 22 + comment != len(data)
    ):
        raise ValueError
    cursor, count = offset, 0
    while cursor < position:
        if time.monotonic() >= deadline:
            raise TimeoutError
        if (
            count >= max_entries
            or cursor + 46 > position
            or data[cursor : cursor + 4] != b"PK\x01\x02"
        ):
            raise ValueError
        name, extra, note = struct.unpack_from("<3H", data, cursor + 28)
        cursor += 46 + name + extra + note
        if cursor > position:
            raise ValueError
        count += 1
    if count != total:
        raise ValueError
