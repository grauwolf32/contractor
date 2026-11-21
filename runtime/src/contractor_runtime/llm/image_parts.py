"""Bounded native image attachments and content-free history projections."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from google.genai import types

IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg"})
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGE_DIMENSION = 4096
MAX_CONTEXT_IMAGES = 4
MAX_CONTEXT_IMAGE_BYTES = 8 * 1024 * 1024
ARTIFACT_IMAGE_PART = "_artifact_image"


def image_result_metadata(value: Any, *, replay: bool) -> Any:
    """Remove our transient native part before JSON history or telemetry encoding."""
    if not isinstance(value, Mapping) or ARTIFACT_IMAGE_PART not in value:
        return value
    part = value[ARTIFACT_IMAGE_PART]
    if not isinstance(part, types.Part) or part.inline_data is None:
        raise ValueError("invalid native artifact image result")
    result = {key: item for key, item in value.items() if key != ARTIFACT_IMAGE_PART}
    if replay:
        result["imageDelivery"] = "reread_required"
    return result


def image_response_metadata(value: Any, *, attached: bool) -> Any:
    """Historical metadata never claims that a missing attachment is still supplied."""
    if isinstance(value, Mapping) and value.get("imageDelivery") == "inline" and not attached:
        return {**value, "imageDelivery": "reread_required"}
    return value


def response_images(response: types.FunctionResponse) -> Iterator[tuple[str, bytes]]:
    for part in response.parts or ():
        blob = part.inline_data
        if (
            blob is None
            or blob.mime_type not in IMAGE_MEDIA_TYPES
            or not isinstance(blob.data, bytes)
            or not 0 < len(blob.data) <= MAX_IMAGE_BYTES
            or any(key != "inline_data" for key in part.model_dump(exclude_none=True))
        ):
            raise ValueError("unsupported native image attachment")
        yield blob.mime_type, blob.data


def image_context_size(contents: Sequence[types.Content]) -> tuple[int, int]:
    count = size = 0
    for content in contents:
        for part in content.parts or ():
            if part.function_response is not None:
                for _, data in response_images(part.function_response):
                    count += 1
                    size += len(data)
                    if count > MAX_CONTEXT_IMAGES or size > MAX_CONTEXT_IMAGE_BYTES:
                        raise ValueError("native image context capacity exceeded")
    return count, size


def discard_event_image_parts(events: Sequence[Any], event_ids: frozenset[str]) -> None:
    """Release only transient attachments of a committed compacted prefix."""
    for event in events:
        if event.id not in event_ids or event.content is None:
            continue
        for part in event.content.parts or ():
            response = part.function_response
            if response is not None and response.parts:
                # Validate before dropping anything; never silently discard another modality.
                tuple(response_images(response))
                response.parts = None
