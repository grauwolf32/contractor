"""Image admission and decode work owned by the allocation's artifact reader."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO

from google.genai import types
from PIL import Image, UnidentifiedImageError

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.llm.image_parts import (
    MAX_CONTEXT_IMAGE_BYTES,
    MAX_CONTEXT_IMAGES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    image_context_size,
)


@dataclass(slots=True)
class ImageInputBudget:
    supports_images: bool | None = None
    count: int = 0
    size: int = 0
    pending: int = 0
    generation: int = 0
    decode_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def reserve(self, size: int) -> ImageReservation:
        if self.count >= MAX_CONTEXT_IMAGES or self.size + size > MAX_CONTEXT_IMAGE_BYTES:
            raise ToolInputError(
                "Image context capacity reached; reuse delivered images or finish this invocation",
                code="image_context_limit",
            )
        self.count += 1
        self.size += size
        self.pending += 1
        return ImageReservation(self, size)

    def reconcile(self, contents: Sequence[types.Content]) -> None:
        if self.pending:
            raise RuntimeError("image reads have not joined before the model request")
        count, size = image_context_size(contents)
        if count < self.count or size < self.size:
            self.generation += 1
        self.count, self.size = count, size


@dataclass(slots=True)
class ImageReservation:
    budget: ImageInputBudget
    size: int
    active: bool = True

    def commit(self) -> None:
        if not self.active:
            raise RuntimeError("image admission has already settled")
        self.budget.pending -= 1
        self.active = False

    def release(self) -> None:
        if self.active:
            self.budget.pending -= 1
            self.budget.count -= 1
            self.budget.size -= self.size
            self.active = False


def validate_image(data: bytes, media_type: str) -> tuple[int, int]:
    """Decode off-loop with Pillow's format checks and decompression protections intact."""
    expected = {"image/png": "PNG", "image/jpeg": "JPEG"}[media_type]
    try:
        with Image.open(BytesIO(data), formats=["PNG", "JPEG"]) as image:
            width, height = image.size
            if (
                image.format != expected
                or width < 1
                or height < 1
                or max(width, height) > MAX_IMAGE_DIMENSION
                or width * height > MAX_IMAGE_PIXELS
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ToolInputError("Image format, dimensions or frame count is unsupported")
            image.verify()
        del image
        with Image.open(BytesIO(data), formats=["PNG", "JPEG"]) as decoded:
            decoded.load()
        return width, height
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        ValueError,
    ):
        raise ToolInputError("Image is invalid or exceeds supported decode bounds") from None
