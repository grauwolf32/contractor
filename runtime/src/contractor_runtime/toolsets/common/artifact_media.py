"""Media types whose bytes the UTF-8 Artifact readers refuse before download."""

from __future__ import annotations

# Top-level types of binary formats (ART-R117). image/svg+xml is XML text.
BINARY_TOP_LEVEL_TYPES = frozenset({"image", "audio", "video", "font", "model"})
TEXT_IMAGE_MEDIA_TYPES = frozenset({"image/svg+xml"})
# application/ subtypes of archive and document formats; a +zip suffix, such
# as application/vnd.contractor.agent-skill+zip, also names an archive.
BINARY_APPLICATION_SUBTYPES = frozenset(
    {
        "gzip",
        "java-archive",
        "pdf",
        "wasm",
        "x-7z-compressed",
        "x-bzip2",
        "x-gzip",
        "x-tar",
        "x-xz",
        "x-zip-compressed",
        "zip",
        "zstd",
    }
)


def is_binary_media_type(media_type: str) -> bool:
    """Return whether ``media_type`` names a binary format, so the Artifact can
    be refused as non-text from its metadata alone.

    Any other media type, including ``application/octet-stream``, may still
    hold UTF-8 text and is decided by decoding the downloaded bytes.
    """

    top_level, _, subtype = media_type.partition("/")
    if top_level in BINARY_TOP_LEVEL_TYPES:
        return media_type not in TEXT_IMAGE_MEDIA_TYPES
    return top_level == "application" and (
        subtype in BINARY_APPLICATION_SUBTYPES or subtype.endswith("+zip")
    )
