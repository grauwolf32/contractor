import unicodedata


def norm_unicode(value: str | None) -> str | None:
    if value is None:
        return None
    return unicodedata.normalize("NFC", value)


def norm_unicode_strict(path: str) -> str:
    result = norm_unicode(path)
    if result is None:
        raise ValueError(f"Cannot normalize path: {path!r}")
    return result


def normalize_slashes(path: str) -> str:
    return path.replace("\\", "/")


def xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
