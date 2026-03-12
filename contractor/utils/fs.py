from typing import Optional
import unicodedata

def norm_unicode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return unicodedata.normalize("NFC", value)


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