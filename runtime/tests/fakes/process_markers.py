"""Complete readiness publication for synthetic subprocess fixtures."""

from pathlib import Path
from textwrap import indent


def atomic_marker_source(marker: Path, contents_expression: str, *, indentation: str = "") -> str:
    """Build self-contained child code publishing one complete readiness payload.

    ``contents_expression`` is Python source owned by the calling test. A
    fixture has one writer; the sibling staging path stays private until its
    write closes and ``os.replace`` makes marker existence mean readiness.
    """

    return indent(
        "import os as _fixture_os\n"
        "from pathlib import Path as _FixturePath\n"
        f"_fixture_marker = _FixturePath({str(marker)!r})\n"
        "_fixture_staging = _fixture_marker.with_name(_fixture_marker.name + '.tmp')\n"
        f"_fixture_staging.write_text({contents_expression}, encoding='utf-8')\n"
        "_fixture_os.replace(_fixture_staging, _fixture_marker)\n",
        indentation,
    )
