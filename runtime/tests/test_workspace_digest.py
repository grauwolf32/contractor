from __future__ import annotations

import hashlib

import jcs
import pytest

from contractor_runtime.projectfs.storage import workspace_digest


@pytest.mark.parametrize(
    ("directories", "text_files"),
    [
        (set(), {}),
        ({"empty", "empty/nested"}, {}),
        (set(), {"empty.txt": ""}),
        ({"z", "a"}, {"z/last.py": "print('last')\n", "a/first.py": "print('first')\n"}),
        (
            {"src", "\u00e9", "e\u0301", "\ue000", "\U0001f600"},
            {
                'src/quotes"and\\slashes.txt': '"\\\b\f\n\r\t\u0001',
                "\u00e9/text.txt": "\u00e9 e\u0301 \u2028 \u2029",
                "\ue000/text.txt": "\u041f\u0440\u0438\u0432\u0435\u0442 \u4e16\u754c",
                "\U0001f600/text.txt": "\U0001d11e \U0001f600",
            },
        ),
    ],
)
def test_workspace_digest_preserves_existing_canonical_document(
    directories: set[str], text_files: dict[str, str]
) -> None:
    # Stored overlays and snapshot cursors already contain hashes of this
    # document. Changing serialization would invalidate those existing values.
    encoded = jcs.canonicalize(
        {
            "directories": sorted(directories),
            "files": [{"path": path, "text": text_files[path]} for path in sorted(text_files)],
        }
    )
    expected = "sha256:" + hashlib.sha256(encoded).hexdigest()

    assert workspace_digest(directories, text_files) == expected
    assert workspace_digest(directories, dict(reversed(list(text_files.items())))) == expected
