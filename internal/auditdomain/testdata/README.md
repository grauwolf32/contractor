# Finding collection conformance

`finding-collection-v1.fixture.json` is a shared Go/Python contract fixture.
`collection` is the metadata value; `contents` maps document IDs to exact UTF-8
text bytes. `package_id` and `package_digest` freeze the expected canonical ZIP.
The fixture file itself is pretty-printed for review; package metadata is RFC
8785 JSON with no trailing newline.

The values were generated independently with Python `hashlib`, `json` and
`zipfile`. Metadata uses ASCII keys and integral numbers, so sorted compact JSON
with UTF-8 strings matches RFC 8785 here. ZIP entries use STORE, a 1980-01-01
timestamp, Unix regular-file mode 0644, creator/reader version 20, no flags,
extra fields or comments; `manifest.json` precedes members sorted by path.

The fixture contains ordinary Run and Audit origins, different generic subject
kinds, an optional hypothesis, a review without decision ID, and identical
binding names/revisions/evidence bytes in two Runs. Runtime V43-003 must consume
these same vectors instead of regenerating expected values with its own codec.
