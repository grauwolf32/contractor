# Retained dependency SQL paths

`manifest-cases.json` contains the exact descriptor read by the surviving
PostgreSQL retention functions. The Go regression checks the typed codec and
migration JSON paths together, so a renamed reference cannot silently lose
its retention hold. Runtime receives ordinary selected Run artifacts.
