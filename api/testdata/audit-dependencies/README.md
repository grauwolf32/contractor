# Retained item dependency conformance

`manifest-cases.json` holds one complete four-origin manifest and bounded
single-path mutations. An empty path leaves the base unchanged. Go and Python
run every case against the production codecs; both also exercise the 32-entry
ceiling, metadata byte bound and task/execution agreement. The source JSON
references intentionally grant no access. The exact source/retained content
uses shared `contracts.ExactArtifact` field names.
