# Contractor private wire contracts v1alpha1

These JSON Schema documents describe Contractor-owned payloads exchanged
between the Go Server and Python Runtime Agent. Every top-level payload carries
`"apiVersion": "contractor/v1alpha1"`, uses camelCase fields, and rejects
unknown fields.

A2A envelopes themselves are owned by the official A2A 1.0 SDKs. Contractor
places `StageContentRequest` and `StageContentResult` in an A2A DataPart with
media type `application/vnd.contractor.stage-content+json`.

The schemas are review artifacts and compatibility contracts. Go and Python
DTOs are maintained explicitly and are checked against shared golden fixtures
under `api/testdata/v1alpha1`.
