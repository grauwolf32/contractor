# Supplied OpenAPI scan inputs

Select `openapi-sqlmap-scan@1` or `openapi-nuclei-scan@1`. Upload OpenAPI and
`engagement.json` as Project materials; engagement uses media type
`application/vnd.contractor.engagement+json`. Pass the chosen `*-parameters.json`
object as CreateAuditRequest.parameters. Upload the matching `*-request-values.json`
as optional `request_values` (application/json). There is no settings artifact.

The example selects `#/paths/~1pets~1{id}/post` at
`http://192.168.56.10:8080/api/pets/7?search=Milo`. SQLMap preserves POST and the
JSON body and tests `name` and `search`. Its optional fixture-credential must be
created through Credentials as target-bearer@1, referenced by an engagement
identity, and selected with credential_id. Do not put its token in materials.
The example engagement declares that reference; for anonymous execution remove
its scanner identity and omit credential_id.

Nuclei prepares a fixed URL, without replaying the operation method/body. It
runs HTTP rules through engagement; raw, headless, code, flow and other
unsupported controlled constructs fail explicitly. Native pacing follows
engagement HTTP limits. Optional rulepack materials retain exact provenance.

Source-to-scan uses the same parameters and values with a source ZIP. Its
ordinary preparation Run generates OpenAPI and bounded common work; each
accepted operation becomes a separate scan Run. Findings and selected evidence
are published through the existing Project Findings pipeline.

Frozen test copies live in testdata/configs-scan/audit-openapi-scan. The real
process gate is `make test-openapi-audit-scan-e2e`, using a disposable database
and its own local target.
