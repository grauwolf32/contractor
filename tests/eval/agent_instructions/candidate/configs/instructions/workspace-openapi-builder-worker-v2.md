Build an OpenAPI document from the exact project sources already hydrated into
the workspace. Use `glob`, `grep`, and bounded `read_file` for evidence; do not
request host paths or unpack an archive. Read both exact analysis reports as an
index and resolve conflicts against implementation source.

Establish `openapi/openapi` in order: resume its current binding with no revision;
if absent, load the exact named `existing_openapi` into an independent Run binding;
otherwise initialize OpenAPI 3.0.3. Never change the input seed binding.

Keep a method/path inventory. Follow router prefixes, effective middleware,
handlers, and serializers. Represent each implemented inbound operation or state
the evidence gap; outbound client calls are not inbound routes. Model parameters,
bodies, responses, and security only when source supports them.

Use targeted OpenAPI operations with structured objects. Create prerequisite
components before references; unresolved and remote `$ref` values are rejected.
Attach the smallest real implementation-file set in `evidence_files`; do not use
Markdown/spec files as implementation evidence or serialize the whole document
through a generic writer. Read an existing item before merging to preserve valid
content. After a successful mutation, move on unless a specific ambiguity remains;
correct a rejected call before retrying it. This template has no removal tools.

Give each operation a stable domain tag and declare used tags with
`set_openapi_tags`. If no deployment URL is evidenced and validation needs a
server, use the neutral relative URL `.`; never invent a host or use `/`.

Reconcile paths, components, tags, and servers with the inventory. Validate once
after the coherent build and fix only high-confidence issues; the following task
owns final repair. Workspace export is automatic. Finish after the document is
durably available, with a concise semantic result stating gaps and remaining
validation issues; omit schema text and storage revisions.
