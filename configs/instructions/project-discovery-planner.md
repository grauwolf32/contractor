Map the exact project archive supplied as `source` for downstream API and
architecture construction. Read the exact named `dependency_report` input first and
verify important claims against source instead of duplicating its full inventory.

Use the string parameter `objective`, when present, as additional emphasis only. Locate and
describe, with `relative/path:line` evidence:

- languages, frameworks, manifests, build/runtime entry points, and service
  boundaries;
- HTTP routers/controllers/handlers, route registration, middleware, and base paths;
- request/response/domain models and serialization/validation rules;
- authentication, authorization, transport security, and error handling;
- databases, queues, outbound clients, storage, and other external integrations;
- configuration/environment sources, deployment descriptors, tests, docs, and any
  existing API or architecture specifications.

Read bounded source windows around high-signal definitions. Separate confirmed facts,
inferences, and gaps. Never treat an existing OpenAPI/LikeC4 document as proof of
implementation behavior.

Write a Markdown report headed `# Project Structure and Runtime Inventory` to
`analysis/project` with media type `text/markdown`. Include an overview, executable
and service topology, inbound API surface, models, security controls, external
systems, evidence index, and unresolved questions. Finish only after that durable
write succeeds. Keep the plain-text completion summary concise and omit the report
body.
