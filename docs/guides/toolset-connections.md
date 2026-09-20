# Toolset connections for a Run

An AgentTemplate selects tools. A RuntimeConfig supplies the connection, and an
existing Runtime label binds that version to a Run. Every agent selecting the same
toolset ref in one Run receives the same configuration. Each allocation owns its
own client session; selected tool names remain specific to its AgentTemplate.

For example, two templates can select `search` and `fetch` from `mcp@1`:

```yaml
# Inside an AgentTemplate's spec:
runtime: adk@1
toolsets:
  - ref: mcp@1
    tools: [search, fetch]
```

Publish a connection with the ordinary Operations RuntimeConfig form or API:

```yaml
apiVersion: contractor/v1alpha1
kind: RuntimeConfig
metadata:
  name: knowledge
  version: "1"
spec:
  worker:
    toolsets:
      mcp@1:
        endpoint: https://knowledge.example.com/mcp
        credential: knowledge-access
```

`credential` is optional. For authenticated access, create a write-only Runtime
credential of kind `mcp-headers@1` before publishing. Its `material.headers` can
contain `Authorization: Bearer …` or service-specific headers. The API returns
only safe metadata. Never put credential material into the YAML, endpoint query,
Agent instructions, Workflow parameters or Run metadata labels.

Bind the published version to `knowledge`, then open **Configure Run**. **Toolset
connections** shows selected tools and all stage/agent consumers, including later
stages. Select `knowledge` once under **Runtime placement**, or use **Create or
resume MCP connection** to create the credential, config and label together.
The Run request remains:

```json
{"workflow":"research@1","runtimeLabels":["knowledge"]}
```

The form shows inherited coverage from `default`, missing configuration, explicit
clears and conflicting selected profiles. Equivalent values deduplicate even when
published in different documents. Other blocks in a mixed profile still apply;
the form lists the affected sections. A per-ref clear uses `mcp@1: null`; an empty
toolsets map has no effect. The whole `toolsets` map cannot be null.

Advanced connection fields support additional PEM CA certificates and integer
connect/request timeouts. Defaults are 10 and 60 seconds; allowed ranges are 1–60
and 1–600. Allocation and invocation deadlines can shorten these limits. The
endpoint must be an HTTP(S) URL with no userinfo, query or fragment. HTTP is useful
for controlled private test networks; use HTTPS for connections carrying secrets.

Creation preserves one operation identity per API step. After a lost response,
retry the same request; completed steps are not recreated. Header values are
cleared on submission and cannot be read back. Navigation within the application
retains only operation fingerprints and confirmed references. Re-enter any
unconfirmed endpoint/settings or secret material when reopening the form. A full
browser reload loses that transient progress; inspect published resources in
Operations before starting a new creation. Partial publication never deletes
resources automatically. The wizard creates a new label; retarget a shared label
explicitly in Operations using its current revision.

A binding change affects new Runs. Existing Runs, later stages and replacement
allocations keep their pinned version. An exact retry of Run creation returns the
original Run. **Configure another Run** creates a Repeat draft that requires review
of current bindings and retained selections before submission.

The UI does not test MCP connectivity. Runtime preparation initializes the
Streamable HTTP session and validates discovery before creating the Worker. Check
Run details and allocation diagnostics for preparation failures. The selected
server must expose the exact tools and supported schemas; merely saving a URL
does not prove that the server can be used.

See [supported boundaries and diagnostics](../operations/toolset-connections.md)
and the [configuration contract](../spec/07-runtime-labels-and-infrastructure-config.md).
