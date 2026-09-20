import type {
  RuntimeConfigAuthorDocument,
  RuntimeCredentialMetadata,
} from "../../../api/operations";

export type MCPConfig = NonNullable<
  NonNullable<
    NonNullable<RuntimeConfigAuthorDocument["spec"]["worker"]>["toolsets"]
  >["mcp@1"]
>;

export interface MCPConfigDraft {
  endpoint: string;
  credential: string;
  caBundlePem: string;
  connectTimeoutSeconds: string;
  requestTimeoutSeconds: string;
}

export const emptyMCPConfigDraft = (): MCPConfigDraft => ({
  endpoint: "",
  credential: "",
  caBundlePem: "",
  connectTimeoutSeconds: "10",
  requestTimeoutSeconds: "60",
});

export function buildMCPConfig(draft: MCPConfigDraft): MCPConfig {
  const bytes = (value: string) => new TextEncoder().encode(value).length;
  try {
    const url = new URL(draft.endpoint);
    if (
      !/^https?:$/.test(url.protocol) ||
      !url.hostname ||
      url.username ||
      url.password ||
      /[?#\s]/.test(draft.endpoint) ||
      bytes(draft.endpoint) > 2048
    )
      throw new Error();
  } catch {
    throw new Error(
      "MCP endpoint must be an HTTP(S) URL without userinfo, query, fragment or whitespace.",
    );
  }
  for (const [value, max, field] of [
    [draft.connectTimeoutSeconds, 60, "Connect timeout"],
    [draft.requestTimeoutSeconds, 600, "Request timeout"],
  ] as const) {
    if (!/^[0-9]+$/.test(value) || Number(value) < 1 || Number(value) > max)
      throw new Error(`${field} must be an integer from 1 to ${max} seconds.`);
  }
  if (
    draft.credential !== "" &&
    !/^[a-z][a-z0-9_-]{0,127}$/.test(draft.credential)
  )
    throw new Error("MCP credential ID is invalid.");
  if (
    draft.caBundlePem !== "" &&
    (bytes(draft.caBundlePem) > 65536 ||
      !/^(?:\s*-----BEGIN CERTIFICATE-----[A-Za-z0-9+/=\s]+-----END CERTIFICATE-----\s*){1,8}$/.test(
        draft.caBundlePem,
      ))
  )
    throw new Error(
      "CA bundle must contain 1–8 PEM certificates, at most 64 KiB. The Server validates each certificate.",
    );
  return {
    endpoint: draft.endpoint,
    ...(draft.credential === "" ? {} : { credential: draft.credential }),
    ...(draft.caBundlePem === "" ? {} : { caBundlePem: draft.caBundlePem }),
    connectTimeoutSeconds: Number(draft.connectTimeoutSeconds),
    requestTimeoutSeconds: Number(draft.requestTimeoutSeconds),
  };
}

export function MCPConfigurationFields({
  value,
  onChange,
  disabled = false,
  showCredential = true,
  credentials = [],
  idPrefix,
}: {
  value: MCPConfigDraft;
  onChange: (value: MCPConfigDraft) => void;
  disabled?: boolean;
  showCredential?: boolean;
  credentials?: readonly RuntimeCredentialMetadata[];
  idPrefix: string;
}) {
  const update = (key: keyof MCPConfigDraft, field: string) =>
    onChange({ ...value, [key]: field });
  return (
    <fieldset className="runtime-config-block" disabled={disabled}>
      <legend>MCP · Streamable HTTP</legend>
      <label>
        MCP endpoint
        <input
          type="url"
          autoComplete="off"
          maxLength={2048}
          value={value.endpoint}
          onChange={(event) => update("endpoint", event.target.value)}
          placeholder="https://tools.example.com/mcp"
        />
      </label>
      {showCredential ? (
        <label>
          Existing MCP credential (optional)
          <input
            list={`${idPrefix}-credentials`}
            autoComplete="off"
            maxLength={128}
            value={value.credential}
            onChange={(event) => update("credential", event.target.value)}
          />
          <datalist id={`${idPrefix}-credentials`}>
            {credentials
              .filter((credential) => credential.kind === "mcp-headers@1")
              .map((credential) => (
                <option
                  key={credential.credentialId}
                  value={credential.credentialId}
                />
              ))}
          </datalist>
        </label>
      ) : null}
      <details>
        <summary>Advanced connection settings</summary>
        <div className="form-grid">
          <label>
            Connect timeout (seconds)
            <input
              type="number"
              min={1}
              max={60}
              step={1}
              value={value.connectTimeoutSeconds}
              onChange={(event) =>
                update("connectTimeoutSeconds", event.target.value)
              }
            />
          </label>
          <label>
            Request timeout (seconds)
            <input
              type="number"
              min={1}
              max={600}
              step={1}
              value={value.requestTimeoutSeconds}
              onChange={(event) =>
                update("requestTimeoutSeconds", event.target.value)
              }
            />
          </label>
          <label>
            Additional CA certificates (optional)
            <textarea
              autoComplete="off"
              maxLength={65536}
              value={value.caBundlePem}
              onChange={(event) => update("caBundlePem", event.target.value)}
            />
          </label>
        </div>
      </details>
    </fieldset>
  );
}

// Code-owned form registry; a configuration document cannot install a renderer.
export const TOOLSET_CONFIGURATION_FORMS = {
  "mcp@1": MCPConfigurationFields,
} as const;
