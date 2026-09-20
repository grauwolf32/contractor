export interface HeaderDraft {
  id: number;
  name: string;
  value: string;
}

const RESERVED = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
  "accept",
  "content-type",
  "mcp-session-id",
  "mcp-protocol-version",
  "last-event-id",
]);

export function mcpHeaderMaterial(
  rows: readonly HeaderDraft[],
): Record<string, string> {
  const material: Record<string, string> = Object.create(null) as Record<
    string,
    string
  >;
  let bytes = 0;
  if (rows.length < 1 || rows.length > 32)
    throw new Error("Provide 1–32 MCP headers.");
  for (const row of rows) {
    const name = row.name.toLowerCase();
    const size = new TextEncoder().encode(row.value).length;
    if (
      !/^[!#$%&'*+.^_`|~0-9a-z-]{1,64}$/.test(name) ||
      RESERVED.has(name) ||
      name in material
    )
      throw new Error(
        "MCP header names must be unique HTTP tokens and cannot override transport or protocol headers.",
      );
    if (size < 1 || size > 4096 || /[\r\n]/.test(row.value))
      throw new Error(
        "MCP header values must contain 1–4096 bytes without line breaks.",
      );
    bytes += name.length + size;
    material[name] = row.value;
  }
  if (bytes > 16384) throw new Error("MCP headers must total at most 16 KiB.");
  return material;
}

export function HeaderFields({
  rows,
  onChange,
  disabled = false,
  maximum = 32,
}: {
  rows: readonly HeaderDraft[];
  onChange: (rows: HeaderDraft[]) => void;
  disabled?: boolean;
  maximum?: number;
}) {
  return (
    <fieldset className="runtime-secret-rows" disabled={disabled}>
      <legend>Authentication headers</legend>
      {rows.map((row) => (
        <div className="runtime-secret-row" key={row.id}>
          <label>
            Header name
            <input
              autoComplete="off"
              value={row.name}
              onChange={(event) =>
                onChange(
                  rows.map((item) =>
                    item.id === row.id
                      ? { ...item, name: event.target.value }
                      : item,
                  ),
                )
              }
            />
          </label>
          <label>
            Header value · write only
            <input
              type="password"
              autoComplete="new-password"
              value={row.value}
              onChange={(event) =>
                onChange(
                  rows.map((item) =>
                    item.id === row.id
                      ? { ...item, value: event.target.value }
                      : item,
                  ),
                )
              }
            />
          </label>
          <button
            className="secondary-button"
            type="button"
            disabled={rows.length === 1}
            aria-label={`Remove header ${row.name || row.id}`}
            onClick={() => onChange(rows.filter((item) => item.id !== row.id))}
          >
            Remove
          </button>
        </div>
      ))}
      <button
        className="secondary-button"
        type="button"
        disabled={rows.length >= maximum}
        onClick={() =>
          onChange([
            ...rows,
            {
              id: Math.max(0, ...rows.map((row) => row.id)) + 1,
              name: "",
              value: "",
            },
          ])
        }
      >
        Add header
      </button>
    </fieldset>
  );
}
