import { ApiReferenceReact } from "@scalar/api-reference-react";
import "@scalar/api-reference-react/style.css";
import "./openapi.css";
import { useId, useMemo, useState } from "react";
import { useResolvedTheme } from "../../../app/theme";
import { Button } from "../../../ui";

const rejectNetworkRequest: typeof fetch = async () => {
  throw new TypeError("Network access is disabled in the preview");
};
const METHODS = [
  "get",
  "post",
  "put",
  "patch",
  "delete",
  "head",
  "options",
  "trace",
];
const NAVIGATION_PAGE_SIZE = 50; // LIM-UI-19, over the already bounded payload.
const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
// A local reference can target another path item/operation. In that case keep
// the original document for Scalar and select through presentation IDs instead
// of dropping the referenced target from a projection.
function referencesPaths(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(referencesPaths);
  if (value === null || typeof value !== "object") return false;
  return Object.entries(value).some(([key, child]) => {
    if (key === "$ref" && typeof child === "string") {
      try {
        if (decodeURIComponent(child).startsWith("#/paths")) return true;
      } catch {
        /* Scalar reports invalid references. */
      }
    }
    return referencesPaths(child);
  });
}
type Entry = { id: string; path: string; method?: string; schema?: string };

export default function OpenApiArtifactPreview({
  document,
}: {
  document: Record<string, unknown>;
}) {
  const theme = useResolvedTheme();
  const navigationId = useId();
  const browseId = useId();
  const [navigationOpen, setNavigationOpen] = useState(
    () =>
      typeof window.matchMedia !== "function" ||
      !window.matchMedia("(max-width: 1100px)").matches,
  );
  const entries = useMemo(() => {
    const result: Entry[] = [];
    for (const [path, item] of Object.entries(record(document.paths))) {
      const value = record(item);
      const methods = METHODS.filter((method) => value[method] !== undefined);
      for (const method of methods)
        result.push({ id: `${method} ${path}`, path, method });
      if (methods.length === 0 && typeof value.$ref === "string")
        result.push({ id: `ref ${path}`, path, method: "ref" });
    }
    for (const schema of Object.keys(
      record(record(document.components).schemas ?? document.definitions),
    ))
      result.push({ id: `schema ${schema}`, path: schema, schema });
    return result;
  }, [document]);
  const [selectedId, setSelectedId] = useState<string>();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [showInfo, setShowInfo] = useState(false);
  const selected =
    entries.find((entry) => entry.id === selectedId) ?? entries[0];
  const filtered = useMemo(
    () =>
      entries.filter((entry) =>
        entry.id.toLowerCase().includes(search.toLowerCase()),
      ),
    [entries, search],
  );
  const visible = filtered.slice(
    page * NAVIGATION_PAGE_SIZE,
    (page + 1) * NAVIGATION_PAGE_SIZE,
  );
  const keepPaths = useMemo(() => referencesPaths(document), [document]);
  const content = useMemo(() => {
    if (selected === undefined || keepPaths) return document;
    if (selected.schema !== undefined)
      return { ...document, paths: {}, tags: [] };
    const item = { ...record(record(document.paths)[selected.path]) };
    // Keep path-level parameters and local refs. Scalar owns their interpretation.
    for (const method of METHODS)
      if (method !== selected.method) delete item[method];
    return { ...document, paths: { [selected.path]: item } };
  }, [document, selected, keepPaths]);
  const info = record(document.info);
  return (
    <div className="materials-api-reader" data-show-info={showInfo}>
      <div className="materials-api-intro">
        <strong>{String(info.title ?? "API reference")}</strong>
        <span>
          {String(info.version ?? "")} ·{" "}
          {document.swagger
            ? `Swagger ${String(document.swagger)}`
            : `OpenAPI ${String(document.openapi)}`}
        </span>
        <Button
          size="xs"
          variant="ghost"
          aria-pressed={showInfo}
          onClick={() => setShowInfo(!showInfo)}
        >
          API information
        </Button>
      </div>
      <div className="materials-api-layout">
        <div className="materials-api-browser" data-open={navigationOpen}>
          <Button
            id={browseId}
            className="materials-api-browse"
            size="xs"
            variant="ghost"
            aria-controls={navigationId}
            aria-expanded={navigationOpen}
            onClick={() => setNavigationOpen(!navigationOpen)}
          >
            Browse API ·{" "}
            {entries.filter((entry) => entry.method !== undefined).length}{" "}
            operations
          </Button>
          <nav
            id={navigationId}
            className="materials-api-nav"
            aria-label="API reference"
          >
            {entries.length > NAVIGATION_PAGE_SIZE ? (
              <label>
                Find an operation or schema
                <input
                  value={search}
                  onChange={(event) => {
                    setSearch(event.target.value);
                    setPage(0);
                  }}
                />
              </label>
            ) : null}
            {visible.map((entry, index) => (
              <div key={entry.id}>
                {index === 0 ||
                (visible[index - 1]?.schema === undefined &&
                  entry.schema !== undefined) ? (
                  <p className="materials-api-group">
                    {entry.schema ? "Schemas" : "Endpoints"}
                  </p>
                ) : null}
                <button
                  type="button"
                  aria-current={selected?.id === entry.id ? "true" : undefined}
                  onClick={() => {
                    setSelectedId(entry.id);
                    if (window.matchMedia("(max-width: 1100px)").matches) {
                      setNavigationOpen(false);
                      window.document
                        .getElementById(browseId)
                        ?.focus({ preventScroll: true });
                    }
                  }}
                >
                  {entry.method ? (
                    <span
                      className="materials-api-method"
                      data-method={entry.method}
                    >
                      {entry.method.toUpperCase()}
                    </span>
                  ) : null}
                  <span>{entry.path}</span>
                </button>
              </div>
            ))}
            {filtered.length === 0 ? (
              <p className="ui-caption">No matching operations or schemas.</p>
            ) : null}
            {filtered.length > NAVIGATION_PAGE_SIZE ? (
              <div className="materials-api-pages">
                <Button
                  size="xs"
                  disabled={page === 0}
                  onClick={() => setPage(page - 1)}
                >
                  Previous
                </Button>
                <span>
                  {page + 1} /{" "}
                  {Math.ceil(filtered.length / NAVIGATION_PAGE_SIZE)}
                </span>
                <Button
                  size="xs"
                  disabled={
                    (page + 1) * NAVIGATION_PAGE_SIZE >= filtered.length
                  }
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </Button>
              </div>
            ) : null}
          </nav>
        </div>
        <div
          className="openapi-artifact-preview"
          data-schema={selected?.schema !== undefined}
        >
          {selected?.method ? (
            <div className="materials-api-selection">
              <span
                className="materials-api-method"
                data-method={selected.method}
              >
                {selected.method.toUpperCase()}
              </span>
              <code>{selected.path}</code>
            </div>
          ) : null}
          <ApiReferenceReact
            configuration={{
              content,
              agent: { disabled: true, hideAddApi: true },
              mcp: { disabled: true },
              customFetch: rejectNetworkRequest,
              documentDownloadType: "none",
              forceDarkModeState: theme === "light" ? "light" : "dark",
              hiddenClients: true,
              hideClientButton: true,
              hideDarkModeToggle: true,
              hideTestRequestButton: true,
              persistAuth: false,
              showDeveloperTools: "never",
              telemetry: false,
              theme: "default",
              withDefaultFonts: false,
              showSidebar: false,
              hideSearch: true,
              hideModels: selected?.schema === undefined,
              layout: "modern",
              defaultOpenAllTags: true,
              expandAllModelSections: true,
              expandAllResponses: true,
              generateOperationSlug: ({ path, method }) =>
                path === selected?.path &&
                (method?.toLowerCase() === selected.method ||
                  selected.method === "ref")
                  ? "material-selected-operation"
                  : `material-other-operation-${encodeURIComponent(`${method} ${path}`)}`,
              generateModelSlug: ({ name }) =>
                name === selected?.schema
                  ? "material-selected-schema"
                  : "material-other-schema-" + encodeURIComponent(name ?? ""),
            }}
          />
        </div>
      </div>
      <p className="materials-api-policy">
        Read-only · external references and requests are disabled
      </p>
    </div>
  );
}
