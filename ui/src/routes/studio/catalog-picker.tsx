import { useInfiniteQuery } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import { usePublicAPI } from "../../api/context";
import {
  listConfigurations,
  listWorkflows,
  type ConfigurationResource,
} from "../../api/workflows";
import { Dialog, DialogHeader } from "../../app/dialog";
import { nextPageCursor, type PageContinuation } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { Button, Field } from "../../ui";
import { record, textValue } from "./document";

export type CatalogKind =
  | "agent-templates"
  | "workflows"
  | "model-policies"
  | "llm-gateways"
  | "execution-configs";
const titles: Record<CatalogKind, string> = {
  "agent-templates": "Choose an agent template",
  workflows: "Choose a Workflow",
  "model-policies": "Choose a model policy",
  "llm-gateways": "Choose an LLM gateway",
  "execution-configs": "Choose an execution configuration",
};
interface Choice {
  selector: string;
  description: string;
  facts: string;
}
interface ChoicePage {
  items: Choice[];
  page: PageContinuation;
}
function configurationChoice(item: ConfigurationResource): Choice {
  const body = record(item.body);
  return {
    selector: `${item.ref.name}@${item.ref.version}`,
    description: textValue(body.description) || textValue(body.model),
    facts:
      item.ref.kind === "agent-templates"
        ? `${textValue(body.runtime)} · ${Array.isArray(body.toolsets) ? body.toolsets.length : 0} toolsets · ${Array.isArray(body.skills) ? body.skills.length : 0} skills`
        : item.ref.kind === "model-policies" && body.contextWindowTokens
          ? `${textValue(body.contextWindowTokens)} context tokens`
          : "Published version",
  };
}

/** Manual selectors keep working offline. The catalog only reads when opened. */
export function CatalogField({
  label,
  kind,
  value,
  onChange,
}: {
  label: string;
  kind: CatalogKind;
  value: string;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="studio-catalog-field">
      <Field label={label} className="studio-field">
        {(control) => (
          <input
            {...control}
            key={value}
            defaultValue={value}
            onBlur={(event) => {
              if (event.target.value !== value) onChange(event.target.value);
            }}
          />
        )}
      </Field>
      <Button
        aria-label={`Choose ${label.toLowerCase()} from catalog`}
        onClick={() => setOpen(true)}
      >
        Choose from catalog
      </Button>
      {open ? (
        <CatalogPicker
          kind={kind}
          current={value}
          onClose={() => setOpen(false)}
          onChoose={(selector) => {
            if (selector !== value) onChange(selector);
            setOpen(false);
          }}
        />
      ) : null}
    </div>
  );
}

function CatalogPicker({
  kind,
  current,
  onChoose,
  onClose,
}: {
  kind: CatalogKind;
  current: string;
  onChoose: (selector: string) => void;
  onClose: () => void;
}) {
  const api = usePublicAPI(),
    heading = useId(),
    search = useRef<HTMLInputElement>(null);
  const [input, setInput] = useState("");
  const [q, setQuery] = useState("");
  const query = useInfiniteQuery({
    queryKey: ["studio", "catalog", kind, q],
    initialPageParam: null as string | null,
    queryFn: async ({ pageParam, signal }): Promise<ChoicePage> => {
      const request = {
        signal,
        ...(q ? { q } : {}),
        ...(pageParam === null ? {} : { cursor: pageParam }),
      };
      if (kind === "workflows") {
        const page = await listWorkflows(api, request);
        return {
          page: page.page,
          items: page.items.map((item) => ({
            selector: `${item.ref.name}@${item.ref.version}`,
            description: item.presentation?.description ?? "",
            facts: `${Object.keys(item.inputs).length} inputs · ${Object.keys(item.outputs).length} outputs · entry ${item.entryStage}`,
          })),
        };
      }
      const page = await listConfigurations(api, kind, request);
      return { page: page.page, items: page.items.map(configurationChoice) };
    },
    getNextPageParam: (last, pages, _lastParam, params) => {
      const cursor = nextPageCursor(last.page);
      return cursor && pages.length < 20 && !params.includes(cursor)
        ? cursor
        : undefined;
    },
    retry: false,
    refetchOnWindowFocus: false,
  });
  return (
    <Dialog
      className="ui-dialog ui-panel studio-catalog-dialog"
      labelledBy={heading}
      initialFocusRef={search}
      onRequestClose={onClose}
    >
      <DialogHeader
        id={heading}
        title={titles[kind]}
        eyebrow="Published catalog"
        close={{ label: "Close catalog", onClose }}
      />
      <p>
        Choose an exact published version for this reference. Current:{" "}
        {current || "unset"}.
      </p>
      <form
        className="studio-catalog-search"
        onSubmit={(event) => {
          event.preventDefault();
          setQuery(input.trim());
        }}
      >
        <Field label="Search catalog" className="studio-field">
          {(control) => (
            <input
              {...control}
              ref={search}
              value={input}
              maxLength={200}
              placeholder="Name, version or description"
              onChange={(event) => setInput(event.target.value)}
            />
          )}
        </Field>
        <Button type="submit">Search</Button>
      </form>
      {/* A failed next page keeps the loaded versions and is reported under
          them, so QueryView only sees the first page's error.
          V385: needs shared continuation-error state in QueryView */}
      <QueryView
        query={{
          data: query.data,
          error: query.isFetchNextPageError ? null : query.error,
          isFetching: query.isFetching,
          refetch: query.refetch,
        }}
        loading={<p role="status">Loading catalog…</p>}
        errorContext="Catalog could not be loaded."
      >
        {(data) => {
          const items = new Map(
            data.pages.flatMap((page) =>
              page.items.map((item) => [item.selector, item] as const),
            ),
          );
          const tail = data.pages.at(-1)?.page;
          return (
            <>
              <p role="status">
                {items.size} loaded versions{q ? ` matching “${q}”` : ""}.
                {tail?.hasMore
                  ? " More versions remain."
                  : " All matching versions loaded."}
              </p>
              {items.size === 0 ? (
                <p>No published versions match this search.</p>
              ) : null}
              <ul
                className="studio-catalog-results"
                aria-label="Published versions"
              >
                {[...items.values()].map((item) => (
                  <li key={item.selector}>
                    <div>
                      <strong>{item.selector}</strong>
                      {item.description ? <p>{item.description}</p> : null}
                      <small>{item.facts}</small>
                      {item.selector === current ? (
                        <small>Current reference</small>
                      ) : null}
                    </div>
                    <Button
                      onClick={() => onChoose(item.selector)}
                      aria-label={`Use ${item.selector}`}
                    >
                      Use version
                    </Button>
                  </li>
                ))}
              </ul>
              {query.isFetchNextPageError ? (
                <p role="alert">
                  More versions could not be loaded. {query.error?.message}{" "}
                  <Button
                    disabled={query.isFetching}
                    onClick={() => void query.fetchNextPage()}
                  >
                    Retry catalog
                  </Button>
                </p>
              ) : null}
              {query.hasNextPage ? (
                <Button
                  pending={query.isFetchingNextPage}
                  pendingLabel="Loading versions…"
                  disabled={query.isFetching}
                  onClick={() => void query.fetchNextPage()}
                >
                  Load more versions
                </Button>
              ) : null}
              {tail?.hasMore && !query.hasNextPage ? (
                <p role="alert">
                  The remaining catalog could not be continued within this view.
                  Narrow the search to find another version.
                </p>
              ) : null}
            </>
          );
        }}
      </QueryView>
      <div className="ui-dialog-actions">
        <Button onClick={onClose}>Cancel</Button>
      </div>
    </Dialog>
  );
}
