import { useId, useMemo, useState } from "react";
import {
  Button,
  EmptyState,
  Facts,
  IdChip,
  MethodChip,
  StatusChip,
} from "../../../ui";
import { Icon } from "../../../app/icon";
import { formatBytes, formatNumber } from "../../../app/format";
import { readableValue } from "../../../app/vocabulary";
import { PanelTabs } from "../tabs";
import {
  ENGAGEMENT_METHODS,
  type EngagementDocument,
  type EngagementRoute,
} from "./engagement-document";
import "./engagement.css";

const PAGE_SIZE = 50; // LIM-UI-23, over the ordinary bounded Artifact preview.
const LIMITS = [
  {
    key: "requestsPerSecond",
    label: "Requests per second",
    unit: "requests/s",
  },
  { key: "concurrentRequests", label: "Concurrent requests", unit: "requests" },
  { key: "requestBodyBytes", label: "Request body", unit: "bytes" },
  { key: "responseBodyBytes", label: "Response body", unit: "bytes" },
] as const;

function Pages({
  page,
  count,
  label,
  onChange,
}: {
  page: number;
  count: number;
  label: string;
  onChange: (page: number) => void;
}) {
  if (count <= PAGE_SIZE) return null;
  return (
    <nav className="ep-pages" aria-label={label}>
      <Button
        size="xs"
        variant="ghost"
        disabled={page === 0}
        onClick={() => onChange(page - 1)}
      >
        Previous
      </Button>
      <span>
        {page + 1} / {Math.ceil(count / PAGE_SIZE)}
      </span>
      <Button
        size="xs"
        variant="ghost"
        disabled={(page + 1) * PAGE_SIZE >= count}
        onClick={() => onChange(page + 1)}
      >
        Next
      </Button>
    </nav>
  );
}
function Values({ title, values }: { title: string; values: string[] }) {
  const [page, setPage] = useState(0);
  return (
    <section className="ep-values" aria-label={title}>
      <h4>
        {title} <span>{formatNumber(values.length)}</span>
      </h4>
      <ul>
        {values
          .slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
          .map((value, index) => (
            <li key={index}>
              <code>{value}</code>
            </li>
          ))}
      </ul>
      <Pages
        page={page}
        count={values.length}
        label={`${title} pages`}
        onChange={setPage}
      />
    </section>
  );
}
function Methods({ methods }: { methods: string[] }) {
  const all = ENGAGEMENT_METHODS.every((method) => methods.includes(method));
  return (
    <div className="ep-methods" aria-label="HTTP methods">
      {all ? (
        <StatusChip tone="info" glyph={false} size="sm">
          All HTTP methods
        </StatusChip>
      ) : (
        methods.map((method) => (
          <span key={method} data-method={method}>
            <MethodChip method={method} />
          </span>
        ))
      )}
    </div>
  );
}
function Egress({ document }: { document: EngagementDocument }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("");
  const [page, setPage] = useState(0);
  const [chosen, setChosen] = useState<string>();
  const [open, setOpen] = useState(
    () =>
      typeof matchMedia !== "function" ||
      !matchMedia("(max-width: 820px)").matches,
  );
  const rows = useMemo(
    () => [
      ...document.egress.routes.map((route, index) => ({
        route,
        excluded: false,
        id: `route-${index}`,
      })),
      ...document.egress.exclusions.map((route, index) => ({
        route,
        excluded: true,
        id: `excluded-${index}`,
      })),
    ],
    [document],
  );
  const filtered = rows.filter(
    ({ route, excluded }) =>
      (!kind || (kind === "excluded") === excluded) &&
      [
        route.origin,
        ...route.methods,
        ...route.pathPrefixes,
        ...route.addressRanges,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const selected = filtered.find((row) => row.id === chosen) ?? visible[0];
  function choose(value: string) {
    setChosen(value);
    if (matchMedia("(max-width: 820px)").matches) {
      setOpen(false);
      window.document
        .getElementById(`${id}-browse`)
        ?.focus({ preventScroll: true });
    }
  }
  return (
    <section className="ep-egress" aria-label="Saved egress">
      <div className="ep-toolbar">
        <label htmlFor={`${id}-find`}>
          Find a route
          <input
            id={`${id}-find`}
            value={query}
            placeholder="Origin, path, method or network…"
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
              setChosen(undefined);
            }}
          />
        </label>
        <label htmlFor={`${id}-kind`}>
          Show
          <select
            id={`${id}-kind`}
            value={kind}
            onChange={(event) => {
              setKind(event.target.value);
              setPage(0);
              setChosen(undefined);
            }}
          >
            <option value="">All routes</option>
            <option value="allowed">Allowed routes</option>
            <option value="excluded">Exclusions</option>
          </select>
        </label>
      </div>
      <div className="ep-layout">
        <aside className="ep-browser" data-open={open}>
          <Button
            id={`${id}-browse`}
            className="ep-browse"
            size="sm"
            variant="ghost"
            aria-expanded={open}
            aria-controls={`${id}-routes`}
            onClick={() => setOpen(!open)}
          >
            Browse routes · {formatNumber(filtered.length)}
            <Icon name="chevron-down" />
          </Button>
          <nav
            id={`${id}-routes`}
            className="ep-nav"
            aria-label="Engagement routes"
          >
            {visible.map((row) => (
              <button
                type="button"
                key={row.id}
                className="ep-route"
                aria-current={row.id === selected?.id ? "true" : undefined}
                onClick={() => choose(row.id)}
              >
                <span className="ep-route-label">
                  <StatusChip
                    tone={row.excluded ? "warning" : "info"}
                    glyph={false}
                    size="sm"
                  >
                    {row.excluded ? "Exclude" : "Allow"}
                  </StatusChip>
                  <span>{row.route.methods.length} methods</span>
                </span>
                <strong>{row.route.origin}</strong>
                <code>
                  {row.route.pathPrefixes[0]}
                  {row.route.pathPrefixes.length > 1
                    ? ` · +${row.route.pathPrefixes.length - 1}`
                    : ""}
                </code>
              </button>
            ))}
            {!filtered.length ? (
              <p className="ui-caption">No routes match these filters.</p>
            ) : null}
          </nav>
          <Pages
            page={page}
            count={filtered.length}
            label="Egress route pages"
            onChange={(next) => {
              setPage(next);
              setChosen(undefined);
            }}
          />
          <p className="ep-list-count">
            {formatNumber(filtered.length)} of {formatNumber(rows.length)}{" "}
            routes
          </p>
        </aside>
        {selected ? (
          <RouteDetails
            key={selected.id}
            route={selected.route}
            excluded={selected.excluded}
          />
        ) : (
          <EmptyState title="No matching route">
            Change the search or route filter.
          </EmptyState>
        )}
      </div>
    </section>
  );
}
function RouteDetails({
  route,
  excluded,
}: {
  route: EngagementRoute;
  excluded: boolean;
}) {
  return (
    <article
      className="ep-detail"
      data-excluded={excluded}
      aria-label="Selected egress route"
    >
      <header className="ep-detail-head">
        <div className="ep-title-row">
          <Icon name="target" size={20} />
          <StatusChip
            tone={excluded ? "warning" : "info"}
            glyph={false}
            size="sm"
          >
            {excluded ? "Exclusion" : "Allowed route"}
          </StatusChip>
        </div>
        <h3>{route.origin}</h3>
        <p>
          {excluded
            ? "Matching requests are excluded from the saved allowed scope."
            : "Requests must match this origin, a path prefix, method and address range."}
        </p>
        <Methods methods={route.methods} />
      </header>
      <div className="ep-route-values">
        <Values title="Path prefixes" values={route.pathPrefixes} />
        <Values title="Address ranges" values={route.addressRanges} />
      </div>
    </article>
  );
}
function Identities({ document }: { document: EngagementDocument }) {
  return (
    <section className="ep-section" aria-label="Saved identities">
      <header>
        <h3>Identities</h3>
        <p>
          Named identities and their permitted origins. Credentials remain
          references.
        </p>
      </header>
      <div className="ep-identities">
        {document.identities.map((identity, index) => (
          <article className="ep-identity" key={index}>
            <header>
              <div className="ep-identity-icon">
                <Icon
                  name={identity.kind === "anonymous" ? "user" : "lock"}
                  size={20}
                />
              </div>
              <div>
                <h4>{identity.id}</h4>
                <p>{readableValue(identity.kind)}</p>
              </div>
              <StatusChip
                tone={identity.kind === "anonymous" ? "neutral" : "info"}
                glyph={false}
                size="sm"
              >
                {identity.kind === "anonymous"
                  ? "No credential"
                  : "Credential ref"}
              </StatusChip>
            </header>
            <Facts
              layout="rows"
              size="sm"
              items={[
                ...(identity.role
                  ? [{ term: "Role", value: identity.role }]
                  : []),
                ...(identity.tenant
                  ? [{ term: "Tenant", value: identity.tenant }]
                  : []),
                ...(identity.credential
                  ? [
                      {
                        term: "Credential",
                        value: (
                          <IdChip
                            value={identity.credential.credentialId}
                            label="credential ID"
                            display={identity.credential.credentialId}
                            wrap
                          />
                        ),
                      },
                      {
                        term: "Credential kind",
                        value: identity.credential.kind,
                      },
                    ]
                  : []),
              ]}
            />
            <Values title="Permitted origins" values={identity.origins} />
          </article>
        ))}
      </div>
    </section>
  );
}
function Limits({ document }: { document: EngagementDocument }) {
  return (
    <section className="ep-section" aria-label="Saved HTTP limits">
      <header>
        <h3>HTTP limits</h3>
        <p>
          Optional per-allocation limits. Run and Audit budgets apply
          separately.
        </p>
      </header>
      <div className="ep-limits">
        {LIMITS.map(({ key, label, unit }) => {
          const value = document.limits?.[key];
          return (
            <article
              key={key}
              className="ep-limit"
              data-configured={value !== undefined}
            >
              <h4>{label}</h4>
              <strong>
                {value === undefined
                  ? "Not set"
                  : unit === "bytes"
                    ? formatBytes(value)
                    : formatNumber(value)}
              </strong>
              {value === undefined ? (
                <p>No engagement limit configured</p>
              ) : (
                <p>
                  {unit === "bytes" ? `${formatNumber(value)} bytes` : unit}
                </p>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
export default function EngagementPreview({
  document,
}: {
  document: EngagementDocument;
}) {
  return (
    <article className="engagement-preview" aria-label="Saved engagement">
      <header className="ep-overview">
        <div className="ep-heading">
          <span className="ep-heading-icon">
            <Icon name="lock" size={24} />
          </span>
          <div>
            <p className="ep-eyebrow">Saved network policy</p>
            <h2>Engagement</h2>
            <p>HTTP scope, identities and traffic limits.</p>
          </div>
        </div>
        <dl className="ep-counts">
          <div>
            <dt>Allowed routes</dt>
            <dd>{formatNumber(document.egress.routes.length)}</dd>
          </div>
          <div>
            <dt>Exclusions</dt>
            <dd>{formatNumber(document.egress.exclusions.length)}</dd>
          </div>
          <div>
            <dt>Identities</dt>
            <dd>{formatNumber(document.identities.length)}</dd>
          </div>
        </dl>
      </header>
      <PanelTabs
        label="Engagement sections"
        tabs={[
          {
            id: "egress",
            label: "Egress",
            panel: <Egress document={document} />,
          },
          {
            id: "identities",
            label: "Identities",
            panel: <Identities document={document} />,
          },
          {
            id: "limits",
            label: "Limits",
            panel: <Limits document={document} />,
          },
        ]}
      />
      <p className="ep-policy">
        Saved configuration · scope and credentials are validated before
        execution.
      </p>
    </article>
  );
}
