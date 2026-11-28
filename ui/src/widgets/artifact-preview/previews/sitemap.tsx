import { useId, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router";
import { Button, EmptyState, Facts, TechnicalDetails } from "../../../ui";
import { Icon } from "../../../app/icon";
import { readableValue } from "../../../app/vocabulary";
import { formatNumber, formatTimestamp } from "../../../app/format";
import { artifactDetailPath } from "../../../app/artifact-path";
import type { SitemapDocument } from "./sitemap-document";
import "./sitemap.css";

type Group = SitemapDocument["groups"][number];
type Member = Group["members"][number];
type Provenance = SitemapDocument["sources"][number]["provenance"];
const PAGE_SIZE = 50; // LIM-UI-22, over one already bounded saved document.

function Method({ value }: { value: string }) {
  return (
    <span className="sm-method" data-method={value.toLowerCase()}>
      {value}
    </span>
  );
}
function Pill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return (
    <span className="sm-pill" data-tone={tone}>
      {children}
    </span>
  );
}
function ProvenanceLabel({ value }: { value: Provenance }) {
  return (
    <span className="sm-source" title={value.sourceId}>
      {value.kind}
      {value.deployment ? ` · ${value.deployment}` : ""}
    </span>
  );
}
function Pagination({
  page,
  count,
  onChange,
  label,
}: {
  page: number;
  count: number;
  onChange: (page: number) => void;
  label: string;
}) {
  if (count <= PAGE_SIZE) return null;
  return (
    <nav className="sm-pages" aria-label={label}>
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
function PagedRows({
  items,
  label,
}: {
  items: readonly ReactNode[];
  label: string;
}) {
  const [selectedPage, setPage] = useState(0);
  const page = Math.min(
    selectedPage,
    Math.max(0, Math.ceil(items.length / PAGE_SIZE) - 1),
  );
  return (
    <>
      <div className="sm-records">
        {items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)}
      </div>
      <Pagination
        page={page}
        count={items.length}
        onChange={setPage}
        label={label}
      />
    </>
  );
}
function memberStatuses(member: Member): number[] {
  return [
    ...new Set(
      member.summaries.map((row) => row.status).filter((status) => status > 0),
    ),
  ];
}
// One display folder level keeps long or deeply nested paths bounded. The
// complete saved route remains on every row, including case and empty segments.
function navigationFolder(route: string): string {
  const end = route.indexOf("/", 1);
  return end < 0 ? "/" : route.slice(0, end + 1);
}
function statusTone(status: number): string {
  return status >= 500
    ? "danger"
    : status >= 300
      ? "warning"
      : status >= 200
        ? "success"
        : "neutral";
}

export default function SitemapPreview({
  document,
}: {
  document: SitemapDocument;
}) {
  const browseId = useId();
  const navId = useId();
  const searchId = useId();
  const methodId = useId();
  const evidenceId = useId();
  const [search, setSearch] = useState("");
  const [method, setMethod] = useState("");
  const [evidence, setEvidence] = useState("");
  const [page, setPage] = useState(0);
  const [selectedId, setSelected] = useState<string>();
  const [memberId, setMember] = useState<string>();
  const [navigationOpen, setNavigationOpen] = useState(
    () =>
      typeof matchMedia !== "function" ||
      !matchMedia("(max-width: 820px)").matches,
  );
  const index = useMemo(() => {
    const suppressed = new Set(
      document.markers.flatMap((marker) => marker.members),
    );
    const rows = [...document.groups]
      .sort(
        (a, b) =>
          a.origin.localeCompare(b.origin) ||
          a.route.localeCompare(b.route) ||
          a.method.localeCompare(b.method),
      )
      .map((group) => ({
        group,
        recorded: group.members.some((member) => member.summaries.length > 0),
        suppressed: group.members.some((member) => suppressed.has(member.id)),
        text: [
          group.origin,
          group.route,
          group.method,
          ...group.members.flatMap((member) => [
            member.path,
            ...member.summaries.flatMap((summary) => [
              summary.provenance.kind,
              ...summary.parameters,
            ]),
          ]),
          ...group.declarations.flatMap((row) => row.parameters),
        ]
          .join(" ")
          .toLowerCase(),
      }));
    return {
      rows,
      suppressed,
      hosts: new Set(rows.map((row) => row.group.origin)),
      methods: [...new Set(rows.map((row) => row.group.method))].sort(),
      members: rows.reduce((count, row) => count + row.group.members.length, 0),
    };
  }, [document]);
  const filtered = useMemo(
    () =>
      index.rows.filter(
        (row) =>
          (!search || row.text.includes(search.toLowerCase())) &&
          (!method || row.group.method === method) &&
          (!evidence ||
            (evidence === "recorded"
              ? row.recorded
              : evidence === "declared"
                ? !row.recorded && row.group.declarations.length > 0
                : row.suppressed)),
      ),
    [index, search, method, evidence],
  );
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const current =
    filtered.find((row) => row.group.id === selectedId) ?? visible[0];
  const selected = current?.group;
  const member =
    selected?.members.find((row) => row.id === memberId) ??
    selected?.members[0];
  const hosts = new Map<string, Group[]>();
  for (const { group } of visible)
    hosts.set(group.origin, [...(hosts.get(group.origin) ?? []), group]);
  function resetFilters(change: () => void) {
    change();
    setPage(0);
    setSelected(undefined);
    setMember(undefined);
  }
  function choose(group: Group) {
    setSelected(group.id);
    setMember(undefined);
    if (matchMedia("(max-width: 820px)").matches) {
      setNavigationOpen(false);
      window.document.getElementById(browseId)?.focus({ preventScroll: true });
    }
  }
  return (
    <article className="sitemap-preview" aria-label="Saved sitemap">
      <header className="sm-overview">
        <div>
          <p className="sm-eyebrow">Saved HTTP surface</p>
          <h2>Sitemap</h2>
          <p className="sm-muted">Explore recorded routes and their sources.</p>
        </div>
        <dl className="sm-facts">
          <div>
            <dt>Origins</dt>
            <dd>{formatNumber(index.hosts.size)}</dd>
          </div>
          <div>
            <dt>Routes</dt>
            <dd>{formatNumber(document.groups.length)}</dd>
          </div>
          <div>
            <dt>Concrete paths</dt>
            <dd>{formatNumber(index.members)}</dd>
          </div>
        </dl>
      </header>
      <div className="sm-toolbar">
        <label htmlFor={searchId}>
          Find a route
          <input
            id={searchId}
            value={search}
            placeholder="Search host, path or parameter…"
            onChange={(event) =>
              resetFilters(() => setSearch(event.target.value))
            }
          />
        </label>
        <label htmlFor={methodId}>
          Method
          <select
            id={methodId}
            value={method}
            onChange={(event) =>
              resetFilters(() => setMethod(event.target.value))
            }
          >
            <option value="">All methods</option>
            {index.methods.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label htmlFor={evidenceId}>
          Records
          <select
            id={evidenceId}
            value={evidence}
            onChange={(event) =>
              resetFilters(() => setEvidence(event.target.value))
            }
          >
            <option value="">All routes</option>
            <option value="recorded">Recorded observations</option>
            <option value="declared">Declared only</option>
            <option value="suppressed">Suppressed members</option>
          </select>
        </label>
      </div>
      <div className="sm-layout">
        <aside className="sm-browser" data-open={navigationOpen}>
          <Button
            id={browseId}
            className="sm-browse"
            size="sm"
            variant="ghost"
            aria-expanded={navigationOpen}
            aria-controls={navId}
            onClick={() => setNavigationOpen(!navigationOpen)}
          >
            Browse routes · {formatNumber(filtered.length)}
            <Icon name="chevron-down" />
          </Button>
          <nav id={navId} className="sm-nav" aria-label="Sitemap routes">
            {[...hosts].map(([host, groups]) => (
              <section key={host} className="sm-host-group" aria-label={host}>
                <h3 className="sm-host">
                  <Icon name="target" />
                  <span>{host}</span>
                </h3>
                <div className="sm-branches">
                  {groups.map((group, position) => {
                    const folder = navigationFolder(group.route);
                    const previous = groups[position - 1];
                    return (
                      <div key={group.id}>
                        {folder &&
                        folder !==
                          (previous && navigationFolder(previous.route)) ? (
                          <p className="sm-folder">{folder}</p>
                        ) : null}
                        <button
                          type="button"
                          className="sm-route"
                          aria-current={
                            selected?.id === group.id ? "true" : undefined
                          }
                          onClick={() => choose(group)}
                        >
                          <Method value={group.method} />
                          <code>{group.route}</code>
                          <span className="sm-count" title="Concrete paths">
                            {group.members.length}
                          </span>
                        </button>
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
            {filtered.length === 0 ? (
              <p className="sm-muted">
                {document.groups.length
                  ? "No routes match these filters."
                  : "No routes in this saved revision."}
              </p>
            ) : null}
          </nav>
          <Pagination
            page={page}
            count={filtered.length}
            onChange={(next) => {
              setPage(next);
              setSelected(undefined);
              setMember(undefined);
            }}
            label="Route pages"
          />
          <p className="sm-list-count">
            {formatNumber(filtered.length)} of{" "}
            {formatNumber(document.groups.length)} routes
          </p>
        </aside>
        <div className="sm-details">
          {selected === undefined ? (
            <EmptyState
              title={
                document.groups.length
                  ? "No matching route"
                  : "No routes recorded"
              }
            >
              The Source tab retains the complete saved document.
            </EmptyState>
          ) : (
            <>
              <header className="sm-route-head">
                <p className="sm-eyebrow">{selected.origin}</p>
                <div className="sm-row">
                  <h3>
                    <Method value={selected.method} />
                    <code>{selected.route}</code>
                  </h3>
                  <Pill tone={current?.recorded ? "info" : "neutral"}>
                    {current?.recorded
                      ? "Recorded"
                      : selected.declarations.length
                        ? "Declared only"
                        : "No observations"}
                  </Pill>
                </div>
                <p className="sm-muted">
                  {formatNumber(selected.members.length)} concrete paths ·{" "}
                  {formatNumber(selected.declarations.length)} declarations
                  {selected.omitted > 0
                    ? ` · ${formatNumber(selected.omitted)} omitted`
                    : ""}
                </p>
              </header>
              <section className="sm-section">
                <h4>Concrete paths</h4>
                <PagedRows
                  key={selected.id}
                  label="Concrete path pages"
                  items={selected.members.map((row) => (
                    <button
                      key={row.id}
                      type="button"
                      className="sm-member"
                      aria-current={member?.id === row.id ? "true" : undefined}
                      onClick={() => setMember(row.id)}
                    >
                      <code>{row.path}</code>
                      <span>
                        {memberStatuses(row).map((status) => (
                          <Pill key={status} tone={statusTone(status)}>
                            {status}
                          </Pill>
                        ))}
                        {index.suppressed.has(row.id) ? (
                          <Pill tone="warning">Suppressed</Pill>
                        ) : null}
                        {row.summaries.length === 0 ? (
                          <span className="sm-muted">No observations</span>
                        ) : null}
                      </span>
                    </button>
                  ))}
                />
                {selected.members.length === 0 ? (
                  <p className="sm-muted">
                    No concrete paths were retained for this route.
                  </p>
                ) : null}
              </section>
              {member ? (
                <MemberDetails
                  key={member.id}
                  member={member}
                  suppressed={index.suppressed.has(member.id)}
                />
              ) : null}
              <section className="sm-section sm-declarations">
                <h4>Declarations</h4>
                {selected.declarations.length ? (
                  <PagedRows
                    key={selected.id}
                    label="Declaration pages"
                    items={selected.declarations.map((row, i) => (
                      <div className="sm-declaration" key={i}>
                        <ProvenanceLabel value={row.provenance} />
                        <code>{row.pointer || "No source pointer"}</code>
                        <p className="sm-muted">
                          {row.rule ? "Grouping rule" : "Declared route"}
                          {row.parameters.length
                            ? ` · Parameters: ${row.parameters.join(", ")}`
                            : ""}
                        </p>
                      </div>
                    ))}
                  />
                ) : (
                  <p className="sm-muted">
                    No declarations retained for this route.
                  </p>
                )}
                <p className="sm-policy">
                  A declaration describes the contract; it does not prove
                  reachability.
                </p>
              </section>
              {selected.structural.length ? (
                <section className="sm-section">
                  <h4>Structural selection</h4>
                  <Facts
                    layout="rows"
                    items={selected.structural.map((row) => ({
                      term: row.name,
                      value: row.values.join(", "),
                    }))}
                  />
                </section>
              ) : null}
            </>
          )}
        </div>
      </div>
      <div className="sm-context">
        <TechnicalDetails summary={`Sources · ${document.sources.length}`}>
          <PagedRows
            label="Source pages"
            items={document.sources.map((row) => (
              <div className="sm-source-row" key={row.provenance.sourceId}>
                <ProvenanceLabel value={row.provenance} />
                <code>{row.provenance.sourceId}</code>
                <span className="sm-muted">
                  Generation {formatNumber(row.generation)}
                </span>
                {row.artifact ? (
                  <code>
                    {row.artifact.ref.namespace}/{row.artifact.ref.name}@
                    {row.artifact.ref.revision}
                  </code>
                ) : null}
              </div>
            ))}
          />
        </TechnicalDetails>
        {document.gaps.length ||
        document.groups.some(
          (group) =>
            group.omitted > 0 ||
            group.members.some((member) => member.omitted > 0),
        ) ? (
          <TechnicalDetails summary="Recorded omissions">
            <p className="sm-muted">
              Some contributions were not retained. Counts are saved aggregates,
              not missing request counts.
            </p>
            <PagedRows
              label="Omission pages"
              items={document.gaps.map((gap, i) => (
                <p key={i}>
                  <strong>{gap.reason}</strong> · {formatNumber(gap.count)}
                  <span className="sm-muted">
                    {" "}
                    · {gap.sourceId || "Source not specified"}
                  </span>
                </p>
              ))}
            />
          </TechnicalDetails>
        ) : null}
        {document.rules.length ? (
          <TechnicalDetails
            summary={`Grouping rules · ${document.rules.length}`}
          >
            <PagedRows
              label="Grouping rule pages"
              items={document.rules.map((rule, i) => (
                <div className="sm-source-row" key={i}>
                  <Method value={rule.method} />
                  <code>
                    {rule.origin}
                    {rule.path}
                  </code>
                  <span className="sm-muted">{rule.sourceId}</span>
                  {rule.structural.map((value) => (
                    <p key={value.name}>
                      {value.name}: {value.allowed.join(", ")}
                    </p>
                  ))}
                </div>
              ))}
            />
          </TechnicalDetails>
        ) : null}
      </div>
      <p className="sm-policy">
        Saved surface metadata · does not establish current reachability or
        security coverage.
      </p>
    </article>
  );
}

function MemberDetails({
  member,
  suppressed,
}: {
  member: Member;
  suppressed: boolean;
}) {
  return (
    <>
      <section className="sm-section">
        <div className="sm-section-head">
          <h4>Recorded observations</h4>
          <code className="sm-muted">{member.path}</code>
        </div>
        {suppressed ? (
          <p className="sm-suppression">
            This concrete path has a saved suppression marker. Its retained
            metadata is shown here.
          </p>
        ) : null}
        {member.summaries.length ? (
          member.summaries.map((row, index) => (
            <article className="sm-observation" key={index}>
              <div className="sm-row">
                <div className="sm-observation-label">
                  {row.status > 0 ? (
                    <Pill tone={statusTone(row.status)}>{row.status}</Pill>
                  ) : null}
                  <strong>
                    {row.outcome === "response"
                      ? "HTTP response"
                      : readableValue(row.outcome)}
                  </strong>
                </div>
                <ProvenanceLabel value={row.provenance} />
              </div>
              <Facts
                layout="rows"
                size="sm"
                className="sm-observation-facts"
                items={[
                  {
                    term: "Parameters",
                    value: row.parameters.length
                      ? row.parameters.join(", ")
                      : "None recorded",
                  },
                  { term: "First seen", value: formatTimestamp(row.first) },
                  { term: "Last seen", value: formatTimestamp(row.last) },
                ]}
              />
            </article>
          ))
        ) : (
          <p className="sm-muted">
            No recorded observations for this concrete path.
          </p>
        )}
        {member.omitted > 0 ? (
          <p className="sm-suppression">
            {formatNumber(member.omitted)} contributions omitted from this
            member.
          </p>
        ) : null}
      </section>
      {member.annotations.length ? (
        <section className="sm-section">
          <h4>Notes & tags</h4>
          <PagedRows
            label="Annotation pages"
            items={member.annotations.map((row, i) => (
              <div className="sm-annotation" key={i}>
                <ProvenanceLabel value={row.provenance} />
                {row.note ? <p>{row.note}</p> : null}
                <div className="sm-tags">
                  {row.tags.map((tag) => (
                    <span key={tag} className="sm-tag">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          />
        </section>
      ) : null}
      {member.examples.length ? (
        <TechnicalDetails
          summary={`Retained examples · ${member.examples.length}`}
        >
          <ul className="sm-examples">
            {member.examples.map((example, i) => (
              <li key={i}>
                <code>{example}</code>
              </li>
            ))}
          </ul>
        </TechnicalDetails>
      ) : null}
      {member.evidence.length ? (
        <section className="sm-section">
          <h4>Retained evidence</h4>
          {member.evidence.map((row, i) => (
            <div className="sm-evidence" key={i}>
              <code>
                {row.artifact.ref.namespace}/{row.artifact.ref.name}@
                {row.artifact.ref.revision}
              </code>
              <Link
                to={artifactDetailPath(
                  { kind: "run", id: row.runId },
                  row.artifact.ref,
                )}
              >
                details →
              </Link>
            </div>
          ))}
        </section>
      ) : null}
    </>
  );
}
