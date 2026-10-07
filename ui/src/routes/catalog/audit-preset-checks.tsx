import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { getAuditStandard, type AuditStandard } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../../app/error-notice";
import {
  capitalize,
  COVERAGE_STATUS_LABELS,
  itemNoun,
  type ItemKind,
} from "../../app/vocabulary";

export function SourceLink({
  url,
  children,
}: {
  url: string;
  children: ReactNode;
}) {
  return /^https?:\/\//i.test(url) ? (
    <a href={url} target="_blank" rel="noreferrer">
      {children}
    </a>
  ) : (
    <span>{children}</span>
  );
}

type Mapping = NonNullable<AuditStandard["mappings"]>[number];
type EvidenceContract = NonNullable<AuditStandard["evidenceContracts"]>[number];

const METHOD_LABELS: Readonly<Record<Mapping["method"], string>> = {
  "source-analysis": "Source analysis",
  "configuration-review": "Configuration review",
  "documentation-review": "Documentation review",
  "active-test": "Active test",
  "manual-review": "Manual review",
};

const DISCLOSURE_LABELS: Readonly<
  Record<AuditStandard["license"]["disclosure"], string>
> = {
  full: "Full text",
  identifiers: "Identifiers only",
  metadata: "Metadata only",
};

const EVIDENCE_KIND_LABELS: Readonly<
  Record<EvidenceContract["evidenceKinds"][number], string>
> = {
  artifact: "file",
  observation: "observation",
  "tool-result": "tool result",
  "manual-attestation": "manual attestation",
  "runtime-metric": "runtime metric",
};

const HUMAN_REVIEW_LABELS: Readonly<
  Record<EvidenceContract["humanReview"], string>
> = {
  never: "Not needed",
  "on-inconclusive": "When the result is inconclusive",
  required: "Always",
};

function labelOf<K extends string>(
  table: Readonly<Record<K, string>>,
  value: K,
): string {
  return Object.hasOwn(table, value)
    ? table[value]
    : capitalize(value.replaceAll(/[_-]+/g, " "));
}

/** Coverage words from the shared vocabulary ("Met", "Issue found", …). */
function outcomeLabel(assessment: string): string {
  return Object.hasOwn(COVERAGE_STATUS_LABELS, assessment)
    ? COVERAGE_STATUS_LABELS[assessment as keyof typeof COVERAGE_STATUS_LABELS]
        .label
    : capitalize(assessment.replaceAll("-", " "));
}

function EvidenceContractFacts({ contract }: { contract: EvidenceContract }) {
  return (
    <dl className="library-evidence">
      <div>
        <dt>Evidence</dt>
        <dd>
          {contract.minimumEvidence}–{contract.maximumEvidence} evidence items ·{" "}
          {contract.evidenceKinds
            .map((kind) => labelOf(EVIDENCE_KIND_LABELS, kind))
            .join(", ")}
        </dd>
      </div>
      <div>
        <dt>Possible outcomes</dt>
        <dd>{contract.assessments.map(outcomeLabel).join(", ")}</dd>
      </div>
      <div>
        <dt>Human review</dt>
        <dd>{labelOf(HUMAN_REVIEW_LABELS, contract.humanReview)}</dd>
      </div>
    </dl>
  );
}

function StandardItems({
  standard,
  profile,
  kind,
  search,
}: {
  standard: AuditStandard;
  profile: AuditProfile;
  kind: ItemKind;
  search: string;
}) {
  const selection = profile.inventory.standardSelection;
  const selected = selection ? new Set(selection.entryIds) : undefined;
  const entries = new Map(
    (standard.entries ?? []).map((entry) => [entry.id, entry]),
  );
  const mappings = (standard.mappings ?? []).filter(
    (mapping) =>
      selected === undefined || mapping.entryIds.some((id) => selected.has(id)),
  );
  const visible = mappings.filter((mapping) =>
    [
      mapping.key,
      mapping.title,
      mapping.objective,
      mapping.method,
      ...mapping.entryIds.flatMap((id) => [
        id,
        entries.get(id)?.title ?? "",
        entries.get(id)?.statement ?? "",
      ]),
    ]
      .join(" ")
      .toLowerCase()
      .includes(search),
  );
  const identifiers = (standard.entries ?? []).filter(
    (entry) =>
      (selected === undefined || selected.has(entry.id)) &&
      entry.id.toLowerCase().includes(search),
  );
  const noun = itemNoun(kind, 1);
  const plural = itemNoun(kind, 2);

  return (
    <section className="library-standard" aria-label={standard.title}>
      <header className="library-standard-header">
        <h4>{standard.title}</h4>
        <p className="library-standard-meta">
          <code>
            {standard.reference.scheme}@{standard.reference.version}
          </code>
          <span>{DISCLOSURE_LABELS[standard.license.disclosure]}</span>
        </p>
        {standard.description ? <p>{standard.description}</p> : null}
      </header>
      {standard.license.disclosure === "metadata" ? (
        <p className="library-note">
          This standard exposes metadata only. {capitalize(noun)} texts and
          identifiers are not available in the Library.
        </p>
      ) : standard.license.disclosure === "identifiers" ? (
        <>
          <p className="library-note">
            This standard exposes identifiers only. {capitalize(noun)} texts are
            not available in the Library.
          </p>
          {identifiers.length === 0 ? (
            <p className="library-muted">No {plural} match this search.</p>
          ) : (
            <ul className="library-identifiers">
              {identifiers.map((entry) => (
                <li key={entry.id}>
                  <code>{entry.id}</code>
                  <span>{entry.kind}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <>
          <p className="library-count" role="status">
            {search
              ? `${visible.length} of ${mappings.length}`
              : mappings.length}{" "}
            {itemNoun(kind, mappings.length)}
          </p>
          {visible.length === 0 ? (
            <p className="library-muted">
              {search
                ? `No ${plural} match this search.`
                : `This check type defines no ${plural}.`}
            </p>
          ) : (
            <div className="library-items">
              {visible.map((mapping) => {
                const contract = standard.evidenceContracts?.find(
                  (item) =>
                    item.id === mapping.evidenceContract.id &&
                    item.version === mapping.evidenceContract.version,
                );
                return (
                  <details className="library-item" key={mapping.key}>
                    <summary>
                      <svg
                        className="library-item-chevron"
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                        focusable="false"
                      >
                        <path d="M9.5 6l6 6-6 6" />
                      </svg>
                      <span className="library-item-heading">
                        <code className="library-item-key">{mapping.key}</code>
                        <span className="library-item-title">
                          {mapping.title}
                        </span>
                      </span>
                      <span className="library-item-method">
                        {labelOf(METHOD_LABELS, mapping.method)}
                      </span>
                    </summary>
                    <div className="library-item-body">
                      <p>{mapping.objective}</p>
                      {mapping.entryIds.map((id) => {
                        const entry = entries.get(id);
                        return (
                          <div className="library-entry" key={id}>
                            <strong>{entry?.title ?? id}</strong>
                            <p className="library-muted">
                              <code>{id}</code>
                              {entry?.level ? ` · Level ${entry.level}` : ""}
                            </p>
                            {entry?.statement ? <p>{entry.statement}</p> : null}
                          </div>
                        );
                      })}
                      {contract ? (
                        <EvidenceContractFacts contract={contract} />
                      ) : null}
                    </div>
                  </details>
                );
              })}
            </div>
          )}
        </>
      )}
      <footer className="library-attribution">
        <span>
          <SourceLink url={standard.source.url}>
            {standard.source.name}
          </SourceLink>{" "}
          ·{" "}
          <SourceLink url={standard.license.url}>
            {standard.license.id}
          </SourceLink>
        </span>
        <span>{standard.license.attribution}</span>
      </footer>
    </section>
  );
}

/** One referenced standard with the requirements (or scenarios) it maps. */
export function AuditPresetStandardChecks({
  reference,
  profile,
  kind,
  search,
}: {
  reference: AuditProfile["standards"][number];
  profile: AuditProfile;
  kind: ItemKind;
  search: string;
}) {
  const api = usePublicAPI();
  const query = useQuery({
    queryKey: queryKeys.catalog.auditStandard(
      reference.scheme,
      reference.version,
    ),
    queryFn: ({ signal }) =>
      getAuditStandard(api, reference.scheme, reference.version, signal),
  });
  if (query.isPending)
    return (
      <p role="status" className="library-muted">
        Loading {itemNoun(kind, 2)} from {reference.scheme}@{reference.version}…
      </p>
    );
  if (query.error)
    return (
      <ErrorNotice
        error={query.error}
        onRetry={() => void query.refetch()}
        retryPending={query.isFetching}
      />
    );
  return (
    <StandardItems
      standard={query.data}
      profile={profile}
      kind={kind}
      search={search}
    />
  );
}
