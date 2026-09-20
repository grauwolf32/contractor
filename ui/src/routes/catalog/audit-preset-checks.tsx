import { useQuery } from "@tanstack/react-query";

import { getAuditStandard, type AuditStandard } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { ErrorNotice } from "../artifacts/common";

function SourceLink({
  url,
  children,
}: {
  url: string;
  children: React.ReactNode;
}) {
  return /^https?:\/\//i.test(url) ? (
    <a href={url} target="_blank" rel="noreferrer">
      {children}
    </a>
  ) : (
    <span>{children}</span>
  );
}

function StandardChecks({
  standard,
  profile,
  search,
}: {
  standard: AuditStandard;
  profile: AuditProfile;
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

  return (
    <section
      className="panel catalog-audit-standard"
      aria-label={standard.title}
    >
      <header>
        <h4>{standard.title}</h4>
        <code>
          {standard.reference.scheme}@{standard.reference.version}
        </code>
        <p>{standard.description}</p>
      </header>
      {standard.license.disclosure === "metadata" ? (
        <p className="notice">
          This standard exposes metadata only. Check texts and identifiers are
          not available in the catalog.
        </p>
      ) : standard.license.disclosure === "identifiers" ? (
        <>
          <p className="notice">
            This standard exposes requirement identifiers only. Check texts are
            not available in the catalog.
          </p>
          {identifiers.length === 0 ? (
            <p>No requirements match this search.</p>
          ) : (
            <ul className="catalog-slot-list">
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
          <p className="muted-copy" role="status">
            {search
              ? `${visible.length} of ${mappings.length}`
              : mappings.length}{" "}
            checks
          </p>
          {visible.length === 0 ? (
            <p>
              {search
                ? "No checks match this search."
                : "No checks are defined for this preset."}
            </p>
          ) : (
            <div className="catalog-audit-check-list">
              {visible.map((mapping) => {
                const contract = standard.evidenceContracts?.find(
                  (item) =>
                    item.id === mapping.evidenceContract.id &&
                    item.version === mapping.evidenceContract.version,
                );
                return (
                  <details className="catalog-audit-check" key={mapping.key}>
                    <summary>
                      <span>
                        <code>{mapping.key}</code>
                        <strong>{mapping.title}</strong>
                      </span>
                      <span className="muted-copy">
                        {mapping.method.replaceAll("-", " ")}
                      </span>
                    </summary>
                    <div className="catalog-audit-check-body">
                      <p>{mapping.objective}</p>
                      {mapping.entryIds.map((id) => {
                        const entry = entries.get(id);
                        return (
                          <div key={id}>
                            <strong>{entry?.title ?? id}</strong>
                            <p className="muted-copy">
                              <code>{id}</code>
                              {entry?.level ? ` · Level ${entry.level}` : ""}
                            </p>
                            {entry?.statement ? <p>{entry.statement}</p> : null}
                          </div>
                        );
                      })}
                      {contract ? (
                        <div>
                          <strong>Evidence</strong>
                          <p>
                            {contract.minimumEvidence}–
                            {contract.maximumEvidence} evidence items ·{" "}
                            {contract.evidenceKinds.join(", ")}
                          </p>
                          <p className="muted-copy">
                            Assessments: {contract.assessments.join(", ")}
                          </p>
                        </div>
                      ) : null}
                    </div>
                  </details>
                );
              })}
            </div>
          )}
        </>
      )}
      <footer className="catalog-audit-attribution">
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

export function AuditPresetStandardChecks({
  reference,
  profile,
  search,
}: {
  reference: AuditProfile["standards"][number];
  profile: AuditProfile;
  search: string;
}) {
  const api = usePublicAPI();
  const query = useQuery({
    queryKey: ["audit-standards", reference.scheme, reference.version],
    queryFn: ({ signal }) =>
      getAuditStandard(api, reference.scheme, reference.version, signal),
  });
  if (query.isPending)
    return (
      <p role="status">
        Loading checks from {reference.scheme}@{reference.version}…
      </p>
    );
  if (query.error) return <ErrorNotice error={query.error} />;
  return (
    <StandardChecks standard={query.data} profile={profile} search={search} />
  );
}
