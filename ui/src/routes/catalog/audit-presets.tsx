import { useState } from "react";

import { auditPresetPath } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice } from "../artifacts/common";
import { auditPresetLabel } from "../projects/audits/labels";
import { compareWorkflowVersions } from "../workflows/families";
import { useCatalogQueryState } from "./query-state";
import {
  auditModeLabels,
  presetScope,
  useAuditPresets,
} from "./audit-preset-data";

export function AuditPresetListRoute() {
  const query = useAuditPresets();
  const state = useCatalogQueryState();
  const [choices, setChoices] = useState<Record<string, string>>({});
  const search = state.committedSearch.toLowerCase();
  const families = new Map<string, AuditProfile[]>();
  for (const profile of query.data ?? []) {
    const searchable = [
      profile.ref.name,
      profile.ref.version,
      auditPresetLabel(profile.ref.name),
      profile.mode,
      auditModeLabels[profile.mode],
      presetScope(profile),
      ...profile.standards.map(
        (standard) => `${standard.scheme}@${standard.version}`,
      ),
    ]
      .join(" ")
      .toLowerCase();
    if (!searchable.includes(search)) continue;
    const versions = families.get(profile.ref.name) ?? [];
    versions.push(profile);
    families.set(profile.ref.name, versions);
  }

  return (
    <section className="catalog-audit-presets">
      <header className="route-header-row catalog-discovery-header">
        <div>
          <h2>Audit presets</h2>
          <p className="lede">
            Explore reusable audit setups and the checks they cover.
          </p>
        </div>
        <label className="catalog-search">
          Search audit presets
          <input
            type="search"
            value={state.draftSearch}
            placeholder="Name, version, standard or mode"
            onChange={(event) => state.changeDraftSearch(event.target.value)}
          />
        </label>
      </header>
      {query.isPending ? (
        <p role="status">Loading audit presets…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : families.size === 0 ? (
        <div className="panel compact-empty">
          <strong>
            {search
              ? "No audit presets match this search."
              : "No audit presets published."}
          </strong>
        </div>
      ) : (
        <>
          <div className="catalog-result-summary" aria-live="polite">
            <span>
              {families.size} {families.size === 1 ? "preset" : "presets"}
            </span>
          </div>
          <div className="catalog-audit-preset-grid">
            {[...families]
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([name, versions]) => {
                versions.sort((a, b) =>
                  compareWorkflowVersions(b.ref.version, a.ref.version),
                );
                const profile =
                  versions.find((item) => item.ref.version === choices[name]) ??
                  versions[0]!;
                return (
                  <article
                    className="panel catalog-audit-preset-card"
                    key={name}
                  >
                    <p className="eyebrow">{auditModeLabels[profile.mode]}</p>
                    <h3>
                      <ContextLink
                        to={auditPresetPath(name, profile.ref.version)}
                        returnLabel="Audit presets"
                      >
                        {auditPresetLabel(name)}
                      </ContextLink>
                    </h3>
                    <p>{presetScope(profile)}</p>
                    <div className="catalog-agent-version">
                      <label htmlFor={`preset-version-${name}`}>Version</label>
                      <select
                        id={`preset-version-${name}`}
                        aria-label={`Version of ${name}`}
                        value={profile.ref.version}
                        onChange={(event) =>
                          setChoices((previous) => ({
                            ...previous,
                            [name]: event.target.value,
                          }))
                        }
                      >
                        {versions.map((item) => (
                          <option
                            key={item.ref.version}
                            value={item.ref.version}
                          >
                            {item.ref.version}
                          </option>
                        ))}
                      </select>
                      <span>
                        {profile.serverCompatible
                          ? "Available"
                          : "Unavailable on this server"}
                      </span>
                    </div>
                    <code className="catalog-exact-selector">
                      {name}@{profile.ref.version}
                    </code>
                    <ContextLink
                      className="catalog-agent-open"
                      to={auditPresetPath(name, profile.ref.version)}
                      returnLabel="Audit presets"
                    >
                      View checks →
                    </ContextLink>
                  </article>
                );
              })}
          </div>
        </>
      )}
    </section>
  );
}
