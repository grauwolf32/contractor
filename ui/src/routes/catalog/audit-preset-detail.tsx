import { useQuery } from "@tanstack/react-query";
import { useLocation, useNavigate, useParams } from "react-router";

import { auditPresetPath } from "../../api/audit-presets";
import { getAuditProfile, type AuditProfile } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { CONFIG_ID_PATTERN, CONFIG_VERSION_PATTERN } from "../../api/workflows";
import { ContextLink, ReturnLink } from "../../app/context-navigation";
import { ErrorNotice } from "../artifacts/common";
import { auditPresetLabel } from "../projects/audits/labels";
import { compareWorkflowVersions } from "../workflows/families";
import { AuditPresetStandardChecks } from "./audit-preset-checks";
import {
  auditModeLabels,
  presetScope,
  useAuditPresets,
} from "./audit-preset-data";
import { useCatalogQueryState } from "./query-state";

function PresetVersionPicker({ profile }: { profile: AuditProfile }) {
  const query = useAuditPresets();
  const navigate = useNavigate();
  const location = useLocation();
  const versions = new Set([profile.ref.version]);
  for (const item of query.data ?? []) {
    if (item.ref.name === profile.ref.name) versions.add(item.ref.version);
  }
  return (
    <div className="catalog-version-picker">
      <label>
        Preset version
        <select
          value={profile.ref.version}
          onChange={(event) =>
            void navigate(
              auditPresetPath(profile.ref.name, event.target.value),
              { state: location.state },
            )
          }
        >
          {[...versions]
            .sort((a, b) => compareWorkflowVersions(b, a))
            .map((version) => (
              <option key={version} value={version}>
                {version}
              </option>
            ))}
        </select>
      </label>
      {query.isPending ? <p role="status">Loading versions…</p> : null}
      {query.error ? <ErrorNotice error={query.error} /> : null}
    </div>
  );
}

function PresetContents({ profile }: { profile: AuditProfile }) {
  const search = useCatalogQueryState();
  const fixedChecks =
    profile.inventory.implementation === "standard-mappings@1";
  const selection = profile.inventory.standardSelection;
  return (
    <>
      <PresetVersionPicker profile={profile} />
      <div className="catalog-audit-summary">
        <span className="state-badge">{auditModeLabels[profile.mode]}</span>
        <span>
          {profile.serverCompatible
            ? "Available on this server"
            : "Unavailable on this server"}
        </span>
      </div>
      {!profile.serverCompatible ? (
        <div className="notice">
          <strong>This preset cannot run on this server.</strong>
          <ul>
            {profile.compatibilityReasons.map((reason) => (
              <li key={reason}>{reason.replaceAll("_", " ")}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <section className="catalog-audit-checks" aria-label="Preset checks">
        <header className="route-header-row catalog-discovery-header">
          <div>
            <h3>Checks</h3>
            <p>{presetScope(profile)}</p>
            {selection ? (
              <p className="muted-copy">
                {selection.entryIds.length} selected requirements · Levels{" "}
                {selection.levels.join(", ")}
              </p>
            ) : null}
          </div>
          {fixedChecks ? (
            <label className="catalog-search">
              Search checks
              <input
                type="search"
                placeholder="ID, title or check text"
                value={search.draftSearch}
                onChange={(event) =>
                  search.changeDraftSearch(event.target.value)
                }
              />
            </label>
          ) : null}
        </header>
        {fixedChecks ? (
          profile.standards.map((reference) => (
            <AuditPresetStandardChecks
              key={`${reference.scheme}@${reference.version}`}
              reference={reference}
              profile={profile}
              search={search.committedSearch.toLowerCase()}
            />
          ))
        ) : (
          <div className="panel">
            <strong>The check list depends on your audit inputs.</strong>
            <p>
              {profile.inventory.source?.source === "prepare-output"
                ? "Preparation generates the inventory from your project inputs."
                : `Supply ${profile.inventory.source?.name ?? "the required inputs"} when creating an audit in a project.`}{" "}
              The generated checks will be available in that audit’s Coverage
              tab.
            </p>
          </div>
        )}
      </section>
      <section
        className="panel catalog-audit-inputs"
        aria-label="Preset inputs"
      >
        <h3>Inputs</h3>
        <ul className="catalog-slot-list">
          {Object.entries(profile.inputs).map(([name, input]) => (
            <li key={name}>
              <code>{name}</code>
              <span>{input.required ? "Required" : "Optional"}</span>
              <small>{input.mediaTypes.join(", ")}</small>
            </li>
          ))}
        </ul>
      </section>
      {Object.keys(profile.workflows ?? {}).length > 0 ? (
        <section className="panel" aria-label="Preset workflows">
          <h3>Workflows</h3>
          <ul className="catalog-slot-list">
            {Object.entries(profile.workflows ?? {}).map(([role, binding]) => (
              <li key={role}>
                <ContextLink
                  to={`/catalog/workflows/${encodeURIComponent(binding.workflow.name)}/${encodeURIComponent(binding.workflow.version)}`}
                  returnLabel={`${auditPresetLabel(profile.ref.name)} @${profile.ref.version}`}
                >
                  {binding.workflow.name}@{binding.workflow.version}
                </ContextLink>
                <span>
                  {role} · {binding.kind}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </>
  );
}

export function AuditPresetDetailRoute() {
  const { name = "", version = "" } = useParams();
  const api = usePublicAPI();
  const valid =
    CONFIG_ID_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.auditProfiles.detail(name, version),
    queryFn: ({ signal }) => getAuditProfile(api, name, version, signal),
    enabled: valid,
  });
  return (
    <section className="catalog-audit-preset-detail">
      <ReturnLink to="/catalog/audit-presets" label="Audit presets" />
      <header>
        <p className="eyebrow">Audit preset</p>
        <h2>
          {auditPresetLabel(name)}
          <span className="catalog-version-label">@{version}</span>
        </h2>
        <code className="catalog-exact-selector">
          {name}@{version}
        </code>
      </header>
      {!valid ? (
        <ErrorNotice error={new Error("Audit preset version is invalid")} />
      ) : query.isPending ? (
        <p role="status">Loading audit preset…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : (
        <PresetContents key={`${name}@${version}`} profile={query.data} />
      )}
    </section>
  );
}
