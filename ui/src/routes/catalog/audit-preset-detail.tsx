import { useQuery } from "@tanstack/react-query";
import { useLocation, useNavigate } from "react-router";
import { auditPresetPath } from "../../api/audit-presets";
import { ENGAGEMENT_MEDIA_TYPE } from "../../api/artifacts";
import { getAuditProfile, type AuditProfile } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  CONFIG_NAME_PATTERN,
  CONFIG_VERSION_PATTERN,
} from "../../api/workflows";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice } from "../../app/error-notice";
import {
  compactDigest,
  compareVersions,
  formatBytes,
  formatTimeLimit,
} from "../../app/format";
import { Icon } from "../../app/icon";
import { QueryView } from "../../app/query-view";
import { Button, ButtonLink, Facts, TechnicalDetails } from "../../ui";
import {
  presentCheckType,
  inputLabel,
  formatList,
  orderedInputs,
  scopeSummary,
} from "../checks/start/check-types";
import { startCheckPath, useAuditPresets } from "./audit-preset-data";
import { LibraryDetailHeading, LibraryTag } from "./browser";
function VersionPicker({ name, version }: { name: string; version: string }) {
  const query = useAuditPresets(false);
  const navigate = useNavigate();
  const location = useLocation();
  const versions = new Set([version]);
  for (const item of query.data ?? []) {
    if (item.ref.name === name) versions.add(item.ref.version);
  }
  return (
    <span className="library-version-picker">
      <label className="library-version">
        <span>Version</span>
        <select
          className="ui-control"
          data-size="xs"
          aria-label="Check type version"
          value={version}
          onChange={(event) =>
            void navigate(
              auditPresetPath(name, event.target.value) + location.search,
              {
                state: location.state,
              },
            )
          }
        >
          {[...versions]
            .sort((left, right) => compareVersions(right, left))
            .map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
        </select>
      </label>
      {query.isPending ? (
        <span className="ui-caption" role="status">
          Loading versions…
        </span>
      ) : null}
      {query.error ? (
        <span className="library-inline-error" role="alert">
          Other versions could not be loaded.{" "}
          <Button
            size="xs"
            pending={query.isFetching}
            onClick={() => void query.refetch()}
          >
            Retry
          </Button>
        </span>
      ) : null}
    </span>
  );
}

function CheckTypeContents({ profile }: { profile: AuditProfile }) {
  const presentation = presentCheckType(profile);
  const limits = profile.limits;
  const http = Object.values(profile.inputs).some((i) =>
    i.media_types.includes(ENGAGEMENT_MEDIA_TYPE),
  );
  const scope = scopeSummary(profile);
  const workflows = Object.entries(profile.program?.workflows ?? {});
  return (
    <article className="library-design-detail library-check-detail">
      <LibraryDetailHeading
        title={presentation.label}
        kind="Check type"
        description={presentation.description}
        selector={`${profile.ref.name}@${profile.ref.version}`}
        identityLabel="check type version"
        icon={presentation.icon}
        tone={http ? "amber" : "blue"}
        version={
          <VersionPicker
            name={profile.ref.name}
            version={profile.ref.version}
          />
        }
        meta={
          <LibraryTag tone={http ? "amber" : "blue"}>
            {http ? "Engagement" : "Published check type"}
          </LibraryTag>
        }
      />
      <div className="library-detail-grid">
        <div className="library-detail-main">
          <section
            className="library-design-section"
            aria-label="What you'll need"
          >
            <header className="library-design-section-header">
              <h3>What you'll need</h3>
              <span>{Object.keys(profile.inputs).length} inputs</span>
            </header>
            <div className="library-materials">
              {orderedInputs(profile).map(([name, input]) => (
                <div className="library-material" key={name}>
                  <span className="library-material-icon">
                    <Icon
                      name={
                        name === "source"
                          ? "source"
                          : name === "engagement"
                            ? "target"
                            : "docs"
                      }
                    />
                  </span>
                  <div>
                    <strong>{inputLabel(name)}</strong>
                    <small>{formatList(input.media_types)}</small>
                  </div>
                  <LibraryTag>
                    {input.required ? "Required" : "Optional"}
                  </LibraryTag>
                </div>
              ))}
              {Object.keys(profile.inputs).length === 0 ? (
                <p className="library-empty-contract">
                  This check type needs no inputs.
                </p>
              ) : null}
            </div>
          </section>
          {workflows.length ? (
            <section className="library-design-section" aria-label="Workflows">
              <header className="library-design-section-header">
                <h3>Workflows</h3>
                <span>Published bindings</span>
              </header>
              <div className="library-check-flow">
                {workflows.map(([role, binding]) => {
                  const [name, version] = binding.ref.split("@");
                  return (
                    <ContextLink
                      className="library-usage-link"
                      key={role}
                      to={`/catalog/workflows/${encodeURIComponent(name!)}/${encodeURIComponent(version!)}`}
                      returnLabel={presentation.label}
                    >
                      <Icon name="workflow" />
                      <span>
                        <strong>{binding.ref}</strong>
                        <small>
                          Role {role} ·{" "}
                          {Object.keys(binding.outputs).join(", ")}
                        </small>
                      </span>
                      <span aria-hidden="true">↗</span>
                    </ContextLink>
                  );
                })}
              </div>
            </section>
          ) : null}
          <section className="library-design-section">
            <header className="library-design-section-header">
              <h3>What to expect</h3>
            </header>
            <p className="library-section-copy">
              {scope?.size ?? "Recorded results and evidence"}
            </p>
            <p className="ui-caption">
              {scope?.detail ??
                "Open the collected results and review possible issues as the check progresses."}
            </p>
          </section>
          <TechnicalDetails
            description="Limits, policy and exact contract"
            className="library-design-technical"
          >
            <Facts
              size="sm"
              items={[
                {
                  term: "Time limit",
                  value: formatTimeLimit(limits.deadline_seconds),
                },
                {
                  term: "Active tests",
                  value:
                    profile.policy.active_operations === "allowed"
                      ? "Use project setting"
                      : "Prohibited",
                },
                { term: "Task limit", value: limits.max_tasks },
                { term: "Attempts per task", value: limits.max_attempts },
                {
                  term: "Digest",
                  value: <code>{compactDigest(profile.ref.digest)}</code>,
                },
                { term: "Runs", value: limits.max_submitted_runs },
                { term: "Batch size", value: limits.batch_size },
                {
                  term: "Evidence",
                  value: formatBytes(limits.max_evidence_bytes),
                },
              ]}
            />
            <dl className="library-exact-contract">
              {Object.entries(profile.inputs).map(([name, input]) => (
                <div key={name}>
                  <dt>
                    <code>{name}</code>
                  </dt>
                  <dd>{input.media_types.join(", ")}</dd>
                </div>
              ))}
            </dl>
            <ButtonLink size="sm" to="/catalog/studio?kind=AuditProfile">
              Open Node Studio
            </ButtonLink>
          </TechnicalDetails>
        </div>
        <aside className="library-context-card library-setup-card">
          <span className="library-context-icon">
            <Icon name="checks" />
          </span>
          <h3>Make it your check</h3>
          <p>Choose a project and the materials you want to check.</p>
          <ButtonLink variant="primary" to={startCheckPath(profile.ref.name)}>
            Set up check <Icon name="arrow-right" />
          </ButtonLink>
          <small>Review the version and inputs before starting.</small>
        </aside>
      </div>
    </article>
  );
}

export function AuditPresetDetailRoute({
  name,
  version,
}: {
  name: string;
  version: string;
}) {
  const api = usePublicAPI();
  const valid =
    CONFIG_NAME_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.auditProfiles.detail(name, version),
    queryFn: ({ signal }) => getAuditProfile(api, name, version, signal),
    enabled: valid,
  });
  return valid ? (
    <QueryView query={query} loading={<p role="status">Loading check type…</p>}>
      {(profile) => (
        <CheckTypeContents key={`${name}@${version}`} profile={profile} />
      )}
    </QueryView>
  ) : (
    <ErrorNotice error={new Error("Check type version is invalid")} />
  );
}
