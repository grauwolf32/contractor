import { useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { useLocation, useNavigate, useParams } from "react-router";

import { auditPresetPath } from "../../api/audit-presets";
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
import { QueryView } from "../../app/query-view";
import { capitalize, TERMS } from "../../app/vocabulary";
import {
  Button,
  ButtonLink,
  DetailHeader,
  DetailPane,
  Facts,
  IdChip,
  TechnicalDetails,
} from "../../ui";
import { auditPresetLabel } from "../projects/audits/labels";
import { workflowFormats } from "../workflows/formats";
import {
  presetScope,
  startCheckPath,
  useAuditPresets,
} from "./audit-preset-data";
import { CheckTypeAvailability } from "./check-type-status";
import { LibraryBackLink } from "./library-parts";

const BACK = {
  returnTo: "/catalog/audit-presets",
  returnLabel: "All check types",
};

function VersionPicker({ name, version }: { name: string; version: string }) {
  const query = useAuditPresets();
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
            void navigate(auditPresetPath(name, event.target.value), {
              state: location.state,
            })
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

function CheckTypeInputs({ profile }: { profile: AuditProfile }) {
  const heading = useId();
  const inputs = Object.entries(profile.inputs).sort(
    ([left, x], [right, y]) =>
      Number(y.required) - Number(x.required) || left.localeCompare(right),
  );
  return (
    <section className="library-block" aria-labelledby={heading}>
      <h3 id={heading} className="library-block-title">
        Inputs
      </h3>
      {inputs.length === 0 ? (
        <p className="ui-caption">This check type needs no inputs.</p>
      ) : (
        <ul className="library-rows">
          {inputs.map(([name, input]) => (
            <li key={name}>
              <code className="library-row-name">{name}</code>
              <span className="library-row-tag">
                {input.required ? "Required" : "Optional"}
              </span>
              <span className="library-row-detail">
                {input.media_types
                  .map((type) =>
                    workflowFormats[type]
                      ? `${workflowFormats[type]} (${type})`
                      : type,
                  )
                  .join(", ")}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function CheckTypeWorkflows({ profile }: { profile: AuditProfile }) {
  const heading = useId();
  const workflows = Object.entries(profile.program?.workflows ?? {});
  if (workflows.length === 0) return null;
  return (
    <section className="library-block" aria-labelledby={heading}>
      <h3 id={heading} className="library-block-title">
        Workflows
      </h3>
      <ul className="library-rows">
        {workflows.map(([role, binding]) => (
          <li key={role}>
            <ContextLink
              className="library-row-name"
              to={`/catalog/workflows/${encodeURIComponent(binding.ref.split("@")[0]!)}/${encodeURIComponent(binding.ref.split("@")[1]!)}`}
              returnLabel={`${auditPresetLabel(profile.ref.name)} @${profile.ref.version}`}
            >
              {binding.ref.split("@")[0]!}@{binding.ref.split("@")[1]!}
            </ContextLink>
            <span className="library-row-detail">
              Role <code>{role}</code> ·{" "}
              {Object.keys(binding.outputs).join(", ")}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function CheckTypeContents({ profile }: { profile: AuditProfile }) {
  const limits = profile.limits;
  return (
    <>
      <Facts
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
        ]}
      />
      <p>{presetScope(profile)}</p>
      <CheckTypeInputs profile={profile} />
      <CheckTypeWorkflows profile={profile} />
      <TechnicalDetails description="Limits and identities, for admins and debugging.">
        <Facts
          size="sm"
          items={[
            {
              term: "Digest",
              value: <code>{compactDigest(profile.ref.digest)}</code>,
            },
            { term: "Runs", value: limits.max_submitted_runs },
            { term: "Batch size", value: limits.batch_size },
            { term: "Evidence", value: formatBytes(limits.max_evidence_bytes) },
          ]}
        />
      </TechnicalDetails>
    </>
  );
}

/** Library → Check types → one check type at an exact version. */
export function AuditPresetDetailRoute() {
  const { name = "", version = "" } = useParams();
  const api = usePublicAPI();
  const titleId = useId();
  const valid =
    CONFIG_NAME_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.auditProfiles.detail(name, version),
    queryFn: ({ signal }) => getAuditProfile(api, name, version, signal),
    enabled: valid,
  });
  const profile = query.data;
  const selector = `${name}@${version}`;
  return (
    <div className="ui-stack">
      <LibraryBackLink fallback={BACK} />
      <article className="library-sheet" aria-labelledby={titleId}>
        <DetailPane
          header={
            <DetailHeader
              title={<span id={titleId}>{auditPresetLabel(name)}</span>}
              status={
                profile === undefined ? undefined : (
                  <CheckTypeAvailability profile={profile} />
                )
              }
              meta={
                <>
                  <span>{capitalize(TERMS.checkType)}</span>
                  <IdChip
                    value={selector}
                    display={selector}
                    label="check type version"
                  />
                  {valid ? (
                    <VersionPicker name={name} version={version} />
                  ) : null}
                </>
              }
              actions={
                <>
                  <ButtonLink to="/catalog/studio?kind=AuditProfile">
                    Open Node Studio
                  </ButtonLink>
                  {profile !== undefined ? (
                    <ButtonLink variant="primary" to={startCheckPath(name)}>
                      Start a check with this type
                    </ButtonLink>
                  ) : null}
                </>
              }
            />
          }
        >
          {!valid ? (
            <ErrorNotice error={new Error("Check type version is invalid")} />
          ) : (
            <QueryView
              query={query}
              loading={
                <p className="ui-caption" role="status">
                  Loading check type…
                </p>
              }
            >
              {(data) => <CheckTypeContents key={selector} profile={data} />}
            </QueryView>
          )}
        </DetailPane>
      </article>
    </div>
  );
}
