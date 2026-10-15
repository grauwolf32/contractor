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
  formatBytes,
  formatTimeLimit,
  compareVersions,
} from "../../app/format";
import { QueryView } from "../../app/query-view";
import {
  capitalize,
  checkItemKind,
  itemCount,
  itemNoun,
  TERMS,
  type ItemKind,
} from "../../app/vocabulary";
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
import { AuditPresetStandardChecks } from "./audit-preset-checks";
import {
  auditModeLabels,
  hasFixedItems,
  presetScope,
  startCheckPath,
  useAuditPresets,
  wordOf,
} from "./audit-preset-data";
import {
  CheckTypeAvailability,
  CompatibilityReasons,
} from "./check-type-status";
import { LibraryBackLink, LibrarySearch } from "./library-parts";
import { useCatalogQueryState } from "./query-state";

const BACK = {
  returnTo: "/catalog/audit-presets",
  returnLabel: "All check types",
};

const ACTIVE_TESTS: Readonly<
  Record<AuditProfile["interaction"]["activeChecks"], string>
> = {
  prohibited: "Never run",
  automatic: "Use project setting",
  "approval-required": "Use project setting",
};

const POSSIBLE_ISSUES: Readonly<
  Record<AuditProfile["interaction"]["findingConfirmation"], string>
> = {
  "human-required": "You confirm each one",
  disabled: "Not proposed",
};

const NOT_APPLICABLE: Readonly<
  Record<AuditProfile["interaction"]["notApplicable"], string>
> = {
  "human-required": "You decide what does not apply",
  "profile-rule": "Set by the check type",
};

const REPORT_ACCEPTANCE: Readonly<
  Record<AuditProfile["interaction"]["reportAcceptance"], string>
> = {
  automatic: "Accepted automatically",
  "human-required": "Needs your acceptance",
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

function CheckTypeItems({
  profile,
  kind,
}: {
  profile: AuditProfile;
  kind: ItemKind;
}) {
  const heading = useId();
  const search = useCatalogQueryState();
  const plural = itemNoun(kind, 2);
  const fixed = hasFixedItems(profile);
  const selection = profile.inventory.standardSelection;
  const source = profile.inventory.source;
  return (
    <section className="library-block" aria-labelledby={heading}>
      <header className="library-block-header">
        <div>
          <h3 id={heading} className="library-block-title">
            {capitalize(plural)}
          </h3>
          <p className="library-block-lede">{presetScope(profile)}</p>
          {selection ? (
            <p className="ui-caption">
              {itemCount(kind, selection.entryIds.length)} selected · Levels{" "}
              {selection.levels.join(", ")}
            </p>
          ) : null}
        </div>
        {fixed ? (
          <LibrarySearch
            label={`Search ${plural}`}
            placeholder="ID, title or text"
            value={search.draftSearch}
            onChange={search.changeDraftSearch}
          />
        ) : null}
      </header>
      {fixed ? (
        profile.standards.map((reference) => (
          <AuditPresetStandardChecks
            key={`${reference.scheme}@${reference.version}`}
            reference={reference}
            profile={profile}
            kind={kind}
            search={search.committedSearch.toLowerCase()}
          />
        ))
      ) : (
        <div className="library-note">
          <strong>
            The {itemNoun(kind, 1)} list depends on the inputs of each check.
          </strong>
          <p>
            {source?.source === "prepare-output" ? (
              "Preparation builds it from your project inputs."
            ) : (
              <>
                You supply{" "}
                {source === undefined ? (
                  "the required inputs"
                ) : (
                  <code>{source.name}</code>
                )}{" "}
                when you start a check in a project.
              </>
            )}{" "}
            The {plural} it finds are listed in that check’s Coverage tab.
          </p>
        </div>
      )}
    </section>
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
                {input.mediaTypes
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
  const workflows = Object.entries(profile.workflows ?? {});
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
              to={`/catalog/workflows/${encodeURIComponent(binding.workflow.name)}/${encodeURIComponent(binding.workflow.version)}`}
              returnLabel={`${auditPresetLabel(profile.ref.name)} @${profile.ref.version}`}
            >
              {binding.workflow.name}@{binding.workflow.version}
            </ContextLink>
            <span className="library-row-detail">
              Role <code>{role}</code> · {binding.kind}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function CheckTypeTechnicalDetails({ profile }: { profile: AuditProfile }) {
  const { execution } = profile;
  return (
    <TechnicalDetails description="Limits and identities, for admins and debugging.">
      <Facts
        size="sm"
        items={[
          {
            term: "Digest",
            value: (
              <code title={profile.ref.digest}>
                {compactDigest(profile.ref.digest)}
              </code>
            ),
          },
          {
            term: "Inventory",
            value: (
              <>
                <code>{profile.inventory.implementation}</code> · role{" "}
                <code>{profile.inventory.itemWorkflowRole}</code>
              </>
            ),
          },
          {
            term: "Input validation",
            value: profile.requiresInputValidation
              ? "Required"
              : "Not required",
          },
          {
            term: "Rounds",
            value: `Up to ${execution.maxRounds} · ${execution.roundMode} · incomplete round: ${execution.incompleteRound}`,
          },
          {
            term: "Items",
            value: `${execution.maxItemsTotal} in total · ${execution.maxItemsPerRound} per round · batches of ${execution.batchSize}`,
          },
          {
            term: "Runs",
            value: `Up to ${execution.maxSubmittedRuns} submitted · ${execution.maxItemRunAttempts} attempts per item`,
          },
          {
            term: "Evidence",
            value: `Up to ${formatBytes(execution.maxEvidenceBytes)}`,
          },
        ]}
      />
    </TechnicalDetails>
  );
}

function CheckTypeContents({ profile }: { profile: AuditProfile }) {
  const kind = checkItemKind(profile);
  const { interaction } = profile;
  return (
    <>
      {profile.serverCompatible ? null : (
        <div className="library-note" data-tone="warning">
          <strong>This check type cannot run on this server.</strong>
          <CompatibilityReasons reasons={profile.compatibilityReasons} />
        </div>
      )}
      <Facts
        items={[
          {
            term: "Standards",
            value:
              profile.standards.length === 0
                ? "None"
                : profile.standards
                    .map((standard) => `${standard.scheme}@${standard.version}`)
                    .join(", "),
          },
          {
            term: "Time limit",
            value: formatTimeLimit(profile.execution.deadlineSeconds),
          },
          {
            term: "Active tests",
            value: wordOf(ACTIVE_TESTS, interaction.activeChecks),
          },
          {
            term: "Possible issues",
            value: wordOf(POSSIBLE_ISSUES, interaction.findingConfirmation),
          },
          {
            term: "Applicability",
            value: wordOf(NOT_APPLICABLE, interaction.notApplicable),
          },
          {
            term: "Report",
            value: wordOf(REPORT_ACCEPTANCE, interaction.reportAcceptance),
          },
        ]}
      />
      <CheckTypeItems profile={profile} kind={kind} />
      <CheckTypeInputs profile={profile} />
      <CheckTypeWorkflows profile={profile} />
      <CheckTypeTechnicalDetails profile={profile} />
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
                  <span>
                    {capitalize(TERMS.checkType)}
                    {profile === undefined
                      ? null
                      : ` · ${auditModeLabels[profile.mode]}`}
                  </span>
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
                  {profile?.serverCompatible ? (
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
