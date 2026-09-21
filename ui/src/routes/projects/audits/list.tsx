import { compareWorkflowVersions } from "../../workflows/families";
import { ContextLink } from "../../../app/context-navigation";
import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { type FormEvent, useEffect, useId, useMemo, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router";

import type { ArtifactMetadata } from "../../../api/artifacts";
import { auditPresetPath } from "../../../api/audit-presets";
import {
  auditNeedsPolling,
  createAudit,
  getAuditProfile,
  listAuditProfiles,
  listProjectAudits,
  type AuditProfile,
  type CreateAuditRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { listProjectArtifacts } from "../../../api/project-artifacts";
import { queryKeys } from "../../../api/query-keys";
import { Dialog } from "../../../app/dialog";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../../artifacts/common";
import { AuditControls } from "./controls";
import { auditProfileLabel } from "./labels";
import { describeStopReason } from "./stop-reason";
import { StateBadge } from "../../runs/components";

import "./styles.css";
import { RefreshButton } from "../../../app/refresh-button";
import { ProjectSectionActions } from "../navigation";

const INITIAL_CURSOR = null;

function nextCursor(page: { page: { hasMore: boolean; nextCursor?: string } }) {
  return page.page.hasMore ? (page.page.nextCursor ?? undefined) : undefined;
}

function profileOption(profile: AuditProfile): string {
  return JSON.stringify([profile.ref.name, profile.ref.version]);
}

function compactDigest(digest: string): string {
  return digest.length <= 28
    ? digest
    : `${digest.slice(0, 15)}…${digest.slice(-8)}`;
}

function compatibleMediaType(
  mediaType: string,
  accepted: readonly string[],
): boolean {
  return accepted.some((candidate) => {
    if (candidate === "*/*" || candidate === mediaType) return true;
    if (!candidate.endsWith("/*")) return false;
    return mediaType.startsWith(candidate.slice(0, -1));
  });
}

function artifactOption(artifact: ArtifactMetadata): string {
  return JSON.stringify(artifact.artifact);
}

function selectedArtifact(
  value: string | undefined,
  artifacts: readonly ArtifactMetadata[],
) {
  return artifacts.find((artifact) => artifactOption(artifact) === value)
    ?.artifact;
}

function AuditCreateForm({
  projectId,
  onClose,
}: {
  projectId: string;
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const heading = useId();
  const [profileSelection, setProfileSelection] = useState("");
  const [artifactSelections, setArtifactSelections] = useState<
    Record<string, string>
  >({});
  const [validationError, setValidationError] = useState<string | null>(null);
  const [keyring] = useState(
    () => new MutationDraftKeyring<CreateAuditRequest>("create-audit"),
  );
  const profiles = useInfiniteQuery({
    queryKey: queryKeys.auditProfiles.all,
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listAuditProfiles(api, pageParam === null ? {} : { cursor: pageParam }),
    getNextPageParam: nextCursor,
  });
  const artifacts = useInfiniteQuery({
    queryKey: queryKeys.projects.artifacts.picker(projectId),
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listProjectArtifacts(api, {
        projectId,
        ...(pageParam === null ? {} : { cursor: pageParam }),
      }),
    getNextPageParam: nextCursor,
  });
  const {
    fetchNextPage: fetchNextArtifactPage,
    hasNextPage: hasNextArtifactPage,
    isFetching: isFetchingArtifacts,
    isError: artifactLoadFailed,
  } = artifacts;
  useEffect(() => {
    if (hasNextArtifactPage && !isFetchingArtifacts && !artifactLoadFailed) {
      void fetchNextArtifactPage();
    }
  }, [
    artifactLoadFailed,
    fetchNextArtifactPage,
    hasNextArtifactPage,
    isFetchingArtifacts,
  ]);
  const profileItems = useMemo(() => {
    const families = new Map<string, AuditProfile[]>();
    for (const profile of profiles.data?.pages.flatMap((page) => page.items) ??
      []) {
      const family = families.get(profile.ref.name) ?? [];
      family.push(profile);
      families.set(profile.ref.name, family);
    }
    return [...families.values()].flatMap((family) =>
      family.sort((a, b) =>
        compareWorkflowVersions(b.ref.version, a.ref.version),
      ),
    );
  }, [profiles.data]);
  const artifactItems = useMemo(
    () => artifacts.data?.pages.flatMap((page) => page.items) ?? [],
    [artifacts.data],
  );
  const selectedSummary =
    profileItems.find(
      (profile) => profileOption(profile) === profileSelection,
    ) ??
    profileItems.find((profile) => profile.serverCompatible) ??
    profileItems[0];
  const effectiveProfileSelection =
    selectedSummary === undefined ? "" : profileOption(selectedSummary);
  const profile = useQuery({
    queryKey:
      selectedSummary === undefined
        ? ["audit-profiles", "detail", "none"]
        : queryKeys.auditProfiles.detail(
            selectedSummary.ref.name,
            selectedSummary.ref.version,
          ),
    queryFn: () =>
      getAuditProfile(
        api,
        selectedSummary!.ref.name,
        selectedSummary!.ref.version,
      ),
    enabled: selectedSummary !== undefined,
  });
  const create = useMutation({
    mutationFn: (request: CreateAuditRequest) =>
      createAudit(api, {
        projectId,
        request,
        idempotencyKey: keyring.keyFor(request),
      }),
    onSuccess: async (audit) => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(projectId),
      });
      void navigate(
        `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(audit.auditId)}`,
      );
    },
  });
  const exactProfile = profile.data;
  const inputEntries = Object.entries(exactProfile?.inputs ?? {}).sort(
    ([left], [right]) => left.localeCompare(right),
  );
  const artifactInventoryComplete =
    artifacts.isSuccess && !artifacts.hasNextPage;
  const inputs = inputEntries.map(([name, contract]) => {
    const compatible = artifactItems.filter((artifact) =>
      compatibleMediaType(artifact.mediaType, contract.mediaTypes),
    );
    const selection =
      artifactSelections[name] ??
      (artifactInventoryComplete && compatible.length === 1
        ? artifactOption(compatible[0]!)
        : "");
    return { name, contract, compatible, selection };
  });
  const missingRequired = inputs.filter(
    ({ contract, compatible, selection }) =>
      contract.required &&
      selectedArtifact(selection, compatible) === undefined,
  );

  function selectProfile(value: string): void {
    setProfileSelection(value);
    setArtifactSelections({});
    setValidationError(null);
    create.reset();
  }

  function close(): void {
    if (!create.isPending) onClose();
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    create.reset();
    if (exactProfile === undefined || !exactProfile.serverCompatible) {
      setValidationError("Select a profile version supported by the Server.");
      return;
    }
    const selectedInputs = Object.fromEntries(
      inputs.flatMap(({ name, compatible, selection }) => {
        const artifact = selectedArtifact(selection, compatible);
        return artifact === undefined ? [] : [[name, { ...artifact }]];
      }),
    );
    if (missingRequired.length > 0) {
      setValidationError(
        `Select Project Artifacts for: ${missingRequired.map(({ name }) => name).join(", ")}.`,
      );
      return;
    }
    const data = new FormData(event.currentTarget);
    const labels = String(data.get("runtimeLabels") ?? "")
      .split(/[\s,]+/u)
      .map((value) => value.trim())
      .filter(
        (value, index, values) =>
          value !== "" && values.indexOf(value) === index,
      );
    const scope = Object.fromEntries(
      ["objective", "target", "authorizationScope"].flatMap((name) => {
        const value = String(data.get(name) ?? "").trim();
        return value === "" ? [] : [[name, value]];
      }),
    );
    create.mutate({
      profile: {
        name: exactProfile.ref.name,
        version: exactProfile.ref.version,
      },
      inputs: selectedInputs,
      ...(labels.length === 0 ? {} : { runtimeLabels: labels }),
      ...(Object.keys(scope).length === 0 ? {} : { scope }),
    });
  }

  return (
    <Dialog
      className="project-dialog panel audit-create-dialog"
      labelledBy={heading}
      onRequestClose={close}
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Pinned program and inputs</p>
          <h2 id={heading}>New Audit</h2>
        </div>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close New Audit form"
          disabled={create.isPending}
          onClick={close}
        >
          ×
        </button>
      </div>
      <form className="audit-create-form" onSubmit={submit}>
        {profiles.isPending ? (
          <p className="loading-copy">Loading Audit profiles…</p>
        ) : profiles.error !== null ? (
          <ErrorNotice error={profiles.error} />
        ) : profileItems.length === 0 ? (
          <div className="compact-empty">
            <strong>No Audit profiles are published.</strong>
          </div>
        ) : (
          <label>
            Audit profile
            <select
              value={effectiveProfileSelection}
              onChange={(event) => selectProfile(event.currentTarget.value)}
            >
              {profileItems.map((candidate) => (
                <option
                  key={profileOption(candidate)}
                  value={profileOption(candidate)}
                >
                  {candidate.ref.name}@{candidate.ref.version}
                  {candidate.serverCompatible
                    ? profileItems.find(
                        (p) =>
                          p.ref.name === candidate.ref.name &&
                          p.serverCompatible,
                      ) === candidate
                      ? " · latest loaded version"
                      : ""
                    : " — unsupported"}
                </option>
              ))}
            </select>
          </label>
        )}
        {profiles.hasNextPage ? (
          <button
            className="secondary-button"
            type="button"
            disabled={profiles.isFetchingNextPage}
            onClick={() => void profiles.fetchNextPage()}
          >
            {profiles.isFetchingNextPage ? "Loading…" : "Load more profiles"}
          </button>
        ) : null}
        {profile.isPending && selectedSummary !== undefined ? (
          <p className="loading-copy">Loading profile contract…</p>
        ) : profile.error !== null ? (
          <ErrorNotice error={profile.error} />
        ) : exactProfile === undefined ? null : (
          <div className="audit-profile-contract">
            <ContextLink
              to={auditPresetPath(
                exactProfile.ref.name,
                exactProfile.ref.version,
              )}
              returnLabel="Project audits"
            >
              Browse preset checks →
            </ContextLink>
            {exactProfile.serverCompatible ? (
              <p className="notice notice-success" role="status">
                This profile version is supported by the Server.
                {exactProfile.requiresInputValidation
                  ? " Inputs are validated when the Audit starts."
                  : ""}
              </p>
            ) : (
              <div className="notice notice-error" role="alert">
                <strong>This profile cannot run on this Server.</strong>
                <ul>
                  {exactProfile.compatibilityReasons.map((reason) => (
                    <li key={reason}>{reason.replaceAll("_", " ")}</li>
                  ))}
                </ul>
              </div>
            )}
            <fieldset className="audit-input-fields">
              <legend>Project inputs</legend>
              {artifacts.isPending ||
              (artifacts.hasNextPage && !artifacts.isError) ? (
                <p className="loading-copy" role="status">
                  Loading Project Artifacts…
                </p>
              ) : null}
              {artifacts.error === null ? null : (
                <>
                  <ErrorNotice error={artifacts.error} />
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={artifacts.isFetching}
                    onClick={() =>
                      void (artifacts.hasNextPage
                        ? artifacts.fetchNextPage()
                        : artifacts.refetch())
                    }
                  >
                    Retry loading Project Artifacts
                  </button>
                </>
              )}
              {inputs.map(({ name, contract, compatible, selection }) => {
                return (
                  <label key={name}>
                    <span>
                      {name} {contract.required ? "(required)" : "(optional)"}
                    </span>
                    <small>{contract.mediaTypes.join(", ")}</small>
                    <select
                      aria-label={`Input ${name}`}
                      value={selection}
                      required={contract.required}
                      onChange={(event) => {
                        const value = event.currentTarget.value;
                        setArtifactSelections((current) => ({
                          ...current,
                          [name]: value,
                        }));
                      }}
                    >
                      <option value="">
                        {compatible.length === 0
                          ? "No compatible current Artifact"
                          : "Select a revision"}
                      </option>
                      {compatible.map((artifact) => (
                        <option
                          key={artifactOption(artifact)}
                          value={artifactOption(artifact)}
                        >
                          {artifact.artifact.namespace}/{artifact.artifact.name}
                          @{artifact.artifact.revision} · {artifact.mediaType}
                        </option>
                      ))}
                    </select>
                  </label>
                );
              })}
            </fieldset>
            <details className="optional-settings">
              <summary>Optional settings</summary>
              <div>
                <div className="audit-profile-summary">
                  <div>
                    <span>Mode</span>
                    <strong>{exactProfile.mode}</strong>
                  </div>
                  <div>
                    <span>Digest</span>
                    <code title={exactProfile.ref.digest}>
                      {compactDigest(exactProfile.ref.digest)}
                    </code>
                  </div>
                  <div>
                    <span>Limit</span>
                    <strong>
                      {exactProfile.execution.maxItemsTotal} items
                    </strong>
                  </div>
                  <div>
                    <span>Attempts</span>
                    <strong>{exactProfile.execution.maxItemRunAttempts}</strong>
                  </div>
                </div>
                {exactProfile.standards.length === 0 ? null : (
                  <p className="notice" data-testid="audit-profile-standards">
                    Standards pinned at start:{" "}
                    {exactProfile.standards
                      .map(
                        (standard) => `${standard.scheme}@${standard.version}`,
                      )
                      .join(", ")}
                  </p>
                )}
                {exactProfile.inventory.standardSelection ===
                undefined ? null : (
                  <div
                    className="notice"
                    data-testid="audit-profile-standard-selection"
                  >
                    <strong>
                      {exactProfile.inventory.standardSelection.scope}
                    </strong>
                    <p>
                      Levels{" "}
                      {exactProfile.inventory.standardSelection.levels.join(
                        ", ",
                      )}{" "}
                      ·{" "}
                      {exactProfile.inventory.standardSelection.entryIds.length}{" "}
                      requirements
                    </p>
                  </div>
                )}
                <div className="form-grid audit-scope-fields">
                  <label>
                    Objective
                    <input name="objective" maxLength={4096} />
                  </label>
                  <label>
                    Target
                    <input name="target" maxLength={4096} />
                  </label>
                  <label>
                    Authorization scope
                    <input name="authorizationScope" maxLength={4096} />
                  </label>
                  <label>
                    Runtime labels
                    <input
                      name="runtimeLabels"
                      placeholder="debug, caido"
                      autoComplete="off"
                    />
                  </label>
                </div>
              </div>
            </details>
          </div>
        )}
        {validationError === null ? null : (
          <p className="form-error" role="alert">
            {validationError}
          </p>
        )}
        {create.error === null ? null : (
          <ErrorNotice error={create.error} reconcileWrite />
        )}
        <div className="project-dialog-actions">
          <button
            type="button"
            className="secondary-button"
            disabled={create.isPending}
            onClick={close}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={
              create.isPending ||
              exactProfile === undefined ||
              !exactProfile.serverCompatible ||
              missingRequired.length > 0
            }
          >
            {create.isPending ? "Creating draft…" : "Create Audit draft"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

export function ProjectAuditWorkspace({
  projectId,
  projectName,
}: {
  projectId: string;
  projectName?: string;
}) {
  const api = usePublicAPI();
  const [showCreate, setShowCreate] = useState(false);
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const cursors = params.getAll("auditCursor");
  function setCursors(values: string[]) {
    const next = new URLSearchParams(params);
    next.delete("auditCursor");
    for (const value of values) next.append("auditCursor", value);
    setParams(next, { state: location.state });
  }
  const cursor = cursors.at(-1);
  const audits = useQuery({
    queryKey: queryKeys.projects.audits.list(projectId, cursor),
    queryFn: () =>
      listProjectAudits(api, {
        projectId,
        ...(cursor === undefined ? {} : { cursor }),
      }),
    refetchOnReconnect: true,
    refetchInterval: (query) =>
      query.state.data?.items.some((audit) => auditNeedsPolling(audit.state))
        ? 5_000
        : false,
  });
  return (
    <div className="audit-page audit-collection">
      <ProjectSectionActions>
        <RefreshButton
          isFetching={audits.isFetching}
          onRefresh={() => void audits.refetch()}
        />
        <button type="button" onClick={() => setShowCreate(true)}>
          New Audit
        </button>
      </ProjectSectionActions>
      {showCreate ? (
        <AuditCreateForm
          projectId={projectId}
          onClose={() => setShowCreate(false)}
        />
      ) : null}
      {audits.isPending ? (
        <p className="loading-copy">Loading Audits…</p>
      ) : audits.error !== null ? (
        <ErrorNotice error={audits.error} />
      ) : audits.data.items.length === 0 ? (
        <div className="empty-state panel">
          <h3>No Audits yet</h3>
          <p>
            Create an audit to run a set of checks against your project sources.
          </p>
        </div>
      ) : (
        <div className="audit-card-grid">
          {audits.data.items.map((audit) => {
            const root = `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(audit.auditId)}`;
            const stop = describeStopReason(audit);
            const inputNames = [
              ...new Set(
                Object.values(audit.inputs).map((input) => input.ref.name),
              ),
            ];
            return (
              <article
                className="panel audit-card"
                key={audit.auditId}
                aria-label={`${auditProfileLabel(audit)} ${audit.auditId}`}
              >
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">
                      {formatTimestamp(audit.createdAt)}
                    </p>
                    <h3>
                      <ContextLink
                        returnLabel="Project Audits"
                        to={audit.state === "draft" ? root : `${root}/coverage`}
                      >
                        {auditProfileLabel(audit)}
                      </ContextLink>
                    </h3>
                    <p className="audit-card-inputs">
                      {inputNames.length === 0 ? (
                        <span>No inputs selected</span>
                      ) : (
                        inputNames.map((name) => <code key={name}>{name}</code>)
                      )}
                      <span
                        className="audit-card-short-id"
                        title={audit.auditId}
                      >
                        #{audit.auditId.slice(-8)}
                      </span>
                    </p>
                  </div>
                  <StateBadge state={audit.state} />
                </div>
                {audit.scope.objective ? (
                  <p className="audit-card-objective">
                    {audit.scope.objective}
                  </p>
                ) : null}
                <dl className="metadata-grid audit-card-stats">
                  <div>
                    <dt>Runs submitted</dt>
                    <dd>{audit.submittedRunCount}</dd>
                  </div>
                  <div>
                    <dt>Runs in progress</dt>
                    <dd>{audit.outstandingRunCount}</dd>
                  </div>
                  <div>
                    <dt>Updated</dt>
                    <dd>{formatTimestamp(audit.updatedAt)}</dd>
                  </div>
                </dl>
                {stop === null ? null : (
                  <p
                    className={
                      stop.tone === "error" && !stop.deadline
                        ? "form-error"
                        : "muted-copy audit-card-stop"
                    }
                  >
                    {stop.deadline
                      ? audit.state === "paused"
                        ? "Time limit reached. Continue with a longer limit or no time limit."
                        : "Time limit reached; no further Runs were submitted."
                      : stop.label === undefined
                        ? stop.message
                        : `${stop.label}. ${stop.message}`}
                  </p>
                )}
                <div className="audit-card-footer">
                  <div className="button-row">
                    <ContextLink
                      returnLabel="Project Audits"
                      className="audit-open-link"
                      to={audit.state === "draft" ? root : `${root}/coverage`}
                    >
                      {audit.state === "draft"
                        ? "Open draft →"
                        : "View checks & results →"}
                    </ContextLink>
                    {audit.state === "waiting_review" ? (
                      <ContextLink
                        returnLabel="Project Audits"
                        to={`${root}/reviews`}
                      >
                        Review decisions
                      </ContextLink>
                    ) : null}
                  </div>
                  <AuditControls
                    audit={audit}
                    projectName={projectName}
                    compact
                  />
                </div>
                <details className="audit-record-details">
                  <summary>Audit identity</summary>
                  <p>
                    <code>{audit.auditId}</code> · {audit.profile.name}@
                    {audit.profile.version}
                  </p>
                </details>
              </article>
            );
          })}
        </div>
      )}
      <CursorControls
        label="Project Audit pages"
        canGoBack={cursors.length > 0}
        {...(audits.data?.page.hasMore === true &&
        audits.data.page.nextCursor !== undefined
          ? { nextCursor: audits.data.page.nextCursor }
          : {})}
        onBack={() => setCursors(cursors.slice(0, -1))}
        onNext={(next) => setCursors([...cursors, next])}
      />
    </div>
  );
}
