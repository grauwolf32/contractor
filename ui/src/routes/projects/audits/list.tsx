import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { type FormEvent, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";

import type { ArtifactMetadata } from "../../../api/artifacts";
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
import { getProject, PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../../artifacts/common";
import { AuditControls } from "./controls";
import { auditProfileLabel } from "./labels";
import { StateBadge } from "../../runs/components";
import { ProjectAuditNavigation } from "./shared";

import "./styles.css";

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

function AuditCreateForm({ projectId }: { projectId: string }) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
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
  const profileItems = useMemo(
    () => profiles.data?.pages.flatMap((page) => page.items) ?? [],
    [profiles.data],
  );
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
  const missingRequired = inputEntries.filter(
    ([name, contract]) =>
      contract.required &&
      selectedArtifact(artifactSelections[name], artifactItems) === undefined,
  );

  function selectProfile(value: string): void {
    setProfileSelection(value);
    setArtifactSelections({});
    setValidationError(null);
    create.reset();
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    create.reset();
    if (exactProfile === undefined || !exactProfile.serverCompatible) {
      setValidationError("Select a Server-compatible exact profile version.");
      return;
    }
    const selectedInputs = Object.fromEntries(
      inputEntries.flatMap(([name]) => {
        const artifact = selectedArtifact(
          artifactSelections[name],
          artifactItems,
        );
        return artifact === undefined ? [] : [[name, { ...artifact }]];
      }),
    );
    if (missingRequired.length > 0) {
      setValidationError(
        `Select exact Project Artifacts for: ${missingRequired.map(([name]) => name).join(", ")}.`,
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
    <form className="panel audit-create-form" onSubmit={submit}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Pinned program and inputs</p>
          <h3>New Audit</h3>
        </div>
        <span className="audit-policy-mark">New draft</span>
      </div>
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
          Exact Audit profile
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
                {candidate.serverCompatible ? "" : " — unsupported"}
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
        <p className="loading-copy">Loading exact profile contract…</p>
      ) : profile.error !== null ? (
        <ErrorNotice error={profile.error} />
      ) : exactProfile === undefined ? null : (
        <div className="audit-profile-contract">
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
              <strong>{exactProfile.execution.maxItemsTotal} items</strong>
            </div>
            <div>
              <span>Attempts</span>
              <strong>{exactProfile.execution.maxItemRunAttempts}</strong>
            </div>
          </div>
          {exactProfile.standards.length === 0 ? null : (
            <p className="notice" data-testid="audit-profile-standards">
              Exact standards pinned at start:{" "}
              {exactProfile.standards
                .map((standard) => `${standard.scheme}@${standard.version}`)
                .join(", ")}
            </p>
          )}
          {exactProfile.inventory.standardSelection === undefined ? null : (
            <div
              className="notice"
              data-testid="audit-profile-standard-selection"
            >
              <strong>{exactProfile.inventory.standardSelection.scope}</strong>
              <p>
                Levels{" "}
                {exactProfile.inventory.standardSelection.levels.join(", ")} ·{" "}
                {exactProfile.inventory.standardSelection.entryIds.length} exact
                requirements
              </p>
            </div>
          )}
          {exactProfile.serverCompatible ? (
            <p className="notice notice-success" role="status">
              This exact profile version is supported by the Server.
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
            <legend>Exact Project inputs</legend>
            {inputEntries.map(([name, contract]) => {
              const compatible = artifactItems.filter((artifact) =>
                compatibleMediaType(artifact.mediaType, contract.mediaTypes),
              );
              return (
                <label key={name}>
                  <span>
                    {name} {contract.required ? "(required)" : "(optional)"}
                  </span>
                  <small>{contract.mediaTypes.join(", ")}</small>
                  <select
                    aria-label={`Input ${name}`}
                    value={artifactSelections[name] ?? ""}
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
                        : "Select an exact revision"}
                    </option>
                    {compatible.map((artifact) => (
                      <option
                        key={artifactOption(artifact)}
                        value={artifactOption(artifact)}
                      >
                        {artifact.artifact.namespace}/{artifact.artifact.name}@
                        {artifact.artifact.revision} · {artifact.mediaType}
                      </option>
                    ))}
                  </select>
                </label>
              );
            })}
          </fieldset>
          {artifacts.hasNextPage ? (
            <button
              className="secondary-button"
              type="button"
              disabled={artifacts.isFetchingNextPage}
              onClick={() => void artifacts.fetchNextPage()}
            >
              {artifacts.isFetchingNextPage
                ? "Loading…"
                : "Load more Project Artifacts"}
            </button>
          ) : null}
          <details>
            <summary>Optional scope and Runtime labels</summary>
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
    </form>
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
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
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
      <div className="section-heading">
        <div>
          <p className="muted-copy">
            Review tasks, evidence and results from your security audits.
          </p>
        </div>
        <div className="button-row">
          <button
            className="secondary-button"
            disabled={audits.isFetching}
            onClick={() => void audits.refetch()}
          >
            {audits.isFetching ? "Refreshing…" : "Refresh audits"}
          </button>
          <button
            type="button"
            aria-expanded={showCreate}
            aria-controls="audit-create-form"
            onClick={() => setShowCreate((value) => !value)}
          >
            {showCreate ? "Close new audit" : "New Audit"}
          </button>
        </div>
      </div>
      {showCreate ? (
        <div id="audit-create-form">
          <AuditCreateForm projectId={projectId} />
        </div>
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
                      <Link
                        to={audit.state === "draft" ? root : `${root}/coverage`}
                      >
                        {auditProfileLabel(audit)}
                      </Link>
                    </h3>
                  </div>
                  <StateBadge state={audit.state} />
                </div>
                {audit.scope.objective ? (
                  <p className="audit-card-objective">
                    {audit.scope.objective}
                  </p>
                ) : null}
                <p className="audit-card-inputs">
                  {Object.values(audit.inputs)
                    .map((input) => input.ref.name)
                    .join(" · ") || "No inputs selected"}
                </p>
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
                {audit.stopReason ? (
                  <p className="form-error">{audit.stopReason.message}</p>
                ) : null}
                <div className="audit-card-footer">
                  <div className="button-row">
                    <Link
                      className="audit-open-link"
                      to={audit.state === "draft" ? root : `${root}/coverage`}
                    >
                      {audit.state === "draft"
                        ? "Open draft →"
                        : "View checks & results →"}
                    </Link>
                    {audit.state === "waiting_review" ? (
                      <Link to={`${root}/reviews`}>Review decisions</Link>
                    ) : null}
                  </div>
                  <AuditControls
                    audit={audit}
                    projectName={projectName}
                    deleteOnly
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
        canGoBack={cursors.length > 1}
        {...(audits.data?.page.hasMore === true &&
        audits.data.page.nextCursor !== undefined
          ? { nextCursor: audits.data.page.nextCursor }
          : {})}
        onBack={() =>
          setCursors((current) =>
            current.slice(0, Math.max(1, current.length - 1)),
          )
        }
        onNext={(next) => setCursors((current) => [...current, next])}
      />
    </div>
  );
}

export function ProjectAuditListRoute() {
  const api = usePublicAPI();
  const { projectId = "" } = useParams();
  const validProject = PROJECT_ID_PATTERN.test(projectId);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validProject,
  });
  if (!validProject)
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Project Audit route is invalid")} />
        <Link to="/projects">Return to Projects</Link>
      </section>
    );
  return (
    <section className="route-page audit-page">
      <header className="route-header-row">
        <div>
          <Link
            className="back-link"
            to={`/projects/${encodeURIComponent(projectId)}#project-audits`}
          >
            ← {project.data?.name ?? "Project"}
          </Link>
          <p className="eyebrow">{project.data?.name ?? "Project"}</p>
          <h2>Audits</h2>
        </div>
      </header>
      {project.error === null ? null : <ErrorNotice error={project.error} />}
      <ProjectAuditNavigation projectId={projectId} current="audits" />
      <ProjectAuditWorkspace
        key={projectId}
        projectId={projectId}
        {...(project.data ? { projectName: project.data.name } : {})}
      />
    </section>
  );
}
