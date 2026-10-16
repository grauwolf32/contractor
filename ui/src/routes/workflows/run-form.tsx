import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useState } from "react";
import { Link } from "react-router";

import type { ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { queryKeys } from "../../api/query-keys";
import {
  createProjectRun,
  createRun,
  type CreateRunRequest,
  type WorkflowResource,
} from "../../api/workflows";
import { ErrorNotice } from "../../app/error-notice";
import { useGuardedNavigate } from "../../app/page-location";
import { useRunDraftStore } from "../../run-drafts/context";
import {
  initialRunDraftState,
  type RunDraftEntry,
  type RunDraftIdentity,
  type RunDraftMemoryStore,
  type RunDraftState,
  type RunDraftSummary,
} from "../../run-drafts/memory";
import { validateRunDraft } from "../../run-drafts/validation";
import { AlertDialog, Button, StatusChip } from "../../ui";
import { InputFields, ParameterFields } from "./run-form-fields";
import { useRunFormInventories } from "./run-form-inventories";
import { runDraftReadiness } from "./run-form-readiness";
import {
  ExecutionOverrides,
  RunMetadata,
  RuntimePlacement,
} from "./run-form-settings";
import { useRunDraftState } from "./run-form-state";
import "./run-drafts.css";

interface WorkflowRunFormProps {
  presentation?: "page" | "drawer";
  onSubmittingChange?: (pending: boolean) => void;
  workflow: WorkflowResource;
  projectId?: string;
  initialArtifactSelections?: Readonly<Record<string, string>>;
  initialArtifacts?: readonly ArtifactMetadata[];
}

function RepeatRunReview({
  repeat,
  error,
  onReviewed,
}: {
  repeat: NonNullable<RunDraftState["repeat"]>;
  error: string | undefined;
  onReviewed: (reviewed: boolean) => void;
}) {
  return (
    <section
      className={`run-repeat-review${repeat.reviewed ? " is-reviewed" : " needs-review"}`}
      aria-labelledby="run-repeat-review-title"
    >
      <div className="run-repeat-review-heading">
        <div>
          <p className="eyebrow">Configure another Run</p>
          <h4 id="run-repeat-review-title">Review retained request values</h4>
        </div>
        <Link to={`/runs/${encodeURIComponent(repeat.sourceRunId)}`}>
          Source Run
        </Link>
      </div>
      <p>
        This draft creates a new Run after you submit it. It does not retry an
        attempt, mutate the source Run or promise identical placement.
      </p>
      {repeat.notices.length === 0 ? (
        <p className="compact-empty">
          Source revisions and settings were retained.
        </p>
      ) : (
        <ul className="run-repeat-notices">
          {repeat.notices.map((notice, index) => (
            <li
              className="run-repeat-notice"
              data-severity={notice.severity}
              key={`${notice.code}:${notice.field ?? "none"}:${index}`}
            >
              <span>{notice.severity}</span>
              <div>
                <code>{notice.code}</code>
                <p>{notice.message}</p>
                {notice.field === undefined ? null : (
                  <small>Retained field: {notice.field}</small>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      <label className="run-repeat-confirmation">
        <input
          type="checkbox"
          checked={repeat.reviewed}
          onChange={(event) => onReviewed(event.target.checked)}
        />
        I reviewed the retained inputs, labels and execution settings for this
        new Run.
      </label>
      {error === undefined ? null : (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

function draftScopeLabel(draft: RunDraftSummary): string {
  return draft.projectId === undefined
    ? "your library"
    : `Project ${draft.projectId}`;
}

function RunDraftCapacity({
  drafts,
  onDiscard,
}: {
  drafts: RunDraftSummary[];
  onDiscard: (key: string) => void;
}) {
  return (
    <section className="run-draft-capacity" role="alert">
      <div>
        <strong>The in-memory Run draft limit is reached.</strong>
        <p>
          Choose one retained draft to discard. Contractor will not evict
          unsaved input automatically.
        </p>
      </div>
      <ul className="run-draft-capacity-list">
        {drafts.map((draft) => (
          <li key={draft.key}>
            <span>
              <code>
                {draft.workflowName}@{draft.workflowVersion}
              </code>
              <small>
                {draftScopeLabel(draft)}
                {draft.ambiguousSubmission
                  ? " · submission outcome unknown"
                  : ""}
              </small>
            </span>
            <Button
              variant="danger"
              size="sm"
              onClick={() => onDiscard(draft.key)}
            >
              Discard this draft
            </Button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function WorkflowRunDraftBoundary(props: WorkflowRunFormProps) {
  const store = useRunDraftStore();
  const identity: RunDraftIdentity = {
    workflowName: props.workflow.ref.name,
    workflowVersion: props.workflow.ref.version,
    ...(props.projectId === undefined ? {} : { projectId: props.projectId }),
  };
  const initialState = initialRunDraftState(
    props.initialArtifactSelections,
    props.initialArtifacts,
  );
  const [acquisition, setAcquisition] = useState(() =>
    store.acquire(identity, initialState),
  );

  function discardAtCapacity(key: string): void {
    store.discard(key);
    setAcquisition(store.acquire(identity, initialState));
  }

  if (acquisition.kind === "capacity") {
    return (
      <RunDraftCapacity
        drafts={acquisition.drafts}
        onDiscard={discardAtCapacity}
      />
    );
  }
  const entry = acquisition.entry;

  function discardCurrent(): void {
    store.discard(entry);
    setAcquisition(store.acquire(identity, initialState));
  }

  return (
    <WorkflowRunFormBody
      {...props}
      key={`${entry.key}:${entry.generation}`}
      draftEntry={entry}
      draftStore={store}
      onDiscardDraft={discardCurrent}
    />
  );
}

/**
 * The Run form of one exact Workflow version, standalone or in a Project.
 * Its draft lives in the tab's draft store, so closing the sheet keeps it;
 * a submission reuses one idempotency key while the request is unchanged.
 */
export function WorkflowRunForm(props: WorkflowRunFormProps) {
  const key = `${props.projectId ?? "standalone"}\u0000${props.workflow.ref.name}\u0000${props.workflow.ref.version}`;
  return <WorkflowRunDraftBoundary {...props} key={key} />;
}

function WorkflowRunFormBody({
  presentation = "page",
  onSubmittingChange,
  workflow,
  projectId,
  draftEntry,
  draftStore,
  onDiscardDraft,
}: WorkflowRunFormProps & {
  draftEntry: RunDraftEntry;
  draftStore: RunDraftMemoryStore;
  onDiscardDraft: () => void;
}) {
  const api = usePublicAPI();
  const navigate = useGuardedNavigate();
  const queryClient = useQueryClient();
  const state = useRunDraftState(draftEntry, draftStore);
  const { draft, errors, setErrors, clearError, change } = state;
  const [discardRequested, setDiscardRequested] = useState(false);
  const [ambiguousSubmission, setAmbiguousSubmission] = useState(
    draftEntry.ambiguousSubmission,
  );
  const [optionalParametersOpen, setOptionalParametersOpen] = useState(() =>
    Object.entries(workflow.parameters).some(
      ([name, slot]) => !slot.required && draft.parameters[name] !== undefined,
    ),
  );
  const [runtimeOptionsOpen, setRuntimeOptionsOpen] = useState(
    () =>
      draft.runtimeLabels.length > 0 ||
      draft.repeat?.notices.some((notice) =>
        notice.field?.startsWith("runtimeLabels"),
      ) === true,
  );
  const [metadataOptionsOpen, setMetadataOptionsOpen] = useState(
    () => draft.metadataLabels.length > 0,
  );
  const [executionOptionsOpen, setExecutionOptionsOpen] = useState(
    () =>
      Object.values(draft.overrides.planner).some((value) => value !== "") ||
      Object.values(draft.overrides.workers).some((value) => value !== "") ||
      Object.keys(draft.overrides.stages ?? {}).length > 0 ||
      draft.repeat?.notices.some(
        (notice) => notice.field === "executionConfig",
      ) === true,
  );
  const inventories = useRunFormInventories({
    projectId,
    knownArtifacts: draft.knownArtifacts,
    executionOptionsOpen,
    runtimeOptionsOpen,
  });
  const { artifactInventory, artifactMap } = inventories;
  const keyring = draftEntry.keyring;
  const keyScope =
    projectId === undefined ? "standalone" : `project:${projectId}`;

  function validate() {
    return validateRunDraft(
      workflow,
      {
        runtimeLabels: draft.runtimeLabels,
        metadataLabels: draft.metadataLabels,
        parameters: draft.parameters,
        artifacts: draft.artifactSelections,
        artifactReviews: draft.artifactReviews,
        overrides: draft.overrides,
      },
      artifactMap,
    );
  }
  const currentValidation = validate();

  const mutation = useMutation({
    mutationFn: ({
      request,
      idempotencyKey,
    }: {
      request: CreateRunRequest;
      idempotencyKey: string;
    }) =>
      projectId === undefined
        ? createRun(api, request, idempotencyKey)
        : createProjectRun(api, projectId, request, idempotencyKey),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.runs.all });
      if (projectId !== undefined) {
        await Promise.all([
          queryClient.invalidateQueries({
            queryKey: queryKeys.projects.runs(projectId),
          }),
          queryClient.invalidateQueries({
            queryKey: queryKeys.projects.artifacts.all(projectId),
          }),
        ]);
      }
      // Refresh confirmed server state even for an old submission, but only
      // finish this form if its draft survived the asynchronous refresh.
      if (!draftStore.isCurrent(draftEntry)) return;
      draftStore.discard(draftEntry);
      await navigate(`/runs/${encodeURIComponent(result.runId)}`);
    },
    onError: (error) => {
      const ambiguous = error instanceof PublicAPIError && error.status === 0;
      if (draftStore.setAmbiguousSubmission(draftEntry, ambiguous)) {
        setAmbiguousSubmission(ambiguous);
      }
    },
  });

  useEffect(() => {
    onSubmittingChange?.(mutation.isPending);
  }, [mutation.isPending, onSubmittingChange]);

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    if (mutation.isPending) return;
    mutation.reset();
    if (draft.repeat !== undefined && !draft.repeat.reviewed) {
      setErrors((current) => ({
        ...current,
        repeatReview:
          "Review and confirm the retained values before creating another Run.",
      }));
      return;
    }
    const validation = validate();
    setErrors(validation.errors);
    if (validation.request === undefined) {
      // Open the disclosures that hold an invalid value.
      const errorKeys = Object.keys(validation.errors);
      if (
        Object.entries(workflow.parameters).some(
          ([name, slot]) =>
            !slot.required && errorKeys.includes(`parameter:${name}`),
        )
      )
        setOptionalParametersOpen(true);
      if (errorKeys.includes("runtimeLabels")) {
        setRuntimeOptionsOpen(true);
      }
      if (
        errorKeys.some(
          (key) => key === "metadataLabels" || key.startsWith("metadataLabel:"),
        )
      ) {
        setMetadataOptionsOpen(true);
      }
      return;
    }
    const idempotencyKey = keyring.keyFor(validation.request, keyScope);
    draftStore.markSubmitted(draftEntry);
    draftStore.setAmbiguousSubmission(draftEntry, false);
    setAmbiguousSubmission(false);
    mutation.mutate({ request: validation.request, idempotencyKey });
  }

  const responseLost =
    ambiguousSubmission ||
    (mutation.error instanceof PublicAPIError && mutation.error.status === 0);
  const exactRetry =
    currentValidation.request !== undefined &&
    keyring.matches(currentValidation.request, keyScope);
  const readiness = runDraftReadiness({
    workflow,
    draft,
    valid: currentValidation.request !== undefined,
    artifactsPending: artifactInventory.isPending,
    inputNoun: projectId === undefined ? "file" : "material",
  });
  const parameterSlots = Object.entries(workflow.parameters);
  const requiredParameters = parameterSlots.filter(([, slot]) => slot.required);
  const optionalParameters = parameterSlots.filter(
    ([, slot]) => !slot.required,
  );

  function parameterFields(slots: typeof parameterSlots, title: string) {
    return (
      <ParameterFields
        slots={slots}
        title={title}
        parameters={draft.parameters}
        errors={errors}
        onChange={(name, value) => {
          change("parameters", (current) => ({ ...current, [name]: value }));
          clearError(`parameter:${name}`);
        }}
      />
    );
  }

  return (
    <form
      className="run-draft"
      id="workflow-run-form"
      onSubmit={submit}
      noValidate
    >
      {discardRequested ? (
        <AlertDialog
          className="run-draft-discard-dialog"
          eyebrow="Unsaved Run setup"
          title="Discard this Run draft?"
          closeLabel="Close discard confirmation"
          cancelLabel="Keep editing"
          confirmLabel="Discard Run draft"
          onCancel={() => setDiscardRequested(false)}
          onConfirm={onDiscardDraft}
        >
          <p>
            Parameters, input selections, labels, overrides and any ambiguous
            submission identity in this tab will be removed.
          </p>
        </AlertDialog>
      ) : null}
      {presentation === "page" ? (
        <div className="run-draft-heading">
          <div>
            <p className="eyebrow">Run setup</p>
            <h3>
              {projectId === undefined
                ? "Start Workflow Run"
                : "Start Project Workflow Run"}
            </h3>
            <p className="run-draft-intro">
              Complete the declared inputs. Optional settings keep the published
              Workflow defaults until you change them.
            </p>
            <code className="run-draft-workflow">
              {workflow.ref.name}@{workflow.ref.version}
            </code>
          </div>
          <span className="run-draft-readiness" aria-live="polite">
            <StatusChip tone={readiness.tone}>{readiness.value}</StatusChip>
            <small>{readiness.copy}</small>
          </span>
        </div>
      ) : (
        <p className="run-draft-readiness is-inline" aria-live="polite">
          <StatusChip tone={readiness.tone} size="sm">
            {readiness.value}
          </StatusChip>
          <span>{readiness.copy}</span>
        </p>
      )}

      {draft.repeat === undefined ? null : (
        <RepeatRunReview
          repeat={draft.repeat}
          error={errors.repeatReview}
          onReviewed={state.setRepeatReviewed}
        />
      )}

      {presentation === "page"
        ? parameterFields(parameterSlots, "String parameters")
        : null}

      <InputFields
        workflow={workflow}
        presentation={presentation}
        projectId={projectId}
        draft={draft}
        errors={errors}
        inventories={inventories}
        isCurrent={() => draftStore.isCurrent(draftEntry)}
        onSelect={state.selectArtifact}
        onConfirm={state.confirmArtifact}
      />
      {presentation === "drawer" ? (
        <>
          {requiredParameters.length > 0
            ? parameterFields(requiredParameters, "Required parameters")
            : null}
          {optionalParameters.length > 0 ? (
            <details
              className="run-draft-disclosure workflow-optional-parameters"
              open={optionalParametersOpen}
              onToggle={(event) =>
                setOptionalParametersOpen(event.currentTarget.open)
              }
            >
              <summary>
                <span className="run-draft-disclosure-title">
                  <strong>Optional parameters</strong>
                  <small>Keep omitted unless needed</small>
                </span>
                <span className="run-draft-disclosure-status">
                  {optionalParameters.length}
                </span>
              </summary>
              {parameterFields(optionalParameters, "Optional strings")}
            </details>
          ) : null}
        </>
      ) : null}

      <div className="run-draft-advanced-heading">
        <div>
          <p className="eyebrow">Optional setup</p>
          <h4>Advanced settings</h4>
        </div>
        <p>
          Keep these collapsed to use the published Workflow and Runtime
          defaults.
        </p>
      </div>

      <RuntimePlacement
        selected={draft.runtimeLabels}
        errors={errors}
        inventories={inventories}
        open={runtimeOptionsOpen}
        onOpenChange={setRuntimeOptionsOpen}
        onChange={(update) => {
          change("runtimeLabels", update);
          clearError("runtimeLabels");
        }}
      />
      <RunMetadata
        labels={draft.metadataLabels}
        labelInput={draft.metadataLabelInput}
        errors={errors}
        editor={state.metadataLabels}
        open={metadataOptionsOpen}
        onOpenChange={setMetadataOptionsOpen}
      />
      <ExecutionOverrides
        overrides={draft.overrides}
        plannerSupported={Object.values(workflow.stages).some(
          (stage) => stage.executionConfig.planner !== undefined,
        )}
        inventories={inventories}
        open={executionOptionsOpen}
        onOpenChange={setExecutionOptionsOpen}
        onOverride={state.updateOverride}
        onRemoveStageOverrides={() =>
          change("overrides", (current) => {
            const next = { ...current };
            delete next.stages;
            return next;
          })
        }
      />

      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
      {responseLost ? (
        <div className="run-draft-notice" role="alert">
          <strong>The Run may already exist.</strong>
          {exactRetry ? (
            <p>
              The draft is unchanged. “Retry same request” reuses the same
              idempotency key and cannot create a second Run for this request.
            </p>
          ) : (
            <p>
              This draft changed after the lost response. Submitting it uses a
              new key and may create another Run; restore the prior values to
              retry the ambiguous request first.
            </p>
          )}
        </div>
      ) : null}
      <div className="run-submit-row">
        <span className="run-submit-status">
          <strong>{readiness.ready ? "Ready to start" : readiness.copy}</strong>
          <small>The Run opens after the Server accepts the request.</small>
        </span>
        <div className="run-draft-actions">
          {draftEntry.meaningful ? (
            <Button
              variant="ghost"
              disabled={mutation.isPending}
              onClick={() => setDiscardRequested(true)}
            >
              Discard saved draft
            </Button>
          ) : null}
          <Button
            type="submit"
            variant="primary"
            pending={mutation.isPending}
            pendingLabel={
              projectId === undefined ? "Submitting…" : "Starting Project Run…"
            }
            disabled={
              readiness.repeatReviewPending ||
              (Object.keys(workflow.inputs).length > 0 &&
                artifactInventory.isPending)
            }
          >
            {responseLost && exactRetry
              ? "Retry same request"
              : responseLost
                ? "Start changed draft with a new key"
                : projectId === undefined
                  ? "Start Workflow Run"
                  : "Start Project Workflow Run"}
          </Button>
        </div>
      </div>
    </form>
  );
}
