import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { getWorkflow, listWorkflows } from "../../api/workflows";
import { ErrorNotice } from "../artifacts/common";
import { WorkflowRunForm } from "../workflows/run-form";
import {
  buildWorkflowCompatibility,
  type WorkflowCompatibility,
} from "./recommendations";

const INITIAL_CURSOR = null;

function nextCursor(page: { page: { hasMore: boolean; nextCursor?: string } }) {
  return page.page.hasMore ? page.page.nextCursor : undefined;
}

function selector(item: WorkflowCompatibility): string {
  return `${item.workflow.ref.name}@${item.workflow.ref.version}`;
}

function WorkflowCompatibilityCard({
  item,
  allWorkflows,
  onRun,
}: {
  item: WorkflowCompatibility;
  allWorkflows: boolean;
  onRun: (item: WorkflowCompatibility) => void;
}) {
  const inputEntries = Object.entries(item.workflow.inputs ?? {}).sort(
    ([left], [right]) => left.localeCompare(right),
  );
  return (
    <article
      className={`project-workflow-card ${item.compatible ? "is-compatible" : "is-blocked"}`}
    >
      <div>
        <p className="eyebrow">
          {item.suppressed
            ? "Primary result exists"
            : item.compatible
              ? "Inputs available"
              : "Missing inputs"}
        </p>
        <h4>{selector(item)}</h4>
      </div>
      {inputEntries.length === 0 ? (
        <p className="project-workflow-fact">No Artifact inputs required.</p>
      ) : (
        <dl className="project-workflow-inputs">
          {inputEntries.map(([name, slot]) => {
            const count = item.candidates[name]?.length ?? 0;
            return (
              <div key={name}>
                <dt>
                  <code>{name}</code>
                  {slot.required ? <strong>required</strong> : null}
                </dt>
                <dd>
                  {count === 0
                    ? "no match"
                    : count === 1
                      ? "1 exact candidate"
                      : `${count} candidates · choose in form`}
                </dd>
              </div>
            );
          })}
        </dl>
      )}
      <div className="project-workflow-output-copy">
        <span>Primary outputs</span>
        {item.primaryOutputs.length === 0 ? (
          <small>none declared</small>
        ) : (
          <code>{item.primaryOutputs.join(", ")}</code>
        )}
      </div>
      {!item.compatible ? (
        <p className="project-workflow-missing">
          Add compatible {item.missingRequiredInputs.join(", ")} Artifact
          {item.missingRequiredInputs.length === 1 ? "" : "s"} to enable this
          Workflow.
        </p>
      ) : null}
      <button
        className={allWorkflows ? "secondary-button" : undefined}
        type="button"
        disabled={!item.compatible && !allWorkflows}
        aria-label={
          item.suppressed
            ? `Run again ${selector(item)}`
            : allWorkflows
              ? `Configure ${selector(item)} from all workflows`
              : `Run ${selector(item)}`
        }
        onClick={() => onRun(item)}
      >
        {item.suppressed ? "Run again" : "Configure Run"}
      </button>
    </article>
  );
}

function ProjectWorkflowLauncher({
  projectId,
  selection,
  onClose,
}: {
  projectId: string;
  selection: WorkflowCompatibility;
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const workflow = useQuery({
    queryKey: queryKeys.workflows.detail(
      selection.workflow.ref.name,
      selection.workflow.ref.version,
    ),
    queryFn: () =>
      getWorkflow(
        api,
        selection.workflow.ref.name,
        selection.workflow.ref.version,
      ),
  });
  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="project-dialog-backdrop" role="presentation">
      <section
        className="project-dialog project-workflow-dialog panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="project-workflow-dialog-title"
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">Project Workflow</p>
            <h2 id="project-workflow-dialog-title">{selector(selection)}</h2>
          </div>
          <button
            className="project-dialog-close"
            type="button"
            aria-label="Close Workflow Run dialog"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        {workflow.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading exact Workflow contract…
          </p>
        ) : workflow.error !== null ? (
          <ErrorNotice error={workflow.error} />
        ) : (
          <WorkflowRunForm
            key={`${projectId}:${selector(selection)}`}
            workflow={workflow.data}
            projectId={projectId}
            initialArtifactSelections={selection.preselected}
          />
        )}
      </section>
    </div>
  );
}

export function ProjectWorkflowRecommendations({
  projectId,
}: {
  projectId: string;
}) {
  const api = usePublicAPI();
  const [selection, setSelection] = useState<WorkflowCompatibility | null>(
    null,
  );
  const workflows = useInfiniteQuery({
    queryKey: queryKeys.workflows.picker,
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listWorkflows(api, pageParam === null ? {} : { cursor: pageParam }),
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
  const workflowItems = useMemo(
    () => workflows.data?.pages.flatMap((page) => page.items ?? []) ?? [],
    [workflows.data],
  );
  const artifactItems = useMemo(
    () => artifacts.data?.pages.flatMap((page) => page.items ?? []) ?? [],
    [artifacts.data],
  );
  const compatibility = useMemo(
    () => buildWorkflowCompatibility(workflowItems, artifactItems),
    [artifactItems, workflowItems],
  );
  const recommended = compatibility.filter(
    (item) => item.compatible && !item.suppressed,
  );
  const inventoryComplete = !workflows.hasNextPage && !artifacts.hasNextPage;

  async function loadRemainingPage(): Promise<void> {
    await Promise.all([
      workflows.hasNextPage ? workflows.fetchNextPage() : Promise.resolve(),
      artifacts.hasNextPage ? artifacts.fetchNextPage() : Promise.resolve(),
    ]);
  }

  if (workflows.isPending || artifacts.isPending) {
    return (
      <section className="panel project-region" id="project-workflows">
        <p className="loading-copy" aria-live="polite">
          Matching Workflows to Project Artifacts…
        </p>
      </section>
    );
  }
  if (workflows.error !== null || artifacts.error !== null) {
    return (
      <section className="panel project-region" id="project-workflows">
        <p className="eyebrow">Workflow matching</p>
        <h3>Recommended Workflows</h3>
        <ErrorNotice error={workflows.error ?? artifacts.error} />
      </section>
    );
  }

  return (
    <section className="panel project-region" id="project-workflows">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Advisory matching</p>
          <h3>Recommended Workflows</h3>
        </div>
        <span className="project-workflow-count">
          {recommended.length} ready
        </span>
      </div>
      <p className="muted-copy">
        Compatibility uses current media types only. Go Server validates every
        exact selection again when the Project Run is created.
      </p>

      {!inventoryComplete ? (
        <div className="notice notice-warning">
          <strong>More catalog data is available.</strong>
          <p>
            Finish loading before recommendations are shown, so a later Artifact
            page cannot change compatibility or output suppression.
          </p>
          <button
            className="secondary-button"
            type="button"
            disabled={
              workflows.isFetchingNextPage || artifacts.isFetchingNextPage
            }
            onClick={() => void loadRemainingPage()}
          >
            {workflows.isFetchingNextPage || artifacts.isFetchingNextPage
              ? "Loading next page…"
              : "Load next catalog page"}
          </button>
        </div>
      ) : recommended.length === 0 ? (
        <div className="compact-empty">
          <strong>No new Workflow result is recommended.</strong>
          <p>
            Required inputs may be missing, or every primary output may already
            exist. All workflows below keeps explicit recomputation available.
          </p>
        </div>
      ) : (
        <div className="project-workflow-grid">
          {recommended.map((item) => (
            <WorkflowCompatibilityCard
              key={`recommended:${selector(item)}`}
              item={item}
              allWorkflows={false}
              onRun={setSelection}
            />
          ))}
        </div>
      )}

      <details className="project-all-workflows">
        <summary>
          <span>
            <strong>All workflows</strong>
            <small>Inspect missing inputs or explicitly run again</small>
          </span>
          <span>{compatibility.length} loaded</span>
        </summary>
        {compatibility.length === 0 ? (
          <div className="compact-empty">No published Workflows found.</div>
        ) : (
          <div className="project-workflow-grid is-all">
            {compatibility.map((item) => (
              <WorkflowCompatibilityCard
                key={`all:${selector(item)}`}
                item={item}
                allWorkflows
                onRun={setSelection}
              />
            ))}
          </div>
        )}
      </details>

      {selection === null ? null : (
        <ProjectWorkflowLauncher
          projectId={projectId}
          selection={selection}
          onClose={() => setSelection(null)}
        />
      )}
    </section>
  );
}
