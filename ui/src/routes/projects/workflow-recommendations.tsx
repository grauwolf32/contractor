import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { getWorkflow } from "../../api/workflows";
import { WorkflowCard } from "../workflows/card";
import { useWorkflowFamilies } from "../workflows/families";
import { useWorkflowInventory } from "../workflows/inventory";
import { Dialog } from "../../app/dialog";
import { artifactOptionKey } from "../../run-drafts/validation";
import { ErrorNotice } from "../artifacts/common";
import { WorkflowRunForm } from "../workflows/run-form";
import {
  workflowDisplayName,
  workflowSelector,
} from "../workflows/presentation";
import {
  buildWorkflowCompatibility,
  type WorkflowCompatibility,
} from "./recommendations";

const INITIAL_CURSOR = null;

function nextCursor(page: { page: { hasMore: boolean; nextCursor?: string } }) {
  return page.page.hasMore ? page.page.nextCursor : undefined;
}

function selector(item: WorkflowCompatibility): string {
  return workflowSelector(item.workflow);
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
  const initialArtifacts = useMemo(() => {
    const selected = new Set(Object.values(selection.preselected));
    return Array.from(
      new Map(
        Object.values(selection.candidates)
          .flat()
          .filter((metadata) =>
            selected.has(artifactOptionKey(metadata.artifact)),
          )
          .map((metadata) => [artifactOptionKey(metadata.artifact), metadata]),
      ).values(),
    );
  }, [selection]);
  return (
    <Dialog
      className="project-dialog project-workflow-dialog panel"
      labelledBy="project-workflow-dialog-title"
      onRequestClose={onClose}
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Project Workflow</p>
          <h2 id="project-workflow-dialog-title">
            {workflowDisplayName(selection.workflow)}
          </h2>
          <code>{selector(selection)}</code>
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
          initialArtifacts={initialArtifacts}
        />
      )}
    </Dialog>
  );
}

export function ProjectWorkflowRecommendations({
  projectId,
  focusRequest = 0,
}: {
  projectId: string;
  focusRequest?: number;
}) {
  const api = usePublicAPI();
  const section = useRef<HTMLElement>(null);
  const handledFocusRequest = useRef(0);
  const [selection, setSelection] = useState<WorkflowCompatibility | null>(
    null,
  );
  const workflows = useWorkflowInventory();
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
  const artifactItems = useMemo(
    () => artifacts.data?.pages.flatMap((page) => page.items ?? []) ?? [],
    [artifacts.data],
  );
  const compatibility = useMemo(
    () => buildWorkflowCompatibility(workflows.data ?? [], artifactItems),
    [artifactItems, workflows.data],
  );
  const { families, selectVersion } = useWorkflowFamilies(workflows.data ?? []);
  const [filter, setFilter] = useState("recommended");
  const [search, setSearch] = useState("");
  const selectedItems = families.map((family) => ({
    ...family,
    matching: compatibility.find(
      (item) => selector(item) === workflowSelector(family.workflow),
    )!,
  }));
  const recommended = selectedItems.filter(
    ({ matching }) => matching.compatible && !matching.suppressed,
  );
  const visible = selectedItems.filter(
    ({ versions, matching }) =>
      (filter === "all" ||
        (filter === "recommended"
          ? matching.compatible && !matching.suppressed
          : !matching.compatible)) &&
      versions.some((w) =>
        `${workflowDisplayName(w)} ${w.ref.name} ${w.presentation?.description ?? ""}`
          .toLowerCase()
          .includes(search.toLowerCase()),
      ),
  );
  const inventoryComplete = !artifacts.hasNextPage;

  useEffect(() => {
    if (
      focusRequest === 0 ||
      focusRequest === handledFocusRequest.current ||
      workflows.isPending ||
      artifacts.isPending
    ) {
      return;
    }
    handledFocusRequest.current = focusRequest;
    const region = section.current;
    const target =
      region?.querySelector<HTMLElement>(
        'button[aria-label^="Run "]:not(:disabled), button[aria-label^="Configure "]:not(:disabled), .project-all-workflows > summary',
      ) ?? region;
    region?.scrollIntoView?.({ block: "start", behavior: "smooth" });
    target?.focus({ preventScroll: true });
  }, [artifacts.isPending, focusRequest, workflows.isPending]);

  async function loadRemainingPage(): Promise<void> {
    await Promise.all([
      artifacts.hasNextPage ? artifacts.fetchNextPage() : Promise.resolve(),
    ]);
  }

  if (workflows.isPending || artifacts.isPending) {
    return (
      <section
        ref={section}
        className="panel project-region"
        id="project-workflows"
        tabIndex={-1}
      >
        <p className="loading-copy" aria-live="polite">
          Matching Workflows to Project Artifacts…
        </p>
      </section>
    );
  }
  if (workflows.error !== null || artifacts.error !== null) {
    return (
      <section
        ref={section}
        className="panel project-region"
        id="project-workflows"
        tabIndex={-1}
      >
        <p className="eyebrow">Workflow matching</p>
        <h3>Recommended Workflows</h3>
        <ErrorNotice error={workflows.error ?? artifacts.error} />
      </section>
    );
  }

  return (
    <section
      ref={section}
      className="panel project-region"
      id="project-workflows"
      tabIndex={-1}
    >
      <div className="section-heading">
        <div>
          <h3>Workflows</h3>
          <p className="muted-copy">
            Choose a Workflow and review its exact inputs before launching.
          </p>
        </div>
        <span className="muted-copy">
          {families.length} workflows · {workflows.data?.length ?? 0} versions
        </span>
      </div>
      <div className="workflow-discovery-toolbar">
        <div className="workflow-filter-tabs" aria-label="Workflow filters">
          {[
            ["recommended", "Recommended"],
            ["all", "All workflows"],
            ["missing", "Missing inputs"],
          ].map(([value, label]) => (
            <button
              key={value}
              type="button"
              className="secondary-button"
              aria-pressed={filter === value}
              onClick={() => setFilter(value!)}
            >
              {label}
            </button>
          ))}
        </div>
        <label className="catalog-search">
          Find workflow
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Name or description"
          />
        </label>
      </div>
      {!inventoryComplete ? (
        <div className="notice notice-warning">
          <strong>More Project artifacts are available.</strong>
          <p>Load all materials before checking input matches.</p>
          <button
            className="secondary-button"
            type="button"
            disabled={artifacts.isFetchingNextPage}
            onClick={() => void loadRemainingPage()}
          >
            {artifacts.isFetchingNextPage
              ? "Loading next page…"
              : "Load next catalog page"}
          </button>
        </div>
      ) : visible.length === 0 ? (
        <div className="compact-empty">
          <strong>
            {filter === "recommended" && recommended.length === 0
              ? "No new Workflow result is recommended."
              : "No matching Workflows."}
          </strong>
          <p>
            Use All workflows to inspect missing inputs or explicitly run again.
          </p>
        </div>
      ) : (
        <div className="workflow-family-grid">
          {visible.map(({ name, versions, workflow, matching }) => (
            <WorkflowCard
              key={name}
              workflow={workflow}
              versions={versions}
              onVersion={(version) => selectVersion(name, version)}
              matching={matching}
              onConfigure={() => setSelection(matching)}
            />
          ))}
        </div>
      )}
      <p className="muted-copy workflow-card-parameters">
        Input matches compare media types. Review content and parameters in the
        Run form.
      </p>

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
