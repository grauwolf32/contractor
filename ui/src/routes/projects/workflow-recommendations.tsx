import { useLocation, useSearchParams } from "react-router";
import { WorkflowRunDrawer } from "../workflows/run-drawer";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { usePublicAPI } from "../../api/context";
import { useProjectArtifactInventory } from "./inventory";
import { queryKeys } from "../../api/query-keys";
import { getWorkflow } from "../../api/workflows";
import { WorkflowCard } from "../workflows/card";
import { useWorkflowFamilies } from "../workflows/families";
import { useWorkflowInventory } from "../workflows/inventory";
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
import { ProjectSectionActions } from "./navigation";

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
  function renderForm(onSubmittingChange?: (pending: boolean) => void) {
    return workflow.isPending ? (
      <p className="loading-copy" role="status">
        Loading Workflow contract…
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
        presentation="drawer"
        {...(onSubmittingChange ? { onSubmittingChange } : {})}
      />
    );
  }
  return (
    <WorkflowRunDrawer
      workflow={selection.workflow}
      projectId={projectId}
      onClose={onClose}
    >
      {renderForm}
    </WorkflowRunDrawer>
  );
}

export function ProjectWorkflowRecommendations({
  projectId,
}: {
  projectId: string;
}) {
  const [selection, setSelection] = useState<WorkflowCompatibility | null>(
    null,
  );
  const workflows = useWorkflowInventory();
  const artifacts = useProjectArtifactInventory(projectId);
  const artifactItems = artifacts.data;
  const compatibility = useMemo(
    () => buildWorkflowCompatibility(workflows.data ?? [], artifactItems ?? []),
    [artifactItems, workflows.data],
  );
  const { families, selectVersion } = useWorkflowFamilies(workflows.data ?? []);
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const filter = ["recommended", "all", "missing"].includes(
    filters.get("workflowFilter") ?? "",
  )
    ? filters.get("workflowFilter")!
    : "recommended";
  const search = filters.get("workflowQ") ?? "";
  function updateFilter(key: string, value: string) {
    setFilters(
      (current) => {
        const next = new URLSearchParams(current);
        if (value === "") next.delete(key);
        else next.set(key, value);
        return next;
      },
      { replace: true, preventScrollReset: true, state: location.state },
    );
  }
  const setFilter = (value: string) => updateFilter("workflowFilter", value);
  const setSearch = (value: string) => updateFilter("workflowQ", value);
  const selectedItems = families.map((family) => {
    const selected =
      family.versions.find(
        (workflow) =>
          workflow.ref.version ===
          filters.get(`workflowVersion.${family.name}`),
      ) ?? family.workflow;
    return {
      ...family,
      workflow: selected,
      matching: compatibility.find(
        (item) => selector(item) === workflowSelector(selected),
      )!,
    };
  });
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

  if (workflows.isPending || artifacts.isPending) {
    return (
      <section className="panel project-region" id="project-workflows">
        <p className="loading-copy" role="status">
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
        <button
          className="secondary-button"
          type="button"
          disabled={workflows.isFetching || artifacts.isFetching}
          onClick={() => {
            void workflows.refetch();
            void artifacts.refetch();
          }}
        >
          Retry matching
        </button>
      </section>
    );
  }

  return (
    <section className="panel project-region" id="project-workflows">
      <ProjectSectionActions>
        <span className="muted-copy">
          {families.length} workflows · {workflows.data?.length ?? 0} versions
        </span>
      </ProjectSectionActions>
      <div className="workflow-discovery-toolbar">
        <div className="workflow-filter-tabs" aria-label="Workflow filters">
          {[
            ["recommended", "Format matches"],
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
      {visible.length === 0 ? (
        <div className="compact-empty">
          <strong>
            {filter === "recommended" && recommended.length === 0
              ? "No new format matches."
              : "No matching Workflows."}
          </strong>
          <p>
            Inspect all Workflows to review missing inputs or explicitly run
            again.
          </p>
          <button
            className="secondary-button"
            type="button"
            onClick={() => {
              setFilters(
                (current) => {
                  const next = new URLSearchParams(current);
                  next.set("workflowFilter", "all");
                  next.delete("workflowQ");
                  return next;
                },
                {
                  replace: true,
                  preventScrollReset: true,
                  state: location.state,
                },
              );
            }}
          >
            Show all workflows
          </button>
        </div>
      ) : (
        <div className="workflow-family-grid">
          {visible.map(({ name, versions, workflow, matching }) => (
            <WorkflowCard
              key={name}
              workflow={workflow}
              versions={versions}
              onVersion={(version) => {
                selectVersion(name, version);
                updateFilter(`workflowVersion.${name}`, version);
              }}
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
