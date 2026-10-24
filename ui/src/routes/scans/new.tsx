import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import {
  getProject,
  PROJECT_ID_PATTERN,
  type Project,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import {
  createProjectRun,
  createRun,
  getWorkflow,
  type CreateRunRequest,
  type WorkflowResource,
} from "../../api/workflows";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { LoadMoreButton } from "../../app/load-more";
import { useGuardedNavigate } from "../../app/page-location";
import { QueryView } from "../../app/query-view";
import { workflowWidgets } from "../../workflow-widgets/registry";
import { RunDraftKeyring } from "../../run-drafts/idempotency";
import { Button, ButtonLink, DetailHeader, Field } from "../../ui";
import { ScanSettingsFields } from "./settings";
import { useScanProjects, useScanSettings } from "./settings-state";
import "./scans.css";

export function WebScanRoute() {
  const api = usePublicAPI();
  const [params] = useSearchParams();
  const projectId = params.get("project");
  useDocumentTitle("Start a web scan");
  const workflow = useQuery({
    queryKey: queryKeys.workflows.detail(
      workflowWidgets.webScan.workflow.name,
      workflowWidgets.webScan.workflow.version,
    ),
    queryFn: ({ signal }) =>
      getWorkflow(
        api,
        workflowWidgets.webScan.workflow.name,
        workflowWidgets.webScan.workflow.version,
        signal,
      ),
  });
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId ?? ""),
    queryFn: () => getProject(api, projectId ?? ""),
    enabled: projectId !== null && PROJECT_ID_PATTERN.test(projectId),
  });
  return (
    <section className="route-page scan-page">
      <DetailHeader
        title="Start a web scan"
        breadcrumb={[{ label: "Checks", to: "/checks" }, { label: "Web scan" }]}
        actions={<ButtonLink to="/checks">Back to checks</ButtonLink>}
      />
      <p>
        Scan one HTTP target with Nuclei. Results are saved as an ordinary
        check.
      </p>
      {projectId !== null && !PROJECT_ID_PATTERN.test(projectId) ? (
        <ErrorNotice error={new Error("Project ID is invalid")} />
      ) : projectId !== null && project.data === undefined ? (
        <QueryView query={project}>{() => null}</QueryView>
      ) : (
        <QueryView query={workflow}>
          {(data) => (
            <WebScanForm
              key={projectId ?? "standalone"}
              workflow={data}
              initialProject={project.data}
            />
          )}
        </QueryView>
      )}
    </section>
  );
}
function WebScanForm({
  workflow,
  initialProject,
}: {
  workflow: WorkflowResource;
  initialProject: Project | undefined;
}) {
  const api = usePublicAPI();
  const navigate = useGuardedNavigate();
  const cache = useQueryClient();
  const projects = useScanProjects();
  const [project, setProject] = useState(initialProject);
  const settings = useScanSettings(initialProject?.httpTarget?.url ?? "");
  const [keyring] = useState(() => new RunDraftKeyring());
  const mutation = useMutation({
    mutationFn: ({
      request,
      key,
      projectId,
    }: {
      request: CreateRunRequest;
      key: string;
      projectId: string | undefined;
    }) =>
      projectId === undefined
        ? createRun(api, request, key)
        : createProjectRun(api, projectId, request, key),
    onSuccess: async (result) => {
      await cache.invalidateQueries({ queryKey: queryKeys.runs.all });
      if (result.projectId !== undefined)
        await cache.invalidateQueries({
          queryKey: queryKeys.projects.runs(result.projectId),
        });
      await navigate(`/runs/${encodeURIComponent(result.runId)}`);
    },
  });
  const compatible = workflowWidgets.webScan.supports(workflow);
  const items = new Map(
    (projects.data?.pages.flatMap((page) => page.items) ?? [])
      .filter((item) => item.lifecycle === "active")
      .map((item) => [item.projectId, item]),
  );
  if (project !== undefined) items.set(project.projectId, project);
  function submit(event: FormEvent) {
    event.preventDefault();
    if (mutation.isPending || !compatible || project?.lifecycle === "deleting")
      return;
    const selection = settings.request();
    const request: CreateRunRequest = {
      workflow: `${workflow.ref.name}@${workflow.ref.version}`,
      parameters: {
        ...selection.parameters,
        target: settings.parameters.target ?? "",
      },
      ...(selection.rulepack === undefined
        ? {}
        : { artifacts: { rulepack: selection.rulepack } }),
    };
    mutation.mutate({
      request,
      key: keyring.keyFor(request, project?.projectId ?? "standalone"),
      projectId: project?.projectId,
    });
  }
  return (
    <form onSubmit={submit} className="ui-stack">
      <fieldset className="scan-fields ui-stack" disabled={mutation.isPending}>
        <Field label="Save results in">
          {(control) => (
            <select
              {...control}
              value={project?.projectId ?? ""}
              onChange={(event) => {
                const next = items.get(event.target.value);
                setProject(next);
                settings.change("target", next?.httpTarget?.url ?? "");
              }}
            >
              <option value="">Standalone scan</option>
              {[...items].map(([id, item]) => (
                <option key={id} value={id}>
                  {item.name}
                </option>
              ))}
            </select>
          )}
        </Field>
        <LoadMoreButton query={projects} label="Load more Projects" />
        {projects.error === null ? null : (
          <ErrorNotice
            error={projects.error}
            onRetry={() => void projects.refetch()}
          />
        )}
        {project === undefined ? null : (
          <p className="ui-caption">
            The target is filled from the Project. Review it before starting
            this scan.
          </p>
        )}
        <ScanSettingsFields settings={settings} workflow={workflow} target />
      </fieldset>
      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
      {mutation.error instanceof PublicAPIError &&
      (mutation.error.status === 0 || mutation.error.retryable) ? (
        <p role="alert">
          The scan may already have been created. Retry without changing the
          form to reuse the same request key.
        </p>
      ) : null}
      {project?.lifecycle === "deleting" ? (
        <p role="alert">This Project is deleting. Choose another Project.</p>
      ) : null}
      <Button
        type="submit"
        variant="primary"
        pending={mutation.isPending}
        pendingLabel="Starting scan…"
        disabled={
          !compatible ||
          project?.lifecycle === "deleting" ||
          (settings.custom && settings.pack === undefined)
        }
      >
        Start scan
      </Button>
    </form>
  );
}
