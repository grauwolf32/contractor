import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import type { Project } from "../../api/projects";
import {
  createScanSweep,
  MAX_SWEEP_PROJECTS,
  sweepKeys,
  type CreateScanSweepRequest,
} from "../../api/scan-sweeps";
import { getWorkflow, type WorkflowResource } from "../../api/workflows";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { LoadMoreButton } from "../../app/load-more";
import { useGuardedNavigate } from "../../app/page-location";
import { QueryView } from "../../app/query-view";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import { Button, ButtonLink, DetailHeader, Field } from "../../ui";
import { workflowWidgets } from "../../workflow-widgets/registry";
import { ScanSettingsFields } from "./settings";
import { useScanProjects, useScanSettings } from "./settings-state";
import "./scans.css";

export function NewSweepRoute() {
  useDocumentTitle("Start a rule sweep");
  const api = usePublicAPI();
  const widget = workflowWidgets.webScan;
  const workflow = useQuery({
    queryKey: [
      "workflows",
      "detail",
      widget.workflow.name,
      widget.workflow.version,
    ],
    queryFn: ({ signal }) =>
      getWorkflow(api, widget.workflow.name, widget.workflow.version, signal),
  });
  return (
    <section className="route-page scan-page">
      <DetailHeader
        title="Start a rule sweep"
        breadcrumb={[
          { label: "Checks", to: "/checks" },
          { label: "Rule sweeps", to: "/checks/sweeps" },
          { label: "Start" },
        ]}
        actions={<ButtonLink to="/checks/sweeps">Back to sweeps</ButtonLink>}
      />
      <p>
        Choose one rule and a set of Projects. Each scan uses its Project's
        target captured by the Server at launch.
      </p>
      <QueryView query={workflow}>
        {(value) => <SweepForm workflow={value} />}
      </QueryView>
    </section>
  );
}
function SweepForm({ workflow }: { workflow: WorkflowResource }) {
  const api = usePublicAPI();
  const cache = useQueryClient();
  const navigate = useGuardedNavigate();
  const projects = useScanProjects();
  const settings = useScanSettings();
  const [all, setAll] = useState(false);
  const [selected, setSelected] = useState(() => new Map<string, Project>());
  const [keyring] = useState(
    () => new MutationDraftKeyring<CreateScanSweepRequest>("sweep"),
  );
  const mutation = useMutation({
    mutationFn: ({
      request,
      key,
    }: {
      request: CreateScanSweepRequest;
      key: string;
    }) => createScanSweep(api, request, key),
    onSuccess: async (value) => {
      await cache.invalidateQueries({ queryKey: sweepKeys.all });
      await navigate(`/checks/sweeps/${encodeURIComponent(value.sweepId)}`);
    },
  });
  const items = new Map(
    (projects.data?.pages.flatMap((page) => page.items) ?? []).map((item) => [
      item.projectId,
      item,
    ]),
  );
  for (const [id, value] of selected) items.set(id, value);
  const compatible = workflowWidgets.webScan.supports(workflow);
  function submit(event: FormEvent) {
    event.preventDefault();
    if (mutation.isPending || !compatible || (!all && selected.size === 0))
      return;
    const request: CreateScanSweepRequest = {
      workflow: `${workflow.ref.name}@${workflow.ref.version}`,
      ...settings.request(),
      ...(all
        ? { allProjects: true }
        : { projectIds: [...selected.keys()].sort() }),
    };
    mutation.mutate({ request, key: keyring.keyFor(request) });
  }
  return (
    <form className="ui-stack" onSubmit={submit}>
      <fieldset className="scan-fields ui-stack" disabled={mutation.isPending}>
        <Field label="Project scope">
          {(control) => (
            <select
              {...control}
              value={all ? "all" : "selected"}
              onChange={(event) => setAll(event.target.value === "all")}
            >
              <option value="selected">Selected Projects</option>
              <option value="all">
                All my Projects (up to {MAX_SWEEP_PROJECTS})
              </option>
            </select>
          )}
        </Field>
        {all ? (
          <p>
            The Server captures the complete scope once. Later target edits and
            library updates do not change this sweep.
          </p>
        ) : (
          <>
            <p role="status">
              {selected.size} Projects selected · maximum {MAX_SWEEP_PROJECTS}
            </p>
            <div
              className="scan-project-options ui-panel"
              role="group"
              aria-label="Select Projects"
            >
              {[...items].map(([id, item]) => (
                <label key={id}>
                  <input
                    type="checkbox"
                    checked={selected.has(id)}
                    disabled={
                      !selected.has(id) && selected.size >= MAX_SWEEP_PROJECTS
                    }
                    onChange={(event) => {
                      const checked = event.target.checked;
                      setSelected((current) => {
                        const next = new Map(current);
                        if (checked && next.size < MAX_SWEEP_PROJECTS)
                          next.set(id, item);
                        else next.delete(id);
                        return next;
                      });
                    }}
                  />
                  <span>
                    {item.name}
                    <small className="ui-caption">
                      {item.lifecycle === "deleting"
                        ? "Deleting · will be skipped"
                        : (item.httpTarget?.url ??
                          "No target · will be skipped")}
                    </small>
                  </span>
                </label>
              ))}
            </div>
            {projects.isPending ? <p role="status">Loading Projects…</p> : null}
            <LoadMoreButton query={projects} label="Load more Projects" />
            {projects.error === null ? null : (
              <ErrorNotice
                error={projects.error}
                onRetry={() => void projects.refetch()}
              />
            )}
          </>
        )}
        <ScanSettingsFields settings={settings} workflow={workflow} />
      </fieldset>
      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
      {mutation.error instanceof PublicAPIError &&
      (mutation.error.status === 0 || mutation.error.retryable) ? (
        <p role="alert">
          The sweep may already exist. Retry this unchanged form to reuse its
          creation key.
        </p>
      ) : null}
      <Button
        type="submit"
        variant="primary"
        pending={mutation.isPending}
        pendingLabel="Starting sweep…"
        disabled={
          !compatible ||
          (!all && selected.size === 0) ||
          (settings.custom && settings.pack === undefined)
        }
      >
        Start sweep
      </Button>
    </form>
  );
}
