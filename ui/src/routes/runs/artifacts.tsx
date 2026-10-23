import "./runs.css";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  FINDING_PROPOSAL_NAMESPACE,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { listRunArtifacts, RUN_ID_PATTERN } from "../../api/runs";
import { ReturnLink } from "../../app/context-navigation";
import { CursorControls } from "../../app/cursor-controls";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { useCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { EmptyState, FilterChips } from "../../ui";
import { ArtifactDetailView } from "../artifacts/artifact-detail-view";
import { ArtifactBindingsTable } from "../artifacts/bindings";
import { useNamespaceFilter } from "../artifacts/namespace-filter";
import { artifactDetailPath } from "../artifacts/paths";
import { type RunDisclosureProps, RunSection } from "./components";

/** Technical details: every Artifact binding of the Run, by namespace. */
export function RunArtifactLibrary({
  runId,
  ...disclosure
}: { runId: string } & RunDisclosureProps) {
  const api = usePublicAPI();
  const [namespace, setNamespace] = useState<string | undefined>();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.runs.artifacts(runId, namespace, cursor),
    queryFn: () =>
      listRunArtifacts(api, {
        runId,
        ...(namespace === undefined ? {} : { namespace }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  const namespaceFilter = useNamespaceFilter({
    value: namespace,
    onApply: (candidate) => {
      setNamespace(candidate);
      pages.reset();
    },
  });

  const count =
    query.data === undefined
      ? undefined
      : `${query.data.items.length}${query.data.page.hasMore ? "+" : ""}`;
  return (
    <RunSection
      id="run-artifacts"
      title="Run artifacts"
      description="Inputs, intermediate Artifacts and outputs"
      aside={
        count === undefined
          ? undefined
          : `${count} Artifact${count === "1" ? "" : "s"}${namespace === undefined ? "" : ` in ${namespace}`}`
      }
      {...disclosure}
    >
      <div className="runs-namespace-filter">
        <FilterChips
          label="Run materials"
          options={[
            { value: "", label: "All materials" },
            {
              value: FINDING_PROPOSAL_NAMESPACE,
              label: "Possible issues and evidence",
            },
          ]}
          value={namespace ?? ""}
          onChange={(value) => {
            setNamespace(value || undefined);
            pages.reset();
          }}
        />
        {namespaceFilter.form}
        {namespaceFilter.error}
      </div>
      <p className="ui-caption">
        HTTP exchanges are saved with possible issues. Other proxied traffic is
        available in Caido.
      </p>
      <QueryView
        query={query}
        loading={
          <p className="ui-caption" role="status">
            Loading Run Artifacts…
          </p>
        }
        isEmpty={(page) => page.items.length === 0}
        empty={<EmptyState title="No bindings match this view." />}
      >
        {(page) => (
          <ArtifactBindingsTable
            items={page.items}
            returnLabel="Run results"
            showLocked
            detailPath={(item) =>
              artifactDetailPath({ kind: "run", id: runId }, item.artifact)
            }
          />
        )}
      </QueryView>
      <CursorControls
        label="Run Artifact pages"
        {...pages.controls(query.data?.page)}
      />
    </RunSection>
  );
}

export function RunArtifactDetailRoute() {
  const { runId = "", namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const valid =
    RUN_ID_PATTERN.test(runId) &&
    ARTIFACT_NAME_PATTERN.test(namespace) &&
    ARTIFACT_NAME_PATTERN.test(name) &&
    (revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision));
  useDocumentTitle(valid ? `${namespace}/${name} · Run` : "Run Artifact");
  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Run Artifact route is invalid")} />
        <Link to="/runs">Return to Runs</Link>
      </section>
    );
  }
  return (
    <ArtifactDetailView
      key={`${runId}/${namespace}/${name}`}
      scope={{ kind: "run", id: runId }}
      namespace={namespace}
      name={name}
      revision={revision}
      className="runs-artifact-page"
      heading={
        <>
          <ReturnLink to={`/runs/${encodeURIComponent(runId)}`} label="Run" />
          <p className="eyebrow">Run inputs</p>
        </>
      }
    />
  );
}
