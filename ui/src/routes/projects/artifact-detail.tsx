import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
} from "../../api/artifacts";
import { PROJECT_ID_PATTERN } from "../../api/projects";
import { ReturnLink } from "../../app/context-navigation";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { ArtifactDetailView } from "../artifacts/artifact-detail-view";
import "../artifacts/materials.css";

function ProjectArtifactDetailRouteView({
  detailRoot,
}: {
  detailRoot: "/projects" | "/evals";
}) {
  const { projectId = "", namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const evaluation = detailRoot === "/evals";
  useDocumentTitle(
    name
      ? `${namespace}/${name} · ${evaluation ? "Eval workspace" : "Materials"}`
      : "Materials",
  );
  const validRoute =
    PROJECT_ID_PATTERN.test(projectId) &&
    ARTIFACT_NAME_PATTERN.test(namespace) &&
    ARTIFACT_NAME_PATTERN.test(name) &&
    (revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision));
  if (!validRoute) {
    return (
      <section className="route-page materials-page materials-invalid">
        <ErrorNotice error={new Error("This material link is not valid.")} />
        <Link to={detailRoot}>
          Return to {evaluation ? "Evals" : "Projects"}
        </Link>
      </section>
    );
  }

  const project = encodeURIComponent(projectId);
  return (
    <ArtifactDetailView
      key={`${projectId}/${namespace}/${name}`}
      scope={{ kind: "project", id: projectId }}
      namespace={namespace}
      name={name}
      revision={revision}
      className="project-artifact-page"
      variant="material"
      heading={
        <ReturnLink
          to={
            evaluation
              ? `/evals/${project}#project-artifacts`
              : `/projects/${project}/artifacts`
          }
          label={evaluation ? "Eval workspace" : "Materials"}
        />
      }
    />
  );
}

/** Project → Materials → one material (`?revision=` pins a version). */
export function ProjectArtifactDetailRoute() {
  return <ProjectArtifactDetailRouteView detailRoot="/projects" />;
}

/** The same page for a material of a legacy evaluation workspace. */
export function EvaluationArtifactDetailRoute() {
  return <ProjectArtifactDetailRouteView detailRoot="/evals" />;
}
