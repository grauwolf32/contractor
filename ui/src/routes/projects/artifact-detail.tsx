import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
} from "../../api/artifacts";
import { PROJECT_ID_PATTERN } from "../../api/projects";
import { ReturnLink } from "../../app/context-navigation";
import { ErrorNotice } from "../../app/error-notice";
import { ArtifactDetailView } from "../artifacts/artifact-detail-view";

function ProjectArtifactDetailRouteView({
  detailRoot,
}: {
  detailRoot: "/projects" | "/evals";
}) {
  const { projectId = "", namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const validRoute =
    PROJECT_ID_PATTERN.test(projectId) &&
    ARTIFACT_NAME_PATTERN.test(namespace) &&
    ARTIFACT_NAME_PATTERN.test(name) &&
    (revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision));
  if (!validRoute) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Project Artifact route is invalid")} />
        <Link to={detailRoot}>
          Return to {detailRoot === "/evals" ? "Evals" : "Projects"}
        </Link>
      </section>
    );
  }

  return (
    <ArtifactDetailView
      key={`${projectId}/${namespace}/${name}`}
      scope={{ kind: "project", id: projectId }}
      namespace={namespace}
      name={name}
      revision={revision}
      className="project-artifact-page"
      heading={
        <>
          <ReturnLink
            to={`${detailRoot}/${encodeURIComponent(projectId)}${detailRoot === "/projects" ? "/artifacts" : "#project-artifacts"}`}
            label={
              detailRoot === "/evals" ? "Eval Artifacts" : "Project Artifacts"
            }
          />
          <p className="eyebrow">Project artifact</p>
        </>
      }
    />
  );
}

export function ProjectArtifactDetailRoute() {
  return <ProjectArtifactDetailRouteView detailRoot="/projects" />;
}

export function EvaluationArtifactDetailRoute() {
  return <ProjectArtifactDetailRouteView detailRoot="/evals" />;
}
