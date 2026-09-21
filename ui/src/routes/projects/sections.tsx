import { useOutletContext } from "react-router";
import type { Project } from "../../api/projects";
import { ProjectArtifactRegion } from "./artifact-region";
import { ProjectAuditWorkspace } from "./audits/list";
import { ProjectHTTPTargetEditor } from "./http-target-editor";
import { ProjectMetadataEditor } from "./metadata-editor";
import { ProjectOverview } from "./overview";
import { ProjectRunsRegion } from "./runs-region";
import { ProjectWorkflowRecommendations } from "./workflow-recommendations";

export function ProjectSectionRoute({
  section,
}: {
  section:
    "overview" | "artifacts" | "workflows" | "runs" | "audits" | "settings";
}) {
  const project = useOutletContext<Project>();
  switch (section) {
    case "overview":
      return <ProjectOverview key={project.projectId} project={project} />;
    case "artifacts":
      return (
        <ProjectArtifactRegion
          key={project.projectId}
          projectId={project.projectId}
          detailRoot="/projects"
          compact
        />
      );
    case "workflows":
      return (
        <ProjectWorkflowRecommendations
          key={project.projectId}
          projectId={project.projectId}
          drawer
        />
      );
    case "runs":
      return (
        <ProjectRunsRegion
          key={project.projectId}
          projectId={project.projectId}
          evaluation={false}
        />
      );
    case "audits":
      return (
        <section className="project-audits-section">
          <ProjectAuditWorkspace
            key={project.projectId}
            projectId={project.projectId}
            projectName={project.name}
          />
        </section>
      );
    case "settings":
      return (
        <section className="project-settings-section">
          <section className="panel">
            <h3>Project details</h3>
            <ProjectMetadataEditor key={project.projectId} project={project} />
          </section>
          <section className="panel">
            <ProjectHTTPTargetEditor
              key={project.projectId}
              project={project}
            />
          </section>
        </section>
      );
  }
}
