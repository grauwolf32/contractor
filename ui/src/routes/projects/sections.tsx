import { useEffect, useId, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router";

import type { Project } from "../../api/projects";
import { RUNTIME_CONFIGURATION_PATH } from "../../app/navigation";
import { useSession } from "../../auth/session";
import { TechnicalDetails } from "../../ui";
import { ProjectArtifactRegion } from "./artifact-region";
import { ProjectAuditWorkspace } from "./audits/list";
import {
  ProjectHTTPTargetEditor,
  type TargetReveal,
} from "./http-target-editor";
import { ProjectFacts, ProjectMetadataEditor } from "./metadata-editor";
import { ProjectOverview } from "./overview";
import {
  LIVE_TARGET_ANCHOR,
  opensTargetSheet,
  withoutTargetSheet,
} from "./project-sections";
import { ProjectRunsRegion } from "./runs-region";
import { ProjectWorkflowRecommendations } from "./workflow-recommendations";
import { useProjectWorkspace } from "./workspace-context";

import "./projects.css";

/** Settings: details, live target, technical details and deletion. */
function ProjectSettings({
  project,
  operator,
  requestDeletion,
}: {
  project: Project;
  operator: boolean;
  requestDeletion: () => void;
}) {
  const details = useId();
  const target = useId();
  const danger = useId();
  const location = useLocation();
  const navigate = useNavigate();
  const targetSection = useRef<HTMLElement>(null);
  const atTarget = location.hash === `#${LIVE_TARGET_ANCHOR}`;
  const openSheet = atTarget && opensTargetSheet(location.state);
  // What Settings was opened for, from its first render: a link to
  // #live-target lands on the Live target; "Add a live target" on the
  // overview also opens its sheet.
  const [reveal] = useState<TargetReveal | undefined>(() =>
    openSheet ? "open" : atTarget ? "focus" : undefined,
  );
  useEffect(() => {
    if (reveal !== undefined)
      targetSection.current?.scrollIntoView?.({ block: "start" });
  }, [reveal]);
  useEffect(() => {
    if (!openSheet) return;
    // The sheet opens once: Back and reload show the section without it.
    void navigate(
      {
        pathname: location.pathname,
        search: location.search,
        hash: location.hash,
      },
      {
        replace: true,
        preventScrollReset: true,
        state: withoutTargetSheet(location.state),
      },
    );
  }, [location, navigate, openSheet]);
  return (
    <div className="projects-settings">
      <section className="projects-panel" aria-labelledby={details}>
        <div className="projects-section-heading">
          <h3 id={details}>Project details</h3>
        </div>
        <ProjectMetadataEditor key={project.projectId} project={project} />
      </section>
      <section
        ref={targetSection}
        className="projects-panel"
        id={LIVE_TARGET_ANCHOR}
        aria-labelledby={target}
      >
        <div className="projects-section-heading">
          <h3 id={target}>Live target</h3>
        </div>
        <p className="projects-caption">
          The running application that active checks of this project may send
          requests to, and the authorization they use.
        </p>
        <ProjectHTTPTargetEditor
          key={project.projectId}
          project={project}
          reveal={reveal}
        />
      </section>
      <TechnicalDetails description="Identifiers, revision and where execution defaults live.">
        <ProjectFacts project={project} showId={false} />
        <p className="projects-caption">
          These settings apply to this project only. Server-wide execution
          defaults live under Operations.
          {operator ? (
            <>
              {" "}
              <Link to={RUNTIME_CONFIGURATION_PATH}>Runtime configuration</Link>
            </>
          ) : null}
        </p>
      </TechnicalDetails>
      <section
        className="projects-panel projects-danger"
        aria-labelledby={danger}
      >
        <div className="projects-section-heading">
          <h3 id={danger}>Delete this project</h3>
        </div>
        <p className="projects-caption">
          Deletion cancels every active Run and permanently removes the
          project&apos;s Runs, history and materials. You confirm by typing the
          project&apos;s name.
        </p>
        <div className="projects-form-actions">
          <button
            type="button"
            className="ui-btn"
            data-variant="danger"
            data-size="sm"
            onClick={requestDeletion}
          >
            Delete this project
          </button>
        </div>
      </section>
    </div>
  );
}

export function ProjectSectionRoute({
  section,
}: {
  section:
    "overview" | "artifacts" | "workflows" | "runs" | "audits" | "settings";
}) {
  const { project, requestDeletion } = useProjectWorkspace();
  const { session } = useSession();
  const operator =
    session?.principal.capabilities.includes("operations") === true;
  switch (section) {
    case "overview":
      return <ProjectOverview key={project.projectId} project={project} />;
    case "artifacts":
      return (
        <ProjectArtifactRegion
          key={project.projectId}
          projectId={project.projectId}
        />
      );
    case "workflows":
      return (
        <ProjectWorkflowRecommendations
          key={project.projectId}
          projectId={project.projectId}
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
        <ProjectSettings
          key={project.projectId}
          project={project}
          operator={operator}
          requestDeletion={requestDeletion}
        />
      );
  }
}
