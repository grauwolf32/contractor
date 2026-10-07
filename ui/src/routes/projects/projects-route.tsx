import { useCallback, useState } from "react";
import {
  Link,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router";

import { useDocumentTitle } from "../../app/document-title";
import { DetailPane, EmptyState, PaneLayout } from "../../ui";
import { NewProjectDialog } from "./new-project-dialog";
import { useNewProjectKeys } from "./new-project-keys";
import { ProjectDraftsContext } from "./project-drafts";
import { ProjectListPane } from "./project-list-pane";
import { projectPath, projectSectionOf } from "./project-sections";
import { ProjectWorkspace } from "./project-workspace";

import "./projects.css";

/** The detail pane on /projects, before a project is chosen. */
function ProjectsStart({ onNewProject }: { onNewProject: () => void }) {
  useDocumentTitle("Projects");
  return (
    <DetailPane>
      <EmptyState
        title="Choose a project"
        action={
          <div className="projects-start-actions">
            <button
              type="button"
              className="ui-btn"
              data-variant="primary"
              data-size="sm"
              onClick={onNewProject}
            >
              New project
            </button>
            <Link className="ui-btn" data-size="sm" to="/checks/new">
              Start a check
            </Link>
          </div>
        }
      >
        <p>
          A project keeps your materials (source code, API specs, documents),
          the checks you run on them and their results together. Choose one from
          the list, or create one and add its materials.
        </p>
      </EmptyState>
    </DetailPane>
  );
}

/**
 * /projects and /projects/:projectId (with its sections) in one component,
 * so choosing a project keeps the list pane mounted. The selected project is
 * the path; `?new=1` on /projects opens the New project dialog.
 */
export function ProjectsRoute() {
  const { projectId } = useParams();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [creating, setCreating] = useState(false);
  const createKeys = useNewProjectKeys();
  const [drafts, setDrafts] = useState<readonly string[]>([]);
  const reportDraft = useCallback((draft: string, unsaved: boolean) => {
    setDrafts((current) =>
      current.includes(draft) === unsaved
        ? current
        : unsaved
          ? [...current, draft]
          : current.filter((item) => item !== draft),
    );
  }, []);
  const requested = projectId === undefined && searchParams.get("new") === "1";
  const section =
    projectId === undefined ? "" : projectSectionOf(pathname, projectId);
  // J / K move through projects only where the section has no list of its
  // own to move through, and never away from an unsaved draft (project
  // details being edited, an objective typed into the composer).
  const keyboard =
    drafts.length === 0 &&
    (projectId === undefined || section === "" || section === "settings");

  function closeCreate() {
    setCreating(false);
    if (requested) {
      const next = new URLSearchParams(searchParams);
      next.delete("new");
      setSearchParams(next, { replace: true });
    }
  }

  return (
    <ProjectDraftsContext value={reportDraft}>
      <PaneLayout
        listLabel="Projects"
        detailLabel={projectId === undefined ? "Getting started" : "Project"}
        showDetail={projectId !== undefined}
        backLink={{ to: "/projects", label: "Back to projects" }}
        list={
          <ProjectListPane
            selectedId={projectId}
            keyboard={keyboard}
            onNewProject={() => setCreating(true)}
          />
        }
        detail={
          projectId === undefined ? (
            <ProjectsStart onNewProject={() => setCreating(true)} />
          ) : (
            <ProjectWorkspace key={projectId} projectId={projectId} />
          )
        }
      />
      {creating || requested ? (
        <NewProjectDialog
          kind="project"
          keys={createKeys}
          wording={{
            title: "New project",
            closeLabel: "Close new project form",
            submitLabel: "Create project",
          }}
          onClose={closeCreate}
          onCreated={(project) => {
            setCreating(false);
            // Opened from ?new=1: Back must not reopen the dialog.
            void navigate(projectPath(project.projectId), {
              replace: requested,
            });
          }}
        />
      ) : null}
    </ProjectDraftsContext>
  );
}
