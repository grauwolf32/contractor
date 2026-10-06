import { useSearchParams } from "react-router";

import { PublicAPIError } from "../../api/error";
import { PROJECT_ID_PATTERN } from "../../api/projects";
import { useDocumentTitle } from "../../app/document-title";
import { useProject } from "./start/data";
import { ProjectPicker } from "./start/project-picker";
import { StartCheckPage } from "./start/start-page";

import "./start/start.css";

/**
 * Start a check (/checks/new). `project` chooses the project (without it the
 * page asks for one), `objective` prefills the objective and `type` (a check
 * type name) preselects the check type (docs/design/ui/v3b-build-contract.md
 * §5).
 */
export function StartCheckRoute() {
  const [params] = useSearchParams();
  const projectId = params.get("project") ?? "";
  const objective = params.get("objective") ?? "";
  useDocumentTitle("Start a check");
  if (projectId === "") return <ProjectPicker />;
  if (!PROJECT_ID_PATTERN.test(projectId))
    return (
      <ProjectPicker notice="The link names a project that does not exist. Choose a project." />
    );
  return (
    <ProjectGate
      // A new project or objective in the URL starts a new form.
      key={`${projectId}\n${objective}`}
      projectId={projectId}
      objective={objective}
    />
  );
}

function ProjectGate({
  projectId,
  objective,
}: {
  projectId: string;
  objective: string;
}) {
  const project = useProject(projectId);
  if (
    project.data === undefined &&
    project.error instanceof PublicAPIError &&
    project.error.status === 404
  )
    return (
      <ProjectPicker notice="This project was not found. It may have been deleted. Choose a project." />
    );
  return <StartCheckPage projectId={projectId} initialObjective={objective} />;
}
