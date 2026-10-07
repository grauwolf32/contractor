import { useOutletContext } from "react-router";

import type { Project } from "../../api/projects";

/** What the selected project's page hands its sections through the Outlet. */
export interface ProjectWorkspaceContext {
  project: Project;
  /** Opens the typed-name deletion dialog of the project header. */
  requestDeletion: () => void;
}

export function useProjectWorkspace(): ProjectWorkspaceContext {
  return useOutletContext<ProjectWorkspaceContext>();
}
