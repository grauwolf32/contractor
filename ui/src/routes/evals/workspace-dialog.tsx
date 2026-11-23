import { usePublicAPI } from "../../api/context";
import { createProject, type Project } from "../../api/projects";
import { NewProjectDialog } from "../projects/new-project-dialog";
import { useEvalOwner } from "./queries";
import { recoverableMutation } from "./recovery";

/** One workspace form for the collection and experiment setup. */
export function EvaluationWorkspaceDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (project: Project) => void;
}) {
  const api = usePublicAPI();
  const owner = useEvalOwner();
  return (
    <NewProjectDialog
      kind="evaluation"
      wording={{
        title: "New evaluation workspace",
        closeLabel: "Close workspace form",
        submitLabel: "Create workspace",
      }}
      createRequest={(request) =>
        recoverableMutation(owner, "workspace", request, (idempotencyKey) =>
          createProject(api, { request, idempotencyKey }),
        )
      }
      onClose={onClose}
      onCreated={onCreated}
    />
  );
}
