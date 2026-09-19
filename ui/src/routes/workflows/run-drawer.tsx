import { useId, useState, type ReactNode } from "react";
import type { WorkflowSummary } from "../../api/workflows";
import { Dialog } from "../../app/dialog";
import { workflowSelector } from "./presentation";
import "./overview.css";

export function WorkflowRunDrawer({
  workflow,
  projectId,
  onClose,
  children,
}: {
  workflow: WorkflowSummary;
  projectId?: string;
  onClose: () => void;
  children: (onSubmittingChange: (pending: boolean) => void) => ReactNode;
}) {
  const heading = useId();
  const description = useId();
  const [submitting, setSubmitting] = useState(false);
  return (
    <Dialog
      className="workflow-run-drawer"
      backdropClassName="workflow-run-backdrop"
      labelledBy={heading}
      describedBy={description}
      onRequestClose={() => {
        if (!submitting) onClose();
      }}
    >
      <header className="workflow-drawer-heading">
        <div>
          <p className="eyebrow">
            {projectId ? "Project Run" : "Standalone Run"}
          </p>
          <h2 id={heading}>Configure Run</h2>
          <code>{workflowSelector(workflow)}</code>
        </div>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close Run setup"
          disabled={submitting}
          onClick={onClose}
        >
          ×
        </button>
      </header>
      <p className="workflow-drawer-description" id={description}>
        {projectId
          ? "Inputs come from this project (ProjectScope)."
          : "Inputs come from your library (UserScope)."}{" "}
        Your draft is kept in this tab when you close this panel.
      </p>
      <div id="workflow-run-setup" className="workflow-drawer-body">
        {children(setSubmitting)}
      </div>
    </Dialog>
  );
}
