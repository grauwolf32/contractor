import { useId, useRef, useState, type ReactNode } from "react";

import type { WorkflowSummary } from "../../api/workflows";
import { Dialog } from "../../app/dialog";
import { Icon } from "../../app/icon";
import { IdChip } from "../../ui";
import { workflowSelector } from "./presentation";
import "./overview.css";

/**
 * The "Configure Run" sheet: the exact Workflow version (UUS:62-63) above
 * the Run form, which keeps its draft in this tab when the sheet closes.
 */
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
  const close = useRef<HTMLButtonElement>(null);
  const [submitting, setSubmitting] = useState(false);
  const selector = workflowSelector(workflow);
  return (
    <Dialog
      className="workflow-run-drawer"
      backdropClassName="workflow-run-backdrop"
      labelledBy={heading}
      describedBy={description}
      initialFocusRef={close}
      onRequestClose={() => {
        if (!submitting) onClose();
      }}
    >
      <header className="workflow-drawer-heading">
        <div className="workflow-drawer-titles">
          <p className="workflow-drawer-kind">
            {projectId ? "Project Run" : "Standalone Run"}
          </p>
          <h2 id={heading}>Configure Run</h2>
          <code className="workflow-drawer-selector">
            <IdChip
              value={selector}
              display={selector}
              label="workflow version"
            />
          </code>
        </div>
        <button
          ref={close}
          className="ui-btn workflow-drawer-close"
          data-variant="ghost"
          type="button"
          aria-label="Close Run setup"
          disabled={submitting}
          onClick={onClose}
        >
          <Icon name="close" />
        </button>
      </header>
      <p className="workflow-drawer-description" id={description}>
        {projectId
          ? "Inputs come from this Project."
          : "Inputs come from your library."}{" "}
        Your draft is kept in this tab when you close this panel.
      </p>
      <div id="workflow-run-setup" className="workflow-drawer-body">
        {children(setSubmitting)}
      </div>
    </Dialog>
  );
}
