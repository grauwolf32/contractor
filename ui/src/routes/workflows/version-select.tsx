import type { WorkflowSummary } from "../../api/workflows";
import "./cards.css";

/** The exact version picker of a Workflow card and the Workflow page. */
export function WorkflowVersionSelect({
  workflow,
  versions,
  onChange,
}: {
  workflow: WorkflowSummary;
  versions: readonly WorkflowSummary[];
  onChange: (version: string) => void;
}) {
  return (
    <label className="workflow-card-select">
      <span>Version</span>
      <select
        aria-label={`Version of ${workflow.ref.name}`}
        value={workflow.ref.version}
        onChange={(event) => onChange(event.target.value)}
      >
        {versions.map((version) => (
          <option key={version.ref.version} value={version.ref.version}>
            {version.ref.version}
          </option>
        ))}
      </select>
    </label>
  );
}

/**
 * Whether a version is the latest of its family ("Latest", highlighted) or
 * an earlier one. Only meaningful when every version is a dotted number
 * (hasNumericVersions).
 */
export function WorkflowVersionTag({ latest }: { latest: boolean }) {
  return (
    <span
      className="ui-badge workflow-card-tag"
      data-tone={latest ? "accent" : undefined}
    >
      {latest ? "Latest" : "Earlier version"}
    </span>
  );
}
