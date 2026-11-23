import { Link } from "react-router";

import { PublicAPIError, type InUseDetails } from "../../api/error";
import { RUNTIME_CONFIGURATION_PATH } from "../../app/navigation";

interface ReferenceGroup {
  label: string;
  values: readonly string[];
  path?: (value: string) => string;
}

function referenceGroups(details: InUseDetails): ReferenceGroup[] {
  const bindings: ReferenceGroup = {
    label: "RuntimeConfig labels",
    values: "bindingLabels" in details ? details.bindingLabels : [],
    path: () => RUNTIME_CONFIGURATION_PATH,
  };
  const runs: ReferenceGroup = {
    label: "Runs",
    values: "runIds" in details ? details.runIds : [],
    path: (id) => `/runs/${encodeURIComponent(id)}`,
  };
  const audits: ReferenceGroup = {
    label: "Audits",
    values: "auditIds" in details ? details.auditIds : [],
  };
  switch (details.kind) {
    case "credential_in_use":
      return [runs, audits, bindings];
    case "runtime_credential_in_use":
      return [
        bindings,
        {
          label: "Projects",
          values: details.projectIds,
          path: (id) => `/projects/${encodeURIComponent(id)}`,
        },
        runs,
        audits,
        { label: "Allocations", values: details.allocationIds },
      ];
    case "runtime_label_in_use":
      return [{ label: "Runtime Agents", values: details.runtimeAgentIds }];
  }
}

export function InUseErrorDetails({ error }: { error: Error | null }) {
  const details = error instanceof PublicAPIError ? error.inUse : undefined;
  if (details === undefined) return null;
  const groups = referenceGroups(details).filter(
    (group) => group.values.length > 0,
  );
  if (groups.length === 0) return null;
  return (
    <div className="notice notice-warning in-use-details">
      <strong>Deletion is blocked by these references:</strong>
      {groups.map((group) => (
        <div key={group.label}>
          <h4>{group.label}</h4>
          <ul>
            {group.values.map((value) => (
              <li key={value}>
                {group.path === undefined ? (
                  <code>{value}</code>
                ) : (
                  <Link to={group.path(value)}>{value}</Link>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
