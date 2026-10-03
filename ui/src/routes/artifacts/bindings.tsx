import type { ReactNode } from "react";

import type { ArtifactMetadata } from "../../api/artifacts";
import { ContextLink } from "../../app/context-navigation";
import { formatBytes } from "../../app/format";
import { RecordedTime } from "../../app/recorded-time";

/**
 * Table of Artifact bindings. `showLocked` adds the Run-only column that
 * tells whether a binding is frozen for the Run.
 */
export function ArtifactBindingsTable({
  items,
  detailPath,
  returnLabel,
  returnHash,
  showLocked = false,
}: {
  items: readonly ArtifactMetadata[];
  detailPath: (item: ArtifactMetadata) => string;
  returnLabel: string;
  returnHash?: string;
  showLocked?: boolean;
}) {
  return (
    <div className="table-scroll">
      <table className="responsive-table">
        <thead>
          <tr>
            <th>Binding</th>
            <th>Current revision</th>
            <th>Media type</th>
            <th>Size</th>
            {showLocked ? <th>Locked</th> : null}
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={`${item.artifact.namespace}/${item.artifact.name}`}>
              <td data-label="Binding">
                <ContextLink
                  returnLabel={returnLabel}
                  {...(returnHash === undefined ? {} : { returnHash })}
                  to={detailPath(item)}
                >
                  {item.artifact.namespace}/{item.artifact.name}
                </ContextLink>
              </td>
              <td data-label="Current revision">
                <code>{item.artifact.revision}</code>
              </td>
              <td data-label="Media type">{item.mediaType}</td>
              <td data-label="Size">{formatBytes(item.size)}</td>
              {showLocked ? (
                <td data-label="Locked">{item.frozen ? "yes" : "no"}</td>
              ) : null}
              <td data-label="Created">
                <RecordedTime value={item.createdAt} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Success notice for a stored revision with a link to its detail page. */
export function ArtifactStoredNotice({
  title,
  artifact,
  to,
  returnLabel,
  children,
}: {
  title: string;
  artifact: { namespace: string; name: string; revision: string };
  to: string;
  returnLabel: string;
  children?: ReactNode;
}) {
  return (
    <div className="notice notice-success" role="status">
      <strong>{title}</strong>
      <ContextLink returnLabel={returnLabel} to={to}>
        Open {artifact.namespace}/{artifact.name}@{artifact.revision}
      </ContextLink>
      {children}
    </div>
  );
}
