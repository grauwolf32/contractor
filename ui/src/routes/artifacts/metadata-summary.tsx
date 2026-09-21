import { type ReactNode, useState } from "react";
import type { ArtifactMetadata } from "../../api/artifacts";
import { workflowFormats } from "../workflows/formats";
import { formatBytes, formatTimestamp } from "./common";
import { GitSourceDetails } from "./git-import-dialog";
import "./reader.css";

export function ArtifactMetadataSummary({
  metadata,
}: {
  metadata: ArtifactMetadata;
}) {
  const revision = metadata.artifact.revision;
  const short =
    revision.length > 22
      ? `${revision.slice(0, 12)}…${revision.slice(-6)}`
      : revision;
  return (
    <div className="artifact-metadata-summary">
      <span>{workflowFormats[metadata.mediaType] ?? metadata.mediaType}</span>
      <span>{formatBytes(metadata.size)}</span>
      <details>
        <summary>
          {metadata.current ? "Current" : "Historical"} ·{" "}
          <code title={revision}>{short}</code> · Details
        </summary>
        <dl className="metadata-grid">
          <div>
            <dt>Revision</dt>
            <dd>
              <code>{revision}</code>
            </dd>
          </div>
          <div>
            <dt>Media type</dt>
            <dd>{metadata.mediaType}</dd>
          </div>
          <div>
            <dt>Created</dt>
            <dd>{formatTimestamp(metadata.createdAt)}</dd>
          </div>
          <div>
            <dt>Locked</dt>
            <dd>{metadata.frozen ? "yes" : "no"}</dd>
          </div>
        </dl>
        <GitSourceDetails source={metadata.gitSource} />
      </details>
    </div>
  );
}

export function ArtifactHistoryDisclosure({
  children,
}: {
  children: ReactNode;
}) {
  const [opened, setOpened] = useState(false);
  return (
    <details
      id="artifact-history"
      className="artifact-history-disclosure"
      onToggle={(event) => setOpened(event.currentTarget.open)}
    >
      <summary>Versions and lineage</summary>
      {opened ? children : null}
    </details>
  );
}

export function ArtifactHistoryButton() {
  return (
    <button
      className="secondary-button"
      type="button"
      onClick={() => {
        const history = document.getElementById("artifact-history");
        if (history instanceof HTMLDetailsElement) {
          history.open = true;
          history.scrollIntoView?.({ block: "start", behavior: "smooth" });
          history.querySelector("summary")?.focus();
        }
      }}
    >
      Versions
    </button>
  );
}
