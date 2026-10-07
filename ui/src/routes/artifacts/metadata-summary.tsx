import { useEffect, useRef, useState } from "react";

import type { ArtifactArchiveScope } from "../../api/artifact-archive";
import type { ArtifactMetadata } from "../../api/artifacts";
import { formatBytes, formatTimestamp } from "../../app/format";
import { RecordedTime } from "../../app/recorded-time";
import { IdChip, StatusGlyph, TechnicalDetails } from "../../ui";
import { ArtifactHistory } from "./history";
import { MaterialIcon } from "./icons";
import { formatLabel } from "./kinds";
import "./materials.css";

/** Id of the technical details that hold versions and lineage. */
const HISTORY_ID = "artifact-history";

/**
 * Whether the shown revision is the current one. Nothing is claimed before
 * the metadata has loaded.
 */
export function ArtifactRevisionLede({
  metadata,
}: {
  metadata: ArtifactMetadata | undefined;
}) {
  if (metadata === undefined) return null;
  return (
    <p
      className="lede materials-revision"
      data-current={metadata.current ? "" : undefined}
    >
      <StatusGlyph tone={metadata.current ? "done" : "idle"} size={14} />
      {metadata.current ? "Current revision" : "Historical revision"}
    </p>
  );
}

/** Format, size, creation, lock and the Git source of one revision. */
export function ArtifactFacts({ metadata }: { metadata: ArtifactMetadata }) {
  const git = metadata.gitSource;
  return (
    <dl className="materials-facts">
      <div>
        <dt>Format</dt>
        <dd>{formatLabel(metadata.mediaType)}</dd>
      </div>
      <div>
        <dt>Size</dt>
        <dd>{formatBytes(metadata.size)}</dd>
      </div>
      <div>
        <dt>Created</dt>
        <dd>
          <time dateTime={metadata.createdAt}>
            {formatTimestamp(metadata.createdAt)}
          </time>
        </dd>
      </div>
      <div>
        <dt>Locked</dt>
        <dd>{metadata.frozen ? "yes" : "no"}</dd>
      </div>
      {git === undefined ? null : (
        <>
          <div className="materials-fact-wide">
            <dt>Git repository</dt>
            <dd className="materials-mono materials-wrap">
              {git.repositoryUrl}
            </dd>
          </div>
          <div>
            <dt>Branch or tag</dt>
            <dd>{git.requestedRef ?? "Default branch"}</dd>
          </div>
          <div>
            <dt>Git commit</dt>
            <dd>
              <IdChip value={git.resolvedCommit} label="Git commit" />
            </dd>
          </div>
          <div>
            <dt>Imported</dt>
            <dd>
              <RecordedTime value={git.importedAt} />
            </dd>
          </div>
        </>
      )}
    </dl>
  );
}

/**
 * Revision, media type, versions and lineage behind Technical details. The
 * history loads the first time the details open.
 */
export function ArtifactTechnicalDetails({
  scope,
  metadata,
}: {
  scope: ArtifactArchiveScope;
  metadata: ArtifactMetadata;
}) {
  const wrapper = useRef<HTMLDivElement>(null);
  const [opened, setOpened] = useState(false);
  useEffect(() => {
    const node = wrapper.current;
    if (node === null) return undefined;
    // `toggle` does not bubble; a capturing listener still sees it.
    const onToggle = (event: Event) => {
      if (event.target instanceof HTMLDetailsElement && event.target.open)
        setOpened(true);
    };
    node.addEventListener("toggle", onToggle, true);
    return () => node.removeEventListener("toggle", onToggle, true);
  }, []);
  return (
    <div ref={wrapper} id={HISTORY_ID} className="materials-tech">
      <TechnicalDetails description="Revision, media type, versions and lineage.">
        <dl className="materials-facts materials-tech-facts">
          <div className="materials-fact-wide">
            <dt>Revision</dt>
            <dd>
              <code>{metadata.artifact.revision}</code>
            </dd>
          </div>
          <div>
            <dt>Media type</dt>
            <dd>
              <code>{metadata.mediaType}</code>
            </dd>
          </div>
        </dl>
        {opened ? (
          <ArtifactHistory
            key={metadata.artifact.revision}
            scope={scope}
            metadata={metadata}
          />
        ) : null}
      </TechnicalDetails>
    </div>
  );
}

/** Opens Technical details at the version history and moves focus there. */
export function ArtifactVersionsButton() {
  return (
    <button
      type="button"
      className="ui-btn"
      data-size="sm"
      data-variant="ghost"
      onClick={() => {
        const node = document.getElementById(HISTORY_ID);
        const details = node?.querySelector("details");
        if (node === null || details === null || details === undefined) return;
        details.open = true;
        const reduceMotion =
          typeof window.matchMedia === "function" &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        node.scrollIntoView?.({
          block: "start",
          behavior: reduceMotion ? "auto" : "smooth",
        });
        details.querySelector("summary")?.focus({ preventScroll: true });
      }}
    >
      <MaterialIcon name="history" />
      Versions
    </button>
  );
}
