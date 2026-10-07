import { useMutation, useQuery } from "@tanstack/react-query";
import { useId } from "react";

import {
  canBrowseArchive,
  type ArtifactArchiveScope,
} from "../../api/artifact-archive";
import { canPreviewArtifact, type ArtifactMetadata } from "../../api/artifacts";
import { ErrorNotice } from "../../app/error-notice";
import { ArchivePreviewPanel } from "./archive-preview";
import { LoadedArtifactPreview } from "./loaded-preview";
import "./materials.css";

/**
 * Bounded preview of one exact revision: a ZIP or Skill package opens the
 * archive browser (with a scope), any other previewable format loads up to
 * 256 KiB of text and picks its renderer by media type.
 */
export function ArtifactPreviewPanel({
  archiveScope,
  titleAs = "h3",
  ...props
}: {
  archiveScope?: ArtifactArchiveScope | undefined;
  metadata: ArtifactMetadata;
  loadPreview: () => Promise<string>;
  loadOnMountKey?: readonly unknown[] | undefined;
  unavailableCopy: string;
  /** Heading level of the section title. Default "h3". */
  titleAs?: "h2" | "h3" | undefined;
}) {
  if (archiveScope !== undefined && canBrowseArchive(props.metadata)) {
    return (
      <ArchivePreviewPanel
        key={JSON.stringify([archiveScope, props.metadata.artifact])}
        metadata={props.metadata}
        scope={archiveScope}
        loadOnMount={props.loadOnMountKey !== undefined}
        titleAs={titleAs}
      />
    );
  }
  return <TextArtifactPreviewPanel {...props} titleAs={titleAs} />;
}

function TextArtifactPreviewPanel({
  metadata,
  loadPreview,
  loadOnMountKey,
  unavailableCopy,
  titleAs: Title,
}: {
  metadata: ArtifactMetadata;
  loadPreview: () => Promise<string>;
  loadOnMountKey?: readonly unknown[] | undefined;
  unavailableCopy: string;
  titleAs: "h2" | "h3";
}) {
  const headingId = useId();
  const manualPreview = useMutation({ mutationFn: loadPreview });
  const canPreview = canPreviewArtifact(metadata);
  const automaticPreview = useQuery({
    queryKey: loadOnMountKey ?? ["artifact-preview", "manual"],
    queryFn: loadPreview,
    enabled: loadOnMountKey !== undefined && canPreview,
  });
  const automatic = loadOnMountKey !== undefined;
  const data = automatic ? automaticPreview.data : manualPreview.data;
  const error = automatic ? automaticPreview.error : manualPreview.error;
  const pending = automatic
    ? automaticPreview.isPending || automaticPreview.isFetching
    : manualPreview.isPending;

  function requestPreview(): void {
    if (automatic) {
      void automaticPreview.refetch();
      return;
    }
    manualPreview.mutate();
  }

  return (
    <section
      className="artifact-preview-panel materials-section"
      aria-labelledby={headingId}
    >
      <div className="materials-section-head">
        <Title id={headingId} className="materials-section-title">
          Preview
        </Title>
        {!canPreview && automatic ? null : (
          <button
            className="ui-btn"
            data-size="sm"
            type="button"
            disabled={!canPreview || pending}
            onClick={requestPreview}
          >
            {pending
              ? "Loading…"
              : data === undefined
                ? automatic
                  ? "Retry preview"
                  : "Load preview"
                : "Reload preview"}
          </button>
        )}
      </div>
      {canPreview ? (
        <p className="materials-quiet">
          {automatic
            ? "Text files up to 256 KiB are shown here."
            : "Text files up to 256 KiB. The preview loads only when you ask for it."}
        </p>
      ) : (
        <p className="materials-quiet materials-unavailable">
          {unavailableCopy}
        </p>
      )}
      {error === null ? null : <ErrorNotice error={error} />}
      {data === undefined ? null : (
        <LoadedArtifactPreview
          key={metadata.artifact.revision}
          mediaType={metadata.mediaType}
          source={data}
        />
      )}
    </section>
  );
}
