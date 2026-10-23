import { useMutation, useQuery } from "@tanstack/react-query";
import { useId } from "react";

import {
  canBrowseArchive,
  type ArtifactArchiveScope,
} from "../../api/artifact-archive";
import {
  canPreviewArtifact,
  HTTP_EXCHANGE_MEDIA_TYPE,
  MAXIMUM_PREVIEW_BYTES,
  PREVIEW_LIMIT_LABEL,
  type ArtifactMetadata,
} from "../../api/artifacts";
import { ErrorNotice } from "../../app/error-notice";
import { Button } from "../../ui";
import { ArchivePreviewPanel } from "./archive-preview";
import { LoadedArtifactPreview } from "./loaded-preview";
import "./materials.css";

/**
 * Bounded preview of one exact revision: a ZIP or Skill package opens the
 * archive browser (with a scope), any other previewable format loads up to
 * MAXIMUM_PREVIEW_BYTES of text and picks its renderer by media type.
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
  return (
    <TextArtifactPreviewPanel
      {...props}
      scope={archiveScope}
      titleAs={titleAs}
    />
  );
}

function TextArtifactPreviewPanel({
  metadata,
  loadPreview,
  loadOnMountKey,
  unavailableCopy,
  titleAs: Title,
  scope,
}: {
  metadata: ArtifactMetadata;
  loadPreview: () => Promise<string>;
  loadOnMountKey?: readonly unknown[] | undefined;
  unavailableCopy: string;
  titleAs: "h2" | "h3";
  scope?: ArtifactArchiveScope | undefined;
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
          <Button
            size="sm"
            disabled={!canPreview}
            pending={pending}
            pendingLabel="Loading…"
            onClick={requestPreview}
          >
            {data === undefined
              ? automatic
                ? "Retry preview"
                : "Load preview"
              : "Reload preview"}
          </Button>
        )}
      </div>
      {canPreview ? (
        <p className="ui-caption">
          {automatic
            ? `Text files up to ${PREVIEW_LIMIT_LABEL} are shown here.`
            : `Text files up to ${PREVIEW_LIMIT_LABEL}. The preview loads only when you ask for it.`}
        </p>
      ) : (
        <p className="ui-caption">
          {metadata.mediaType === HTTP_EXCHANGE_MEDIA_TYPE &&
          metadata.size > MAXIMUM_PREVIEW_BYTES
            ? `This HTTP exchange exceeds the ${PREVIEW_LIMIT_LABEL} preview limit. Download this revision to inspect the complete saved request and available response.`
            : unavailableCopy}
        </p>
      )}
      {error === null ? null : <ErrorNotice error={error} />}
      {data === undefined ? null : (
        <LoadedArtifactPreview
          key={metadata.artifact.revision}
          mediaType={metadata.mediaType}
          source={data}
          context={{ metadata, scope }}
        />
      )}
    </section>
  );
}
