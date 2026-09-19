import { useMutation, useQuery } from "@tanstack/react-query";
import { canPreviewArtifact, type ArtifactMetadata } from "../../api/artifacts";
import {
  canBrowseArchive,
  type ArtifactArchiveScope,
} from "../../api/artifact-archive";
import { ArchivePreviewPanel } from "./archive-preview";
import { ErrorNotice } from "./common";
import { LoadedArtifactPreview } from "./loaded-preview";

export function ArtifactPreviewPanel({
  archiveScope,
  ...props
}: {
  archiveScope?: ArtifactArchiveScope;
  metadata: ArtifactMetadata;
  loadPreview: () => Promise<string>;
  loadOnMountKey?: readonly unknown[];
  unavailableCopy: string;
}) {
  if (archiveScope !== undefined && canBrowseArchive(props.metadata)) {
    return (
      <ArchivePreviewPanel
        key={JSON.stringify([archiveScope, props.metadata.artifact])}
        metadata={props.metadata}
        scope={archiveScope}
        loadOnMount={props.loadOnMountKey !== undefined}
      />
    );
  }
  return <TextArtifactPreviewPanel {...props} />;
}

function TextArtifactPreviewPanel({
  metadata,
  loadPreview,
  loadOnMountKey,
  unavailableCopy,
}: {
  metadata: ArtifactMetadata;
  loadPreview: () => Promise<string>;
  loadOnMountKey?: readonly unknown[];
  unavailableCopy: string;
}) {
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
    <div className="panel artifact-preview-panel">
      <div className="section-heading">
        <div>
          <h3>Preview</h3>
        </div>
        {!canPreview && automatic ? null : (
          <button
            className={data === undefined ? undefined : "secondary-button"}
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
        <p className="muted-copy">Documents up to 256 KiB can be previewed.</p>
      ) : (
        <div className="compact-empty">{unavailableCopy}</div>
      )}
      {error === null ? null : <ErrorNotice error={error} />}
      {data === undefined ? null : (
        <LoadedArtifactPreview
          key={metadata.artifact.revision}
          mediaType={metadata.mediaType}
          source={data}
        />
      )}
    </div>
  );
}
