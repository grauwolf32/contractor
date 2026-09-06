import { useEffect, useId, useState } from "react";

import type { ArtifactWriteResponse } from "../../api/artifacts";
import { Dialog } from "../../app/dialog";
import { ArtifactWriteForm } from "../artifacts/common";
import { ProjectArtifactWriteForm } from "../projects/common";

function fixedMediaType(mediaTypes: readonly string[]): string | undefined {
  return mediaTypes.length === 1 && mediaTypes[0] !== "*/*"
    ? mediaTypes[0]
    : undefined;
}

export function RunInputUploadDialog({
  projectId,
  slotName,
  mediaTypes,
  onClose,
  onUploaded,
}: {
  projectId?: string;
  slotName: string;
  mediaTypes: readonly string[];
  onClose: () => void;
  onUploaded: (result: ArtifactWriteResponse) => void;
}) {
  const heading = useId();
  const description = useId();
  const [controller] = useState(() => new AbortController());
  const [pending, setPending] = useState(false);
  const mediaType = fixedMediaType(mediaTypes);

  useEffect(() => () => controller.abort(), [controller]);

  function close(): void {
    controller.abort();
    onClose();
  }

  function uploaded(result: ArtifactWriteResponse): void {
    if (controller.signal.aborted) return;
    onUploaded(result);
    onClose();
  }

  const form =
    projectId === undefined ? (
      <ArtifactWriteForm
        fixedNamespace="inputs"
        {...(mediaType === undefined ? {} : { fixedMediaType: mediaType })}
        acceptedMediaTypes={mediaTypes}
        signal={controller.signal}
        submitLabel="Upload and select exact revision"
        onPendingChange={setPending}
        onWritten={uploaded}
      />
    ) : (
      <ProjectArtifactWriteForm
        projectId={projectId}
        fixedNamespace="artifacts"
        {...(mediaType === undefined ? {} : { fixedMediaType: mediaType })}
        acceptedMediaTypes={mediaTypes}
        signal={controller.signal}
        submitLabel="Upload and select exact revision"
        onPendingChange={setPending}
        onWritten={uploaded}
      />
    );

  return (
    <Dialog
      className="project-dialog panel run-input-upload-dialog"
      labelledBy={heading}
      describedBy={description}
      onRequestClose={close}
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">
            {projectId === undefined ? "UserScope" : "ProjectScope"} input
          </p>
          <h2 id={heading}>Upload local file for {slotName}</h2>
        </div>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close local file upload"
          onClick={close}
        >
          ×
        </button>
      </div>
      <p id={description} className="muted-copy">
        A confirmed upload selects only the exact returned revision in this
        input slot. Closing{pending ? " now" : ""} cancels the client wait;
        because the Server may already have committed the Artifact, refresh the
        library before retrying an ambiguous write.
      </p>
      {form}
    </Dialog>
  );
}
