import { useState } from "react";

import type { ArtifactWriteResponse } from "../../api/artifacts";
import { ArtifactWriteForm } from "../artifacts/common";
import { UploadDialog } from "../artifacts/upload-dialog";

function fixedMediaType(mediaTypes: readonly string[]): string | undefined {
  return mediaTypes.length === 1 && mediaTypes[0] !== "*/*"
    ? mediaTypes[0]
    : undefined;
}

/**
 * Uploads a local file for one input slot of the Run form and selects the
 * returned revision. Closing the dialog aborts the upload (UploadDialog); a
 * write that answers after that is not selected.
 */
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
  const [pending, setPending] = useState(false);
  const mediaType = fixedMediaType(mediaTypes);
  // The dialog title names the slot and labels the form, so the form shows
  // no heading of its own (and no internal "Artifact" noun).
  return (
    <UploadDialog
      eyebrow={`${projectId === undefined ? "Library" : "Project"} input`}
      title={`Upload local file for ${slotName}`}
      closeLabel="Close local file upload"
      note={`A confirmed upload selects the returned revision in this input slot; if you close${pending ? " now" : ""} before it completes, refresh the library before retrying.`}
      onClose={onClose}
    >
      {(upload) => (
        <ArtifactWriteForm
          {...(projectId === undefined
            ? { fixedNamespace: "inputs" }
            : {
                scope: { kind: "project", id: projectId },
                fixedNamespace: "artifacts",
              })}
          {...(mediaType === undefined ? {} : { fixedMediaType: mediaType })}
          acceptedMediaTypes={mediaTypes}
          startOperation={upload.start}
          submitLabel="Upload and select"
          headingId={upload.headingId}
          onPendingChange={setPending}
          onWritten={(result) => {
            if (upload.aborted()) return;
            onUploaded(result);
            onClose();
          }}
        />
      )}
    </UploadDialog>
  );
}
