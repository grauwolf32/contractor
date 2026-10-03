import { useEffect, useId, useRef, useState } from "react";

import type { ArtifactWriteResponse } from "../../api/artifacts";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ArtifactWriteForm } from "../artifacts/common";

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
  const operation = useRef<AbortController | null>(null);
  const [pending, setPending] = useState(false);
  const mediaType = fixedMediaType(mediaTypes);

  useEffect(() => () => operation.current?.abort(), []);

  function startOperation(): AbortSignal {
    operation.current?.abort();
    const controller = new AbortController();
    operation.current = controller;
    return controller.signal;
  }

  function close(): void {
    operation.current?.abort();
    onClose();
  }

  function uploaded(result: ArtifactWriteResponse): void {
    if (operation.current?.signal.aborted) return;
    onUploaded(result);
    onClose();
  }

  const form =
    projectId === undefined ? (
      <ArtifactWriteForm
        fixedNamespace="inputs"
        {...(mediaType === undefined ? {} : { fixedMediaType: mediaType })}
        acceptedMediaTypes={mediaTypes}
        startOperation={startOperation}
        submitLabel="Upload and select"
        onPendingChange={setPending}
        onWritten={uploaded}
      />
    ) : (
      <ArtifactWriteForm
        scope={{ kind: "project", id: projectId }}
        fixedNamespace="artifacts"
        {...(mediaType === undefined ? {} : { fixedMediaType: mediaType })}
        acceptedMediaTypes={mediaTypes}
        startOperation={startOperation}
        submitLabel="Upload and select"
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
      <DialogHeader
        id={heading}
        eyebrow={<>{projectId === undefined ? "Library" : "Project"} input</>}
        title={<>Upload local file for {slotName}</>}
        close={{ label: "Close local file upload", onClose: close }}
      />
      <p id={description} className="muted-copy">
        A confirmed upload selects the returned revision in this input slot; if
        you close{pending ? " now" : ""} before it completes, refresh the
        library before retrying.
      </p>
      {form}
    </Dialog>
  );
}
