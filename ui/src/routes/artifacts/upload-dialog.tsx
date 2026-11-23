import { useId, type ReactNode } from "react";

import { Dialog, DialogHeader } from "../../app/dialog";
import { useUploadOperation } from "./upload-operation";
import "./materials.css";

/**
 * A sheet that uploads one file: heading with a close button, what it adds
 * (`intro`) and what closing it means (`note`), then the form `children`
 * renders with the abortable operation.
 */
export function UploadDialog({
  eyebrow,
  title,
  closeLabel,
  intro,
  note,
  onClose,
  children,
}: {
  eyebrow: ReactNode;
  title: ReactNode;
  closeLabel: string;
  intro?: ReactNode;
  note: ReactNode;
  onClose: () => void;
  children: (
    upload: ReturnType<typeof useUploadOperation> & { headingId: string },
  ) => ReactNode;
}) {
  const heading = useId();
  const introId = useId();
  const noteId = useId();
  const upload = useUploadOperation(onClose);
  return (
    <Dialog
      className="ui-dialog ui-panel materials-sheet"
      labelledBy={heading}
      describedBy={intro === undefined ? noteId : `${introId} ${noteId}`}
      onRequestClose={upload.close}
    >
      <DialogHeader
        id={heading}
        eyebrow={eyebrow}
        title={title}
        close={{ label: closeLabel, onClose: upload.close }}
      />
      {intro === undefined ? null : (
        <p id={introId} className="materials-sheet-intro">
          {intro}
        </p>
      )}
      <p id={noteId} className="ui-caption">
        {note}
      </p>
      {children({ ...upload, headingId: heading })}
    </Dialog>
  );
}
