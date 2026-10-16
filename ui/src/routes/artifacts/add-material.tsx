import { useId, useState } from "react";
import { Link } from "react-router";

import {
  MAXIMUM_ARTIFACT_BYTES,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import type { GitImportResult } from "../../api/git-artifacts";
import { writeConflict } from "../../app/conflict";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ArtifactWriteForm } from "./common";
import { GitImportDialog } from "./git-import-dialog";
import { MaterialIcon, type MaterialIconName } from "./icons";
import { MATERIAL_UPLOAD_KINDS, type MaterialUploadKind } from "./kinds";
import { UploadDialog } from "./upload-dialog";
import { uploadCancelNote } from "./upload-operation";
import "./materials.css";

const SSH_KEY_SETTINGS = "/operations/settings#repository-access";

function KindButton({
  icon,
  label,
  description,
  accessibleName,
  onSelect,
}: {
  icon: MaterialIconName;
  label: string;
  description: string;
  accessibleName: string;
  onSelect: () => void;
}) {
  const descriptionId = useId();
  return (
    <li>
      <button
        type="button"
        className="materials-kind"
        aria-label={accessibleName}
        aria-describedby={descriptionId}
        onClick={onSelect}
      >
        <span className="materials-kind-icon">
          <MaterialIcon name={icon} size={20} />
        </span>
        <span className="materials-kind-text">
          <strong>{label}</strong>
          <small id={descriptionId}>{description}</small>
        </span>
      </button>
    </li>
  );
}

/**
 * Upload of one material of a chosen kind. The kind only suggests the
 * namespace and media type. Closing the dialog aborts a running upload
 * (UploadDialog).
 */
function MaterialUploadDialog({
  projectId,
  kind,
  onClose,
  onWritten,
}: {
  projectId: string;
  kind: MaterialUploadKind;
  onClose: () => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  return (
    <UploadDialog
      eyebrow="Add material"
      title={kind.label}
      closeLabel="Close upload dialog"
      intro={`${kind.description}. The kind only suggests a namespace and media type; change them if they do not fit. Checks and Runs read the exact version you add.`}
      note={uploadCancelNote("material")}
      onClose={onClose}
    >
      {(upload) => (
        <ArtifactWriteForm
          scope={{ kind: "project", id: projectId }}
          suggested={kind}
          headingId={upload.headingId}
          startOperation={upload.start}
          writeErrorHint={(error) =>
            // A create that met an existing binding: nothing was replaced.
            writeConflict(error) === undefined ? null : (
              <p className="materials-hint">
                Nothing was replaced. If a material with this namespace and name
                already exists, open it from the list to upload a new version,
                or choose another name.
              </p>
            )
          }
          onCancel={upload.close}
          onWritten={onWritten}
        />
      )}
    </UploadDialog>
  );
}

/**
 * The "Add material" sheet: the kinds of material a project takes, then the
 * upload form for one kind or the Git import. Every dialog stacks on the
 * sheet, so closing it returns to the kind that opened it.
 */
export function AddMaterialSheet({
  projectId,
  onClose,
  onAdded,
}: {
  projectId: string;
  onClose: () => void;
  onAdded: (result: ArtifactWriteResponse | GitImportResult) => void;
}) {
  const heading = useId();
  const description = useId();
  const [kind, setKind] = useState<MaterialUploadKind | null>(null);
  const [gitOpen, setGitOpen] = useState(false);

  return (
    <>
      <Dialog
        className="ui-dialog ui-panel materials-sheet"
        labelledBy={heading}
        describedBy={description}
        onRequestClose={onClose}
      >
        <DialogHeader
          id={heading}
          title="Add material"
          close={{ label: "Close material choices", onClose }}
        />
        <p id={description} className="materials-sheet-intro">
          Choose what you are adding. Checks and Runs read the exact version you
          add, so a later version never changes their results.
        </p>
        <ul className="materials-kinds" aria-label="Kinds of material">
          <KindButton
            icon="git"
            label="Git repository"
            description="Import a branch or tag as a source ZIP"
            accessibleName="Import Git repository"
            onSelect={() => setGitOpen(true)}
          />
          {MATERIAL_UPLOAD_KINDS.map((option) => (
            <KindButton
              key={option.id}
              icon={option.id}
              label={option.label}
              description={option.description}
              accessibleName={option.label}
              onSelect={() => setKind(option)}
            />
          ))}
        </ul>
        <p className="materials-sheet-note">
          Uploads take one file of up to{" "}
          {MAXIMUM_ARTIFACT_BYTES / (1024 * 1024)} MiB. Private repositories
          over SSH use your{" "}
          <Link to={SSH_KEY_SETTINGS}>Git key in Settings</Link>.
        </p>
      </Dialog>
      {gitOpen ? (
        <GitImportDialog
          projectId={projectId}
          onClose={() => setGitOpen(false)}
          onImported={(result) => {
            setGitOpen(false);
            onAdded(result);
          }}
        />
      ) : null}
      {kind === null ? null : (
        <MaterialUploadDialog
          projectId={projectId}
          kind={kind}
          onClose={() => setKind(null)}
          onWritten={(result) => {
            setKind(null);
            onAdded(result);
          }}
        />
      )}
    </>
  );
}
