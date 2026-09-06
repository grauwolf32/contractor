import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  useEffect,
  useId,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";

import {
  ARTIFACT_NAME_PATTERN,
  writeArtifact,
  type ArtifactWriteResponse,
} from "../api/artifacts";
import { usePublicAPI } from "../api/context";
import { queryKeys } from "../api/query-keys";
import { artifactFileStem } from "./artifacts/artifact-file";
import { ArtifactFileDrop, ErrorNotice } from "./artifacts/common";

const MAXIMUM_SKILL_ARCHIVE_BYTES = 16 * 1024 * 1024;

export function SkillUploadDialog({
  onClose,
  onWritten,
}: {
  onClose: () => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const heading = useId();
  const description = useId();
  const dialog = useRef<HTMLElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: ({ name, file }: { name: string; file: File }) =>
      writeArtifact(api, {
        namespace: "skills",
        name,
        mediaType: "application/vnd.contractor.agent-skill+zip",
        payload: file,
      }),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.artifacts.all,
      });
      onWritten(result);
    },
    onError: async () => {
      // Reconcile an ambiguous write without automatically repeating the PUT.
      await queryClient.invalidateQueries({
        queryKey: queryKeys.artifacts.all,
      });
    },
  });

  useEffect(() => {
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    nameInput.current?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected)
        previousFocus.focus();
    };
  }, []);

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      if (!mutation.isPending) onClose();
    }
    if (event.key !== "Tab") return;
    const focusable = Array.from(
      dialog.current?.querySelectorAll<HTMLElement>(
        'button, input, a[href], [tabindex="0"]',
      ) ?? [],
    ).filter((element) => !element.matches(":disabled"));
    const first = focusable[0];
    const last = focusable.at(-1);
    if (first === undefined) {
      event.preventDefault();
      dialog.current?.focus();
    } else if (
      event.shiftKey &&
      (document.activeElement === first ||
        document.activeElement === dialog.current)
    ) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function selectFile(selected: File | undefined) {
    if (mutation.isPending) return;
    setFile(selected ?? null);
    setError(null);
    mutation.reset();
    if (selected !== undefined && name.trim() === "")
      setName(artifactFileStem(selected.name));
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (mutation.isPending) return;
    setError(null);
    mutation.reset();
    const selectedName = name.trim();
    if (!ARTIFACT_NAME_PATTERN.test(selectedName)) {
      setError(
        "Name must use 1–128 letters, digits, dot, dash, or underscore.",
      );
    } else if (file === null) {
      setError("Choose a Skill ZIP package to upload.");
    } else if (file.size > MAXIMUM_SKILL_ARCHIVE_BYTES) {
      setError("Skill package exceeds the 16 MiB upload limit.");
    } else {
      mutation.mutate({ name: selectedName, file });
    }
  }

  return (
    <div className="project-dialog-backdrop" role="presentation">
      <section
        ref={dialog}
        className="project-dialog panel skill-upload-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={heading}
        aria-describedby={description}
        tabIndex={-1}
        onKeyDown={onKeyDown}
      >
        <div className="project-dialog-heading">
          <h2 id={heading}>Upload Skills</h2>
          <button
            className="project-dialog-close"
            type="button"
            aria-label="Close upload dialog"
            disabled={mutation.isPending}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p id={description} className="muted-copy">
          Upload one reviewed ZIP package with a root <code>SKILL.md</code>. It
          will be available across your Projects.
        </p>
        <form onSubmit={submit}>
          <fieldset
            disabled={mutation.isPending}
            className="skill-upload-fields"
          >
            <label>
              Name
              <input
                ref={nameInput}
                name="name"
                required
                maxLength={128}
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <ArtifactFileDrop
              file={file}
              maximumBytes={MAXIMUM_SKILL_ARCHIVE_BYTES}
              onSelect={selectFile}
            />
          </fieldset>
          {error === null ? null : (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          {mutation.error === null ? null : (
            <ErrorNotice error={mutation.error} reconcileWrite />
          )}
          <div className="skill-upload-actions">
            <button
              className="secondary-button"
              type="button"
              disabled={mutation.isPending}
              onClick={onClose}
            >
              Cancel
            </button>
            <button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? "Uploading…" : "Upload"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}
