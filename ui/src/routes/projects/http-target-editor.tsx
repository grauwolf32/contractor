import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useId, useRef, useState } from "react";

import { usePublicAPI } from "../../api/context";
import {
  createRuntimeCredential,
  listRuntimeCredentials,
  type CreateRuntimeCredentialRequest,
  type RuntimeCredentialMetadata,
} from "../../api/operations";
import {
  normalizeProjectHTTPTarget,
  updateProject,
  type Project,
} from "../../api/projects";
import { loadCompletePages } from "../../api/pages";
import { queryKeys } from "../../api/query-keys";
import { AlertDialog, Button, Field } from "../../ui";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";
import { createMutationIdempotencyKey } from "../../mutations/idempotency";

import "./projects.css";

type TargetAuthMode = "none" | "existing" | "basic" | "bearer";
type OriginCredential = RuntimeCredentialMetadata & {
  kind: "http-origin-basic@1" | "http-origin-bearer@1";
};

const CREDENTIAL_KIND_LABELS: Record<OriginCredential["kind"], string> = {
  "http-origin-basic@1": "Basic credential",
  "http-origin-bearer@1": "Bearer token",
};

/**
 * How Settings was opened for the live target: "focus" moves focus to
 * Configure (or Edit) target, "open" also opens the target sheet.
 */
export type TargetReveal = "focus" | "open";

/**
 * The project's live target: the URL active checks may call and the origin
 * credential they authorize with. Secrets are write-only: they go to a new
 * Runtime credential and only its reference is stored on the project.
 */
export function ProjectHTTPTargetEditor({
  project,
  reveal,
}: {
  project: Project;
  /** How Settings was opened; acted on when the editor mounts. */
  reveal?: TargetReveal | undefined;
}) {
  const [editing, setEditing] = useState<Project | null>(null);
  const configure = useRef<HTMLButtonElement>(null);
  // The button takes focus before the sheet opens, so closing the sheet
  // returns focus to it. Settings keeps `reveal` from its first render.
  useEffect(() => {
    const button = configure.current;
    if (reveal === undefined || button === null) return;
    button.focus({ preventScroll: true });
    if (reveal === "open") button.click();
  }, [reveal]);
  // The project as it was when removal was asked for: its revision guards
  // the removal, so a target changed meanwhile is not removed unseen.
  const [removing, setRemoving] = useState<Project | null>(null);
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const removal = useMutation({
    mutationFn: (target: Project) =>
      updateProject(api, {
        projectId: target.projectId,
        expectedRevision: target.revision,
        request: { httpTarget: null },
      }),
    onSuccess: async (updated) => {
      queryClient.setQueryData(
        queryKeys.projects.detail(project.projectId),
        updated,
      );
      setRemoving(null);
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.lists(project.kind),
      });
    },
  });
  const target = project.httpTarget;

  return (
    <div className="projects-editor">
      {target === undefined ? (
        <p className="ui-caption">
          No target is configured. Workers receive no Project Authorization.
        </p>
      ) : (
        <dl className="projects-facts">
          <div className="projects-facts-wide">
            <dt>URL</dt>
            <dd>
              <code>{target.url}</code>
            </dd>
          </div>
          <div className="projects-facts-wide">
            <dt>Authorization</dt>
            <dd>
              {target.credential === undefined ? (
                "None"
              ) : (
                <>
                  {CREDENTIAL_KIND_LABELS[target.credential.kind]} ·{" "}
                  <code>{target.credential.kind}</code> ·{" "}
                  <code>{target.credential.credentialId}</code>
                </>
              )}
            </dd>
          </div>
        </dl>
      )}
      <div className="ui-toolbar">
        <Button ref={configure} size="sm" onClick={() => setEditing(project)}>
          {target === undefined ? "Configure target" : "Edit target"}
        </Button>
        {target === undefined ? null : (
          <Button
            variant="danger"
            size="sm"
            onClick={() => {
              removal.reset();
              setRemoving(project);
            }}
          >
            Remove target
          </Button>
        )}
      </div>
      {editing ? (
        <ProjectHTTPTargetDialog
          project={editing}
          onClose={() => setEditing(null)}
        />
      ) : null}
      {removing === null || removing.httpTarget === undefined ? null : (
        <AlertDialog
          eyebrow="Live target"
          title="Remove the live target?"
          pendingLabel="Removing…"
          confirmLabel="Remove target"
          pending={removal.isPending}
          error={
            removal.error === null ? undefined : (
              <ErrorNotice error={removal.error} reconcileWrite />
            )
          }
          onCancel={() => {
            setRemoving(null);
            removal.reset();
          }}
          onConfirm={() => removal.mutate(removing)}
        >
          <p>
            Checks of this project will no longer receive{" "}
            <code>{removing.httpTarget.url}</code> or its authorization. The
            Runtime credential itself is kept under Operations.
          </p>
        </AlertDialog>
      )}
    </div>
  );
}

function ProjectHTTPTargetDialog({
  project,
  onClose,
}: {
  project: Project;
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const heading = useId();
  const urlInput = useRef<HTMLInputElement>(null);
  const currentCredential = project.httpTarget?.credential;
  const [url, setURL] = useState(project.httpTarget?.url ?? "");
  const [authMode, setAuthMode] = useState<TargetAuthMode>(
    currentCredential === undefined ? "none" : "existing",
  );
  const [credentialID, setCredentialID] = useState(
    currentCredential?.credentialId ?? "",
  );
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  // Secrets stay masked unless the user asks to see them while typing.
  const [showSecrets, setShowSecrets] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const credentials = useQuery({
    queryKey: queryKeys.operations.runtimeCredentials.picker,
    queryFn: ({ signal }) =>
      loadCompletePages(
        "The Runtime credential list",
        (cursor) =>
          listRuntimeCredentials(api, cursor === undefined ? {} : { cursor }),
        { signal },
      ),
  });
  const originCredentials = (credentials.data ?? []).filter(
    (credential): credential is OriginCredential =>
      credential.kind === "http-origin-basic@1" ||
      credential.kind === "http-origin-bearer@1",
  );
  const currentCredentialMissing =
    currentCredential !== undefined &&
    !originCredentials.some(
      (credential) =>
        credential.credentialId === currentCredential.credentialId,
    );

  function clearSecrets(): void {
    setUsername("");
    setPassword("");
    setToken("");
  }

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    let target;
    try {
      target = normalizeProjectHTTPTarget({ url });
    } catch (reason) {
      setError(reason);
      return;
    }
    setPending(true);
    let createdCredentialID: string | undefined;
    try {
      if (authMode === "existing") {
        const credential =
          originCredentials.find(
            (candidate) => candidate.credentialId === credentialID,
          ) ??
          (currentCredential?.credentialId === credentialID
            ? currentCredential
            : undefined);
        if (credential === undefined) {
          throw new Error("Select an active HTTP origin credential.");
        }
        target = normalizeProjectHTTPTarget({
          url,
          credential: {
            credentialId: credential.credentialId,
            kind: credential.kind,
          },
        });
      } else if (authMode === "basic" || authMode === "bearer") {
        if (authMode === "basic" && (username === "" || password === "")) {
          throw new Error("Username and password are required.");
        }
        if (authMode === "bearer" && token === "") {
          throw new Error("Bearer token is required.");
        }
        const id = `project-target-${crypto.randomUUID()}`;
        const request: CreateRuntimeCredentialRequest =
          authMode === "basic"
            ? {
                credentialId: id,
                kind: "http-origin-basic@1",
                material: { username, password },
              }
            : {
                credentialId: id,
                kind: "http-origin-bearer@1",
                material: { token },
              };
        clearSecrets();
        const credential = await createRuntimeCredential(
          api,
          request,
          createMutationIdempotencyKey("create-project-target"),
        );
        createdCredentialID = credential.credentialId;
        target = normalizeProjectHTTPTarget({
          url,
          credential: {
            credentialId: credential.credentialId,
            kind:
              authMode === "basic"
                ? "http-origin-basic@1"
                : "http-origin-bearer@1",
          },
        });
      }
      const updated = await updateProject(api, {
        projectId: project.projectId,
        expectedRevision: project.revision,
        request: { httpTarget: target },
      });
      queryClient.setQueryData(
        queryKeys.projects.detail(project.projectId),
        updated,
      );
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.projects.lists(project.kind),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeCredentials.all,
        }),
      ]);
      onClose();
    } catch (reason) {
      // The credential exists but the project still points elsewhere: offer
      // the new credential for a plain retry instead of creating another.
      if (createdCredentialID !== undefined) {
        setError(
          new Error(
            `Credential ${createdCredentialID} is active, but the Project update failed. Select it under Existing credential and retry.`,
          ),
        );
        setCredentialID(createdCredentialID);
        setAuthMode("existing");
        await queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeCredentials.all,
        });
      } else {
        setError(reason);
      }
    } finally {
      clearSecrets();
      setPending(false);
    }
  }

  return (
    <Dialog
      className="ui-dialog ui-panel projects-target-dialog"
      labelledBy={heading}
      initialFocusRef={urlInput}
      onRequestClose={() => {
        if (!pending) onClose();
      }}
    >
      <DialogHeader
        id={heading}
        eyebrow="Live target"
        title="Application access"
        close={{
          label: "Close target dialog",
          disabled: pending,
          onClose: onClose,
        }}
      />
      <p className="ui-caption">
        The URL and credential reference are safe metadata. Secret material is
        write-only and reaches only matching HTTP-enabled allocations.
      </p>
      <form
        className="projects-form"
        autoComplete="off"
        onSubmit={(event) => void submit(event)}
      >
        <Field label="Application URL" className="projects-field">
          {(control) => (
            <input
              {...control}
              required
              ref={urlInput}
              type="url"
              placeholder="https://app.example.test"
              value={url}
              onChange={(event) => setURL(event.target.value)}
            />
          )}
        </Field>
        <Field label="Authorization" className="projects-field">
          {(control) => (
            <select
              {...control}
              value={authMode}
              onChange={(event) => {
                setAuthMode(event.target.value as TargetAuthMode);
                clearSecrets();
              }}
            >
              <option value="none">No Authorization</option>
              <option value="existing">Existing origin credential</option>
              <option value="basic">New Basic credential</option>
              <option value="bearer">New Bearer credential</option>
            </select>
          )}
        </Field>
        {authMode === "existing" ? (
          <>
            {credentials.isPending ? (
              <p className="loading-copy" role="status">
                Loading active credentials…
              </p>
            ) : null}
            {credentials.error === null ? null : (
              <ErrorNotice
                error={credentials.error}
                onRetry={() => void credentials.refetch()}
                retryPending={credentials.isFetching}
              />
            )}
            <Field
              label="Active HTTP origin credential"
              className="projects-field"
            >
              {(control) => (
                <select
                  {...control}
                  required
                  value={credentialID}
                  disabled={credentials.isFetching}
                  onChange={(event) => setCredentialID(event.target.value)}
                >
                  <option value="">Select credential</option>
                  {currentCredentialMissing ? (
                    <option value={currentCredential.credentialId}>
                      {currentCredential.credentialId} ·{" "}
                      {currentCredential.kind} (current)
                    </option>
                  ) : null}
                  {originCredentials.map((credential) => (
                    <option
                      key={credential.credentialId}
                      value={credential.credentialId}
                    >
                      {credential.credentialId} · {credential.kind}
                    </option>
                  ))}
                </select>
              )}
            </Field>
          </>
        ) : authMode === "basic" ? (
          <div className="projects-field-row">
            <Field label="Username" className="projects-field">
              {(control) => (
                <input
                  {...control}
                  autoComplete="off"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                />
              )}
            </Field>
            <Field label="Password · write only" className="projects-field">
              {(control) => (
                <input
                  {...control}
                  type={showSecrets ? "text" : "password"}
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              )}
            </Field>
          </div>
        ) : authMode === "bearer" ? (
          <Field label="Bearer token · write only" className="projects-field">
            {(control) => (
              <input
                {...control}
                type={showSecrets ? "text" : "password"}
                autoComplete="new-password"
                value={token}
                onChange={(event) => setToken(event.target.value)}
              />
            )}
          </Field>
        ) : null}
        {authMode === "basic" || authMode === "bearer" ? (
          <label className="projects-checkbox">
            <input
              type="checkbox"
              checked={showSecrets}
              onChange={(event) => setShowSecrets(event.target.checked)}
            />
            Show secret while entering
          </label>
        ) : null}
        {error === null ? null : <ErrorNotice error={error} reconcileWrite />}
        <div className="ui-dialog-actions">
          <Button
            variant="primary"
            type="submit"
            pending={pending}
            pendingLabel="Saving…"
            disabled={
              authMode === "existing" &&
              (credentials.isFetching || credentials.error !== null)
            }
          >
            Save target
          </Button>
          <Button disabled={pending} onClick={onClose}>
            Cancel
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
