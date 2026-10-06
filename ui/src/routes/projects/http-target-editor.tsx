import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useId, useRef, useState } from "react";

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
import { queryKeys } from "../../api/query-keys";
import { ConfirmRemovalDialog } from "../../app/confirm-removal-dialog";
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
 * The project's live target: the URL active checks may call and the origin
 * credential they authorize with. Secrets are write-only: they go to a new
 * Runtime credential and only its reference is stored on the project.
 */
export function ProjectHTTPTargetEditor({ project }: { project: Project }) {
  const [editing, setEditing] = useState<Project | null>(null);
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
        <p className="projects-caption">
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
      <div className="projects-form-actions">
        <button
          className="ui-btn"
          data-size="sm"
          type="button"
          onClick={() => setEditing(project)}
        >
          {target === undefined ? "Configure target" : "Edit target"}
        </button>
        {target === undefined ? null : (
          <button
            className="ui-btn"
            data-variant="danger"
            data-size="sm"
            type="button"
            onClick={() => {
              removal.reset();
              setRemoving(project);
            }}
          >
            Remove target
          </button>
        )}
      </div>
      {editing ? (
        <ProjectHTTPTargetDialog
          project={editing}
          onClose={() => setEditing(null)}
        />
      ) : null}
      {removing === null || removing.httpTarget === undefined ? null : (
        <ConfirmRemovalDialog
          eyebrow="Live target"
          title="Remove the live target?"
          description={
            <>
              Checks of this project will no longer receive{" "}
              <code>{removing.httpTarget.url}</code> or its authorization. The
              Runtime credential itself is kept under Operations.
            </>
          }
          confirmLabel="Remove target"
          pending={removal.isPending}
          dismissOnBackdrop={false}
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
        />
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
    queryFn: async ({ signal }) => {
      const items: RuntimeCredentialMetadata[] = [];
      const seen = new Set<string>();
      let cursor: string | undefined;
      for (;;) {
        signal.throwIfAborted();
        const page = await listRuntimeCredentials(
          api,
          cursor === undefined ? {} : { cursor },
        );
        signal.throwIfAborted();
        items.push(...page.items);
        if (!page.page.hasMore) return items;
        cursor = page.page.nextCursor;
        if (!cursor || seen.has(cursor)) {
          throw new Error(
            "Runtime credentials could not be fully loaded. Retry before choosing a credential.",
          );
        }
        seen.add(cursor);
      }
    },
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
      className="project-dialog panel projects-target-dialog"
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
      <p className="projects-caption">
        The URL and credential reference are safe metadata. Secret material is
        write-only and reaches only matching HTTP-enabled allocations.
      </p>
      <form
        className="projects-form"
        autoComplete="off"
        onSubmit={(event) => void submit(event)}
      >
        <label className="projects-field">
          <span>Application URL</span>
          <input
            required
            ref={urlInput}
            type="url"
            placeholder="https://app.example.test"
            value={url}
            onChange={(event) => setURL(event.target.value)}
          />
        </label>
        <label className="projects-field">
          <span>Authorization</span>
          <select
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
        </label>
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
            <label className="projects-field">
              <span>Active HTTP origin credential</span>
              <select
                required
                value={credentialID}
                disabled={credentials.isFetching}
                onChange={(event) => setCredentialID(event.target.value)}
              >
                <option value="">Select credential</option>
                {currentCredentialMissing ? (
                  <option value={currentCredential.credentialId}>
                    {currentCredential.credentialId} · {currentCredential.kind}{" "}
                    (current)
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
            </label>
          </>
        ) : authMode === "basic" ? (
          <div className="projects-field-row">
            <label className="projects-field">
              <span>Username</span>
              <input
                autoComplete="off"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
              />
            </label>
            <label className="projects-field">
              <span>Password · write only</span>
              <input
                type={showSecrets ? "text" : "password"}
                autoComplete="new-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
          </div>
        ) : authMode === "bearer" ? (
          <label className="projects-field">
            <span>Bearer token · write only</span>
            <input
              type={showSecrets ? "text" : "password"}
              autoComplete="new-password"
              value={token}
              onChange={(event) => setToken(event.target.value)}
            />
          </label>
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
        <div className="projects-form-actions">
          <button
            className="ui-btn"
            data-variant="primary"
            type="submit"
            disabled={
              pending ||
              (authMode === "existing" &&
                (credentials.isFetching || credentials.error !== null))
            }
          >
            {pending ? "Saving…" : "Save target"}
          </button>
          <button
            className="ui-btn"
            type="button"
            disabled={pending}
            onClick={onClose}
          >
            Cancel
          </button>
        </div>
      </form>
    </Dialog>
  );
}
