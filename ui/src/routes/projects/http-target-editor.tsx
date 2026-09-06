import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useId, useState } from "react";
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
import { ErrorNotice } from "../artifacts/common";

type TargetAuthMode = "none" | "existing" | "basic" | "bearer";
type OriginCredential = RuntimeCredentialMetadata & {
  kind: "http-origin-basic@1" | "http-origin-bearer@1";
};

export function ProjectHTTPTargetEditor({ project }: { project: Project }) {
  const [editing, setEditing] = useState<Project | null>(null);
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () =>
      updateProject(api, {
        projectId: project.projectId,
        expectedRevision: project.revision,
        request: { httpTarget: null },
      }),
    onSuccess: async (updated) => {
      queryClient.setQueryData(
        queryKeys.projects.detail(project.projectId),
        updated,
      );
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.list(project.kind),
      });
    },
  });

  return (
    <div className="project-target-card">
      <div>
        <p className="eyebrow">Allocation-only HTTP configuration</p>
        <h4>Application target</h4>
        {project.httpTarget === undefined ? (
          <p className="muted-copy">
            No target is configured. Workers receive no Project Authorization.
          </p>
        ) : (
          <dl className="metadata-grid project-target-metadata">
            <div>
              <dt>URL</dt>
              <dd>
                <code>{project.httpTarget.url}</code>
              </dd>
            </div>
            <div>
              <dt>Authorization</dt>
              <dd>
                {project.httpTarget.credential === undefined
                  ? "None"
                  : `${project.httpTarget.credential.kind} · ${project.httpTarget.credential.credentialId}`}
              </dd>
            </div>
          </dl>
        )}
      </div>
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <div className="project-form-actions">
        <button
          className="secondary-button"
          type="button"
          onClick={() => setEditing(project)}
        >
          {project.httpTarget === undefined
            ? "Configure target"
            : "Edit target"}
        </button>
        {project.httpTarget === undefined ? null : (
          <button
            className="secondary-button"
            type="button"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Removing…" : "Remove target"}
          </button>
        )}
      </div>
      {editing ? (
        <ProjectHTTPTargetDialog
          project={editing}
          onClose={() => setEditing(null)}
        />
      ) : null}
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
  const [showSecrets, setShowSecrets] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const credentials = useQuery({
    queryKey: queryKeys.operations.runtimeCredentials.list(),
    queryFn: () => listRuntimeCredentials(api),
  });
  const originCredentials = (credentials.data?.items ?? []).filter(
    (credential): credential is OriginCredential =>
      credential.kind === "http-origin-basic@1" ||
      credential.kind === "http-origin-bearer@1",
  );

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !pending) onClose();
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose, pending]);

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
          `create-project-target-${crypto.randomUUID()}`,
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
          queryKey: queryKeys.projects.list(project.kind),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeCredentials.all,
        }),
      ]);
      onClose();
    } catch (reason) {
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
    <div className="project-dialog-backdrop" role="presentation">
      <section
        className="project-dialog panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={heading}
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">Project HTTP target</p>
            <h2 id={heading}>Application access</h2>
          </div>
          <button
            className="project-dialog-close"
            type="button"
            aria-label="Close target dialog"
            disabled={pending}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p className="muted-copy">
          The URL and credential reference are safe metadata. Secret material is
          write-only and reaches only matching HTTP-enabled allocations.
        </p>
        <form
          className="configuration-draft project-target-form"
          autoComplete="off"
          onSubmit={(event) => void submit(event)}
        >
          <label>
            Application URL
            <input
              required
              type="url"
              placeholder="https://app.example.test"
              value={url}
              onChange={(event) => setURL(event.target.value)}
            />
          </label>
          <label>
            Authorization
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
            credentials.isPending ? (
              <p className="loading-copy">Loading active credentials…</p>
            ) : credentials.error !== null ? (
              <ErrorNotice error={credentials.error} />
            ) : (
              <label>
                Active HTTP origin credential
                <select
                  required
                  value={credentialID}
                  onChange={(event) => setCredentialID(event.target.value)}
                >
                  <option value="">Select credential</option>
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
            )
          ) : authMode === "basic" ? (
            <div className="form-grid">
              <label>
                Username
                <input
                  autoComplete="off"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                />
              </label>
              <label>
                Password · write only
                <input
                  type={showSecrets ? "text" : "password"}
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              </label>
            </div>
          ) : authMode === "bearer" ? (
            <label>
              Bearer token · write only
              <input
                type={showSecrets ? "text" : "password"}
                autoComplete="new-password"
                value={token}
                onChange={(event) => setToken(event.target.value)}
              />
            </label>
          ) : null}
          {authMode === "basic" || authMode === "bearer" ? (
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showSecrets}
                onChange={(event) => setShowSecrets(event.target.checked)}
              />
              Show secret while entering
            </label>
          ) : null}
          {error === null ? null : <ErrorNotice error={error} reconcileWrite />}
          <div className="project-form-actions">
            <button
              type="submit"
              disabled={
                pending || (authMode === "existing" && credentials.isFetching)
              }
            >
              {pending ? "Saving…" : "Save target"}
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={pending}
              onClick={onClose}
            >
              Cancel
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}
