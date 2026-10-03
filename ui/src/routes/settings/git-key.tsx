import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { usePublicAPI } from "../../api/context";
import {
  getGitKey,
  gitKeyQueryKey,
  removeGitKey,
  replaceGitKey,
} from "../../api/git-artifacts";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { ConfirmRemovalDialog } from "../../app/confirm-removal-dialog";
import { QueryView } from "../../app/query-view";

export function GitKeySettings({ ordinal = "02" }: { ordinal?: string } = {}) {
  const api = usePublicAPI();
  const cache = useQueryClient();
  const query = useQuery({
    queryKey: gitKeyQueryKey,
    queryFn: ({ signal }) => getGitKey(api, signal),
  });
  const [privateKey, setPrivateKey] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>();
  const [saved, setSaved] = useState("");
  const [confirmRemoval, setConfirmRemoval] = useState(false);
  const operation = useRef<AbortController | null>(null);
  useEffect(() => () => operation.current?.abort(), []);
  async function change(remove: boolean) {
    if (pending) return;
    if (
      !remove &&
      (privateKey === "" || new TextEncoder().encode(privateKey).length > 32768)
    ) {
      setError(
        new Error("Enter an unencrypted private key of at most 32 KiB."),
      );
      return;
    }
    const controller = new AbortController();
    operation.current = controller;
    setPending(true);
    setError(undefined);
    setSaved("");
    try {
      // An older metadata response must not overwrite the mutation result.
      await cache.cancelQueries({ queryKey: gitKeyQueryKey, exact: true });
      if (controller.signal.aborted) return;
      // Keep the secret out of mutation variables and the shared query cache.
      const state = remove
        ? (await removeGitKey(api, controller.signal), { configured: false })
        : await replaceGitKey(api, privateKey, controller.signal);
      if (controller.signal.aborted) return;
      setPrivateKey("");
      cache.setQueryData(gitKeyQueryKey, state);
      setSaved(remove ? "Git SSH key removed." : "Git SSH key saved.");
      if (remove) setConfirmRemoval(false);
    } catch (error) {
      if (!controller.signal.aborted) {
        setError(error);
        // Recover metadata if the initial read was cancelled before a failed
        // write; the write itself is never retried automatically.
        void cache.invalidateQueries({ queryKey: gitKeyQueryKey, exact: true });
      }
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }
  function submit(event: FormEvent) {
    event.preventDefault();
    void change(false);
  }
  const status = query.isPending
    ? "Checking"
    : query.error
      ? "Unavailable"
      : query.data?.configured
        ? "Configured"
        : "Not configured";
  const statusClass = query.isPending
    ? "settings-state-pending"
    : query.error
      ? "settings-state-error"
      : query.data?.configured
        ? "settings-state-active"
        : "settings-state-empty";
  return (
    <section
      id="repository-access"
      className="settings-section"
      aria-labelledby="git-key-heading"
    >
      <header className="settings-section-header">
        <div className="settings-section-identity">
          <span className="settings-section-mark" aria-hidden="true">
            {ordinal}
          </span>
          <div>
            <p className="eyebrow">Repository access</p>
            <h3 id="git-key-heading">Git SSH key</h3>
          </div>
        </div>
        <span className="settings-scope-badge">Personal credential</span>
      </header>

      <div className="settings-section-grid">
        <div className="settings-section-copy">
          <p>
            Use an SSH private key to import snapshots from private Git
            repositories. Public HTTPS repositories do not require one.
          </p>
          <div className="settings-behavior-note settings-credential-note">
            <span aria-hidden="true">◇</span>
            <div>
              <strong>Write-only from the UI</strong>
              <p>
                The saved key cannot be read back. Only its fingerprint, key
                type and update time are returned to the browser.
              </p>
            </div>
          </div>
          <p className="settings-related-link">
            Removing the key does not affect repository snapshots already
            imported as artifacts.
          </p>
        </div>

        <div className="settings-editor">
          <div className="settings-editor-heading">
            <div>
              <p className="eyebrow">Credential state</p>
              <h4>Private repository key</h4>
            </div>
            <span className={`settings-state-badge ${statusClass}`}>
              {status}
            </span>
          </div>

          <QueryView
            query={query}
            loading={
              <p className="loading-copy" role="status">
                Loading Git key settings…
              </p>
            }
            onRetry={() => void query.refetch()}
          >
            {(data) =>
              data.configured ? (
                <dl className="settings-fact-grid settings-key-facts">
                  <div>
                    <dt>Fingerprint</dt>
                    <dd>
                      <code>{data.fingerprint}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Key type</dt>
                    <dd>{data.keyType}</dd>
                  </div>
                  <div>
                    <dt>Last updated</dt>
                    <dd>
                      {data.updatedAt
                        ? formatTimestamp(data.updatedAt)
                        : "Unknown"}
                    </dd>
                  </div>
                </dl>
              ) : (
                <div className="settings-empty-state">
                  <span aria-hidden="true">+</span>
                  <div>
                    <strong>No Git SSH key configured.</strong>
                    <p>Add one below when a private SSH import requires it.</p>
                  </div>
                </div>
              )
            }
          </QueryView>

          <form className="settings-form" onSubmit={submit}>
            <label className="settings-primary-field">
              SSH private key
              <textarea
                className="settings-secret-input"
                aria-label="SSH private key"
                value={privateKey}
                onChange={(event) => setPrivateKey(event.target.value)}
                autoComplete="off"
                spellCheck={false}
                rows={7}
                disabled={pending}
                aria-describedby="git-key-help"
                placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"
              />
              <small id="git-key-help" className="field-guidance">
                Unencrypted OpenSSH or PEM · maximum 32 KiB.
              </small>
            </label>
            {error === undefined || confirmRemoval ? null : (
              <ErrorNotice error={error} />
            )}
            {saved ? (
              <div className="notice notice-success" role="status">
                {saved}
              </div>
            ) : null}
            <div className="settings-form-actions">
              <button type="submit" disabled={pending || privateKey === ""}>
                {pending
                  ? "Saving…"
                  : query.data?.configured
                    ? "Replace Git key"
                    : "Save Git key"}
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={pending || !query.data?.configured}
                aria-haspopup="dialog"
                onClick={() => {
                  setError(undefined);
                  setConfirmRemoval(true);
                }}
              >
                Remove Git key
              </button>
            </div>
          </form>
          {confirmRemoval ? (
            <ConfirmRemovalDialog
              title="Remove Git key?"
              description={
                <>
                  Remove the Git SSH key with fingerprint{" "}
                  <code>{query.data?.fingerprint}</code>? Private Git imports
                  will require a new private key.
                </>
              }
              confirmLabel="Remove Git key"
              pending={pending}
              error={error === undefined ? null : <ErrorNotice error={error} />}
              onCancel={() => {
                setConfirmRemoval(false);
                setError(undefined);
              }}
              onConfirm={() => void change(true)}
            />
          ) : null}
        </div>
      </div>
    </section>
  );
}
