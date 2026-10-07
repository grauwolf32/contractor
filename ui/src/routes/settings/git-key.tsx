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
import type { StatusTone } from "../../app/status-tone";
import { ConfirmRemovalDialog } from "../../app/confirm-removal-dialog";
import { QueryView } from "../../app/query-view";
import { IdChip, StatusChip, StatusGlyph } from "../../ui";
import { Glance } from "../operations/common";
import { SettingSection, type SettingHeadingLevel } from "./section";

export function GitKeySettings({
  titleAs = "h3",
}: { titleAs?: SettingHeadingLevel | undefined } = {}) {
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
  // One level below the section title: h3 on the personal page, h4 under
  // Operations.
  const EditorHeading = titleAs === "h2" ? "h3" : "h4";
  const [status, tone]: [string, StatusTone] = query.isPending
    ? ["Checking", "progress"]
    : query.error
      ? ["Unavailable", "blocked"]
      : query.data?.configured
        ? ["Configured", "done"]
        : ["Not configured", "idle"];
  return (
    <SettingSection
      id="repository-access"
      headingId="git-key-heading"
      titleAs={titleAs}
      eyebrow="Repository access"
      title="Git SSH key"
      scope="Personal credential"
      about={
        <>
          <p>
            Use an SSH private key to import snapshots from private Git
            repositories. Public HTTPS repositories do not require one.
          </p>
          <div className="ops-callout">
            <StatusGlyph tone="success" size={15} />
            <div>
              <strong>Write-only from the UI</strong>
              <p>
                The saved key cannot be read back. Only its fingerprint, key
                type and update time are returned to the browser.
              </p>
            </div>
          </div>
          <p className="ops-note">
            Removing the key does not affect repository snapshots already
            imported as artifacts.
          </p>
        </>
      }
    >
      <div className="ops-setting-editor-head">
        <EditorHeading className="ops-setting-editor-title">
          Private repository key
        </EditorHeading>
        <StatusChip tone={tone} size="sm">
          {status}
        </StatusChip>
      </div>

      <QueryView
        query={query}
        loading={
          <p className="ops-loading" role="status">
            Loading Git key settings…
          </p>
        }
        onRetry={() => void query.refetch()}
      >
        {(data) =>
          data.configured ? (
            <Glance
              className="ops-key-facts"
              label="Saved key"
              items={[
                [
                  "Fingerprint",
                  data.fingerprint === undefined ? (
                    "Unknown"
                  ) : (
                    <IdChip
                      key="fingerprint"
                      value={data.fingerprint}
                      display={data.fingerprint}
                      label="fingerprint"
                    />
                  ),
                ],
                ["Key type", data.keyType],
                [
                  "Last updated",
                  data.updatedAt ? formatTimestamp(data.updatedAt) : "Unknown",
                ],
              ]}
            />
          ) : (
            <div className="ops-empty ops-setting-empty">
              <strong>No Git SSH key configured.</strong>
              <p>Add one below when a private SSH import requires it.</p>
            </div>
          )
        }
      </QueryView>

      <form className="ops-setting-form" onSubmit={submit}>
        <label className="ops-field">
          SSH private key
          <textarea
            className="ops-secret-input"
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
        <div className="ops-form-actions">
          <button
            className="ui-btn"
            data-variant="primary"
            type="submit"
            disabled={pending || privateKey === ""}
          >
            {pending
              ? "Saving…"
              : query.data?.configured
                ? "Replace Git key"
                : "Save Git key"}
          </button>
          <button
            className="ui-btn"
            data-variant="danger"
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
          className="ops-confirm"
          title="Remove Git key?"
          description={
            <>
              Remove the Git SSH key with fingerprint{" "}
              <code>{query.data?.fingerprint}</code>? Private Git imports will
              require a new private key.
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
    </SettingSection>
  );
}
