import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { usePublicAPI } from "../../api/context";
import {
  getGitKey,
  gitKeyQueryKey,
  removeGitKey,
  replaceGitKey,
} from "../../api/git-artifacts";
import { ErrorNotice, formatTimestamp } from "../artifacts/common";

export function GitKeySettings() {
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
  return (
    <section
      className="panel configuration-draft"
      aria-labelledby="git-key-heading"
    >
      <div>
        <h3 id="git-key-heading">Git SSH key</h3>
        <p className="muted-copy">
          Your key is used to import private SSH repositories. Public HTTPS
          repositories need no key. Existing imported artifacts remain available
          after key removal.
        </p>
      </div>
      {query.isPending ? (
        <p>Loading Git key settings…</p>
      ) : query.error ? (
        <ErrorNotice error={query.error} />
      ) : (
        <p>
          {query.data?.configured ? (
            <>
              Configured: <code>{query.data.fingerprint}</code> ·{" "}
              {query.data.keyType} ·{" "}
              {query.data.updatedAt
                ? formatTimestamp(query.data.updatedAt)
                : ""}
            </>
          ) : (
            "No Git SSH key configured."
          )}
        </p>
      )}
      <form onSubmit={submit}>
        <label>
          SSH private key
          <textarea
            value={privateKey}
            onChange={(event) => setPrivateKey(event.target.value)}
            autoComplete="off"
            spellCheck={false}
            rows={6}
            disabled={pending}
            aria-describedby="git-key-help"
          />
        </label>
        <p id="git-key-help" className="muted-copy">
          Unencrypted OpenSSH or PEM, at most 32 KiB. The saved private key
          cannot be read back.
        </p>
        {error === undefined ? null : <ErrorNotice error={error} />}
        <div className="inline-actions">
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
            onClick={() => void change(true)}
          >
            Remove Git key
          </button>
        </div>
        {saved ? <p role="status">{saved}</p> : null}
      </form>
    </section>
  );
}
