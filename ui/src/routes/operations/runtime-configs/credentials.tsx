import {
  useMutation,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import { type FormEvent, useId, useRef, useState } from "react";

import { RUNTIME_CREDENTIAL_ID } from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import {
  createRuntimeCredential,
  deleteRuntimeCredential,
  type CreateRuntimeCredentialRequest,
  type RuntimeCredentialMetadata,
  type RuntimeCredentialPage,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import type { CursorStackControls } from "../../../app/pagination";
import { QueryView } from "../../../app/query-view";
import { RecordedTime } from "../../../app/recorded-time";
import { createMutationIdempotencyKey } from "../../../mutations/idempotency";
import { AlertDialog, Button, Field, IconButton } from "../../../ui";
import { OpsSection } from "../common";
import { InUseErrorDetails } from "../in-use-details";
import { ruleMessage } from "./identity";

type CredentialKind = CreateRuntimeCredentialRequest["kind"];

/** Every kind the contract accepts, in the order the form offers them. */
const KIND_LABELS: Record<CredentialKind, string> = {
  "otlp-headers@1": "OTLP headers",
  "http-proxy-basic@1": "HTTP proxy basic auth",
  "http-proxy-bearer@1": "HTTP proxy bearer token",
  "caido-bearer@1": "Caido bearer token",
  "interactsh-token@1": "Interactsh server token",
  "http-origin-basic@1": "HTTP origin basic auth",
  "http-origin-bearer@1": "HTTP origin bearer token",
};

interface HeaderDraft {
  id: number;
  name: string;
  value: string;
}

export function RuntimeCredentialCreateForm({
  onClose,
}: {
  onClose: () => void;
}) {
  const heading = useId();
  const initialFocus = useRef<HTMLInputElement>(null);
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const nextHeaderID = useRef(2);
  const [credentialId, setCredentialId] = useState("");
  const [kind, setKind] = useState<CredentialKind>("otlp-headers@1");
  const [headers, setHeaders] = useState<HeaderDraft[]>([
    { id: 1, name: "Authorization", value: "" },
  ]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [created, setCreated] = useState<RuntimeCredentialMetadata>();
  const basic = kind === "http-proxy-basic@1" || kind === "http-origin-basic@1";

  function clearSecretFields() {
    setHeaders((current) =>
      current.map((header) => ({ ...header, value: "" })),
    );
    setUsername("");
    setPassword("");
    setToken("");
  }

  function updateHeader(id: number, change: Partial<HeaderDraft>) {
    setHeaders((current) =>
      current.map((header) =>
        header.id === id ? { ...header, ...change } : header,
      ),
    );
  }

  /** The request the form describes, or why it cannot be sent. */
  function request(): CreateRuntimeCredentialRequest | Error {
    if (!RUNTIME_CREDENTIAL_ID.test(credentialId)) {
      return new Error(ruleMessage("Credential ID", RUNTIME_CREDENTIAL_ID));
    }
    if (kind === "otlp-headers@1") {
      const material = Object.fromEntries(
        headers.map((header) => [header.name.trim(), header.value]),
      );
      return headers.length === 0 ||
        Object.keys(material).length !== headers.length ||
        Object.entries(material).some(
          ([name, value]) => name === "" || value === "",
        )
        ? new Error("Provide unique non-empty OTLP header names and values.")
        : { credentialId, kind, material: { headers: material } };
    }
    if (kind === "http-proxy-basic@1" || kind === "http-origin-basic@1") {
      return username === "" || password === ""
        ? new Error("Username and password are required.")
        : { credentialId, kind, material: { username, password } };
    }
    return token === ""
      ? new Error("Token is required.")
      : { credentialId, kind, material: { token } };
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) return;
    setCreated(undefined);
    const body = request();
    // Secret values never outlive a submit, whatever its outcome.
    clearSecretFields();
    if (body instanceof Error) {
      setError(body);
      return;
    }
    setError(null);
    setPending(true);
    try {
      const result = await createRuntimeCredential(
        api,
        body,
        createMutationIdempotencyKey("create-runtime-credential"),
      );
      setCreated(result);
      setCredentialId("");
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeCredentials.all,
      });
    } catch (reason) {
      setError(reason);
    } finally {
      setPending(false);
    }
  }

  return (
    <Dialog
      className="ui-dialog ui-panel ops-dialog"
      labelledBy={heading}
      initialFocusRef={initialFocus}
      onRequestClose={() => {
        if (!pending) onClose();
      }}
    >
      <form
        className="ops-dialog-form runtime-credential-form"
        onSubmit={(event) => void submit(event)}
        noValidate
        autoComplete="off"
      >
        <DialogHeader
          id={heading}
          eyebrow="Secret"
          title="Create Runtime credential"
          close={{
            label: "Close Runtime credential form",
            disabled: pending,
            onClose: onClose,
          }}
        />
        <p className="ops-dialog-note">
          Secret values are erased from the form immediately on submit and are
          never readable through the API. A failed request requires re-entry.
        </p>
        <div className="ops-form-grid">
          <Field label="Runtime credential ID">
            {(control) => (
              <input
                {...control}
                ref={initialFocus}
                autoComplete="off"
                value={credentialId}
                onChange={(event) => setCredentialId(event.target.value)}
              />
            )}
          </Field>
          <Field label="Credential kind">
            {(control) => (
              <select
                {...control}
                value={kind}
                onChange={(event) => {
                  setKind(event.target.value as CredentialKind);
                  clearSecretFields();
                }}
              >
                {Object.entries(KIND_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            )}
          </Field>
        </div>
        {kind === "otlp-headers@1" ? (
          <div className="ops-secret-rows">
            {headers.map((header) => (
              <div className="ops-secret-row" key={header.id}>
                <Field label="Header name">
                  {(control) => (
                    <input
                      {...control}
                      autoComplete="off"
                      value={header.name}
                      onChange={(event) =>
                        updateHeader(header.id, { name: event.target.value })
                      }
                    />
                  )}
                </Field>
                <Field label="Header value · write only">
                  {(control) => (
                    <input
                      {...control}
                      type="password"
                      autoComplete="new-password"
                      value={header.value}
                      onChange={(event) =>
                        updateHeader(header.id, { value: event.target.value })
                      }
                    />
                  )}
                </Field>
                <IconButton
                  variant="danger"
                  label={`Remove header ${header.name || header.id}`}
                  icon={<Icon name="delete" />}
                  disabled={headers.length === 1}
                  onClick={() =>
                    setHeaders((current) =>
                      current.filter((candidate) => candidate.id !== header.id),
                    )
                  }
                />
              </div>
            ))}
            <Button
              size="sm"
              onClick={() => {
                const id = nextHeaderID.current++;
                setHeaders((current) => [
                  ...current,
                  { id, name: "", value: "" },
                ]);
              }}
            >
              Add header
            </Button>
          </div>
        ) : basic ? (
          <div className="ops-form-grid">
            <Field
              label={`${kind === "http-origin-basic@1" ? "Origin" : "Proxy"} username · write only`}
            >
              {(control) => (
                <input
                  {...control}
                  autoComplete="off"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                />
              )}
            </Field>
            <Field
              label={`${kind === "http-origin-basic@1" ? "Origin" : "Proxy"} password · write only`}
            >
              {(control) => (
                <input
                  {...control}
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              )}
            </Field>
          </div>
        ) : (
          <Field
            label={
              kind === "interactsh-token@1"
                ? "Server token · write only"
                : "Bearer token · write only"
            }
          >
            {(control) => (
              <input
                {...control}
                type="password"
                autoComplete="new-password"
                value={token}
                onChange={(event) => setToken(event.target.value)}
              />
            )}
          </Field>
        )}
        {error === null ? null : <ErrorNotice error={error} />}
        {created === undefined ? null : (
          <div className="notice notice-success" role="status">
            Created safe metadata for {created.credentialId}; secret material
            cannot be read back.
          </div>
        )}
        <div className="ui-dialog-actions">
          <Button disabled={pending} onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            type="submit"
            pending={pending}
            pendingLabel="Creating…"
          >
            Create active Runtime credential
          </Button>
        </div>
      </form>
    </Dialog>
  );
}

/** One page of Runtime credentials, with deletion behind a confirmation. */
export function RuntimeCredentialList({
  credentials,
  controls,
  onCreate,
}: {
  credentials: UseQueryResult<RuntimeCredentialPage>;
  controls: CursorStackControls;
  onCreate: () => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [confirmRemoval, setConfirmRemoval] =
    useState<RuntimeCredentialMetadata | null>(null);
  const deletion = useMutation({
    mutationFn: (credentialId: string) =>
      deleteRuntimeCredential(
        api,
        credentialId,
        createMutationIdempotencyKey("delete-runtime-credential"),
      ),
    onSuccess: async () => {
      setConfirmRemoval(null);
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeCredentials.all,
      });
    },
  });
  return (
    <OpsSection
      id="runtime-credentials-heading"
      titleAs="h3"
      title="Runtime credentials"
      description="Secrets for telemetry sinks, HTTP proxies, Caido, Interactsh and project live targets (HTTP origins). Values are write-only; only safe metadata is listed."
      aside={
        credentials.data === undefined
          ? undefined
          : `${credentials.data.items.length} on this page`
      }
      actions={
        <Button
          size="sm"
          icon={<Icon name="add-credential" />}
          aria-haspopup="dialog"
          onClick={onCreate}
        >
          Add Runtime credential
        </Button>
      }
    >
      <QueryView
        query={credentials}
        loading={
          <p className="ops-loading" role="status">
            Loading Runtime credentials…
          </p>
        }
        errorContext="Could not load Runtime credentials"
        isEmpty={(page) => page.items.length === 0}
        empty={
          <p className="ops-empty">
            {controls.canGoBack
              ? "This page lists no Runtime credentials. Earlier pages may list more."
              : "No Runtime credentials exist."}
          </p>
        }
      >
        {(page) => (
          <div className="ops-table-wrap">
            <table className="ops-table" data-stack="">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Kind</th>
                  <th>Created</th>
                  <th>
                    <span className="ui-visually-hidden">Lifecycle</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((credential) => (
                  <tr key={credential.credentialId}>
                    <td data-label="ID">
                      <code className="ui-mono">{credential.credentialId}</code>
                    </td>
                    <td data-label="Kind">
                      <code className="ops-label-chip">{credential.kind}</code>
                    </td>
                    <td data-label="Created">
                      <RecordedTime value={credential.createdAt} />
                    </td>
                    <td data-label="" className="ops-cell-actions">
                      <IconButton
                        size="sm"
                        variant="danger"
                        label={`Delete Runtime credential ${credential.credentialId}`}
                        icon={<Icon name="delete" />}
                        aria-haspopup="dialog"
                        disabled={deletion.isPending}
                        onClick={() => {
                          deletion.reset();
                          setConfirmRemoval(credential);
                        }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </QueryView>
      <CursorControls label="Runtime credential pages" {...controls} />
      {confirmRemoval ? (
        <AlertDialog
          eyebrow="Confirm removal"
          className="ops-confirm"
          dismissOnBackdrop
          title={`Delete Runtime credential ${confirmRemoval.credentialId}?`}
          pendingLabel="Removing…"
          confirmLabel="Delete Runtime credential"
          pending={deletion.isPending}
          error={
            deletion.error === null ? null : (
              <>
                <ErrorNotice error={deletion.error} />
                <InUseErrorDetails error={deletion.error} />
              </>
            )
          }
          onCancel={() => {
            setConfirmRemoval(null);
            deletion.reset();
          }}
          onConfirm={() => deletion.mutate(confirmRemoval.credentialId)}
        >
          <p>
            This permanently deletes the <code>{confirmRemoval.kind}</code>{" "}
            credential <code>{confirmRemoval.credentialId}</code>. Its ID cannot
            be reused, and RuntimeConfig versions that name it cannot be bound
            again. Restore access by entering the secret under a new ID and
            publishing a new version.
          </p>
        </AlertDialog>
      ) : null}
    </OpsSection>
  );
}
