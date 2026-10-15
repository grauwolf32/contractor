import {
  useMutation,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import { type FormEvent, useId, useState } from "react";

import { RUNTIME_INFRASTRUCTURE_ID } from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import {
  deleteRuntimeLabel,
  listAllRuntimeLabels,
  putRuntimeLabel,
  type RuntimeConfigResource,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { writeConflict } from "../../../app/conflict";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { ErrorNotice } from "../../../app/error-notice";
import { compactDigest } from "../../../app/format";
import { Icon } from "../../../app/icon";
import {
  createMutationIdempotencyKey,
  MutationDraftKeyring,
} from "../../../mutations/idempotency";
import { AlertDialog, Button, Facts, Field, IconButton } from "../../../ui";
import { InUseErrorDetails } from "../in-use-details";
import { exactKey, ruleMessage } from "./identity";

/** A version's name@version and short digest, for the compare grid. */
function VersionFact({
  version,
}: {
  version: { name: string; version: string; digest: string };
}) {
  return (
    <>
      <strong>
        {version.name}@{version.version}
      </strong>
      <code title={version.digest}>{compactDigest(version.digest)}</code>
    </>
  );
}

function BindingEditor({
  binding,
  configs,
}: {
  binding: RuntimeLabelBinding;
  configs: readonly RuntimeConfigResource[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState(() => {
    const current = configs.find(
      (resource) => resource.ref.digest === binding.config.digest,
    );
    return current === undefined ? "" : exactKey(current.ref);
  });
  const [conflictRevision, setConflictRevision] = useState<string>();
  const [confirmRemoval, setConfirmRemoval] = useState(false);
  const stale = conflictRevision === binding.revision;
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{
        label: string;
        config: string;
        revision: string;
      }>("bind-runtime-label"),
  );
  const reloadBindings = () =>
    queryClient.invalidateQueries({
      queryKey: queryKeys.operations.runtimeLabels.all,
    });
  // A 412 keeps the reviewed revision until the user reloads it on purpose;
  // any other failure reads the bindings again.
  const settleFailure = async (error: Error) => {
    const outdated = writeConflict(error) === "stale";
    setConflictRevision(outdated ? binding.revision : undefined);
    if (!outdated) await reloadBindings();
  };
  const mutation = useMutation({
    mutationFn: (resource: RuntimeConfigResource) =>
      putRuntimeLabel(
        api,
        binding.label,
        resource.ref,
        keyring.keyFor({
          label: binding.label,
          config: exactKey(resource.ref),
          revision: binding.revision,
        }),
        binding.revision,
      ),
    onSuccess: async () => {
      setConflictRevision(undefined);
      await reloadBindings();
    },
    onError: settleFailure,
  });
  const deletion = useMutation({
    mutationFn: () =>
      deleteRuntimeLabel(
        api,
        binding.label,
        binding.revision,
        createMutationIdempotencyKey("delete-runtime-label"),
      ),
    onSuccess: async () => {
      setConfirmRemoval(false);
      await reloadBindings();
    },
    onError: settleFailure,
  });
  const selectedResource = configs.find(
    (resource) => exactKey(resource.ref) === selected,
  );
  const failure = mutation.error ?? deletion.error;
  return (
    <article className="ops-binding-card runtime-binding-card">
      <header className="ops-binding-head">
        <strong>{binding.label}</strong>
        {binding.label === "default" ? (
          <span className="ops-scope">always applied</span>
        ) : null}
        <small>Current binding revision {binding.revision}</small>
      </header>
      <Facts
        className="ops-glance ops-binding-compare"
        items={[
          {
            term: "Current version",
            value: <VersionFact version={binding.config} />,
          },
          {
            term: "Proposed version",
            value:
              selectedResource === undefined ? (
                "Choose a loaded version"
              ) : (
                <VersionFact version={selectedResource.ref} />
              ),
          },
        ]}
      />
      <div className="ops-binding-actions">
        <select
          aria-label={`RuntimeConfig for ${binding.label}`}
          value={selected}
          onChange={(event) => {
            setSelected(event.target.value);
            mutation.reset();
          }}
        >
          <option value="">Current version is outside this loaded page</option>
          {configs.map((resource) => (
            <option key={exactKey(resource.ref)} value={exactKey(resource.ref)}>
              {resource.ref.name}@{resource.ref.version} ·{" "}
              {compactDigest(resource.ref.digest)}
            </option>
          ))}
        </select>
        <Button
          variant="primary"
          pending={mutation.isPending}
          pendingLabel="Rebinding…"
          disabled={
            selectedResource === undefined ||
            selectedResource.ref.digest === binding.config.digest ||
            stale
          }
          onClick={() =>
            selectedResource === undefined
              ? undefined
              : mutation.mutate(selectedResource)
          }
        >
          Rebind with current revision
        </Button>
        {binding.label === "default" ? null : (
          <IconButton
            variant="danger"
            label={`Remove binding ${binding.label}`}
            icon={<Icon name="delete" />}
            aria-haspopup="dialog"
            disabled={deletion.isPending || stale}
            onClick={() => {
              deletion.reset();
              setConfirmRemoval(true);
            }}
          />
        )}
      </div>
      {stale ? (
        <div className="notice notice-warning" role="alert">
          <strong>Binding changed in another view.</strong>
          <p>Reload the saved revision and review it before retrying.</p>
          <Button
            size="sm"
            onClick={() => {
              mutation.reset();
              deletion.reset();
              void reloadBindings();
            }}
          >
            Reload binding
          </Button>
        </div>
      ) : failure === null ? null : (
        <>
          <ErrorNotice error={failure} />
          <InUseErrorDetails error={failure} />
        </>
      )}
      {confirmRemoval ? (
        <AlertDialog
          eyebrow="Confirm removal"
          className="ops-confirm"
          dismissOnBackdrop
          title={`Remove binding ${binding.label}?`}
          pendingLabel="Removing…"
          confirmLabel="Remove binding"
          pending={deletion.isPending}
          confirmDisabled={stale}
          error={
            deletion.error === null ? null : (
              <>
                <ErrorNotice error={deletion.error} />
                <InUseErrorDetails error={deletion.error} />
              </>
            )
          }
          onCancel={() => {
            setConfirmRemoval(false);
            deletion.reset();
          }}
          onConfirm={() => deletion.mutate()}
        >
          <p>
            New Runs will no longer be able to select the label{" "}
            <code>{binding.label}</code>. To restore it, create a new binding.
          </p>
        </AlertDialog>
      ) : null}
    </article>
  );
}

/**
 * The label bindings of every loaded version, opened from one version: each
 * binding's editor, then a form that binds a new label.
 */
export function RuntimeBindingsDialog({
  target,
  configs,
  bindings,
  onClose,
}: {
  target: RuntimeConfigResource;
  configs: readonly RuntimeConfigResource[];
  bindings: readonly RuntimeLabelBinding[];
  onClose: () => void;
}) {
  const heading = useId();
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [label, setLabel] = useState("");
  const [selected, setSelected] = useState(exactKey(target.ref));
  const [error, setError] = useState<string>();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{ label: string; config: string }>(
        "create-runtime-label",
      ),
  );
  const mutation = useMutation({
    mutationFn: ({
      name,
      resource,
    }: {
      name: string;
      resource: RuntimeConfigResource;
    }) =>
      putRuntimeLabel(
        api,
        name,
        resource.ref,
        keyring.keyFor({ label: name, config: exactKey(resource.ref) }),
      ),
    onSuccess: async () => {
      setLabel("");
      setSelected(exactKey(target.ref));
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeLabels.all,
      });
    },
  });
  const ordered = [...bindings].sort((left, right) => {
    if (left.label === "default") return -1;
    if (right.label === "default") return 1;
    return left.label.localeCompare(right.label);
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.reset();
    const resource = configs.find(
      (candidate) => exactKey(candidate.ref) === selected,
    );
    if (!RUNTIME_INFRASTRUCTURE_ID.test(label) || label === "default") {
      setError(
        ruleMessage("Label", RUNTIME_INFRASTRUCTURE_ID, "not be default"),
      );
      return;
    }
    if (resource === undefined) {
      setError("Select a RuntimeConfig version.");
      return;
    }
    setError(undefined);
    mutation.mutate({ name: label, resource });
  }
  return (
    <Dialog
      className="ui-dialog ui-panel ops-dialog"
      labelledBy={heading}
      onRequestClose={onClose}
    >
      <DialogHeader
        id={heading}
        eyebrow={
          <>
            {target.ref.name}@{target.ref.version}
          </>
        }
        title="Runtime label bindings"
        close={{ label: "Close Runtime bindings", onClose: onClose }}
      />
      <div className="ops-dialog-form">
        <p className="ops-dialog-note">
          Default and Run-selected labels are pinned when a Run is created.
          Rebinding changes new Run snapshots. Labels assigned to a Runtime
          Agent are resolved again for future allocations, including allocations
          for an existing Run. Already prepared allocations keep their pinned
          settings.
        </p>
        {ordered.length === 0 ? (
          <p className="ops-empty">No label bindings are loaded.</p>
        ) : null}
        <div>
          <h3 className="ops-section-title">Current label bindings</h3>
          <p className="ops-section-description">
            Review current and proposed versions before rebinding.
          </p>
        </div>
        <div className="ops-binding-list">
          {ordered.map((binding) => (
            <BindingEditor
              key={binding.label}
              binding={binding}
              configs={configs}
            />
          ))}
        </div>
        <form
          className="ops-label-create runtime-label-create"
          onSubmit={submit}
          noValidate
        >
          <Field label="New label">
            {(control) => (
              <input
                {...control}
                value={label}
                onChange={(event) => {
                  setLabel(event.target.value);
                  setError(undefined);
                }}
              />
            )}
          </Field>
          <Field label="RuntimeConfig">
            {(control) => (
              <select
                {...control}
                value={selected}
                onChange={(event) => {
                  setSelected(event.target.value);
                  setError(undefined);
                }}
              >
                <option value="">Select version</option>
                {configs.map((resource) => (
                  <option
                    key={exactKey(resource.ref)}
                    value={exactKey(resource.ref)}
                  >
                    {resource.ref.name}@{resource.ref.version}
                  </option>
                ))}
              </select>
            )}
          </Field>
          <Button
            variant="primary"
            type="submit"
            pending={mutation.isPending}
            pendingLabel="Binding…"
          >
            Create binding
          </Button>
        </form>
        {error === undefined ? null : (
          <p className="ops-field-error" role="alert">
            {error}
          </p>
        )}
        {mutation.error === null ? null : (
          <ErrorNotice error={mutation.error} />
        )}
      </div>
    </Dialog>
  );
}

/** A version's bound labels and the control that opens its bindings. */
export function BindingCell({
  resource,
  labels,
  onManage,
}: {
  resource: RuntimeConfigResource;
  labels: UseQueryResult<Awaited<ReturnType<typeof listAllRuntimeLabels>>>;
  onManage: () => void;
}) {
  const bound = labels.isSuccess
    ? labels.data.items.filter(
        (binding) => binding.config.digest === resource.ref.digest,
      )
    : [];
  return (
    <div className="ops-binding-cell">
      {labels.isSuccess ? (
        bound.map((binding) => (
          <span
            className="ops-label-chip"
            data-tone={binding.label === "default" ? "default" : undefined}
            key={binding.label}
          >
            {binding.label}
          </span>
        ))
      ) : (
        <small className="ops-muted">Labels unavailable</small>
      )}
      {labels.isSuccess && bound.length === 0 ? (
        <small className="ops-muted">No labels</small>
      ) : null}
      <IconButton
        className={
          bound.length > 0
            ? "runtime-binding-trigger has-bindings"
            : "runtime-binding-trigger"
        }
        size="sm"
        label={`Manage bindings for ${resource.ref.name}@${resource.ref.version}`}
        icon={<Icon name="binding" />}
        aria-haspopup="dialog"
        disabled={!labels.isSuccess}
        onClick={onManage}
      />
    </div>
  );
}
