import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import {
  getRuntimeAgentPrincipal,
  replaceRuntimeAgentPrincipalLabels,
  type RuntimeAgentPrincipal,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { Dialog } from "../../../app/dialog";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../artifacts/common";

export function AgentLabelsDialog({
  principal,
  bindings,
  onClose,
}: {
  principal: RuntimeAgentPrincipal;
  bindings: RuntimeLabelBinding[];
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const heading = useId();
  const description = useId();
  const firstInput = useRef<HTMLInputElement>(null);
  const [baseline, setBaseline] = useState(principal);
  const [labels, setLabels] = useState([...principal.labels]);
  const [conflict, setConflict] = useState(false);
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{
        runtimeAgentId: string;
        revision: string;
        labels: string[];
      }>("runtime-agent-labels"),
  );
  const refresh = () =>
    Promise.all([
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.snapshot,
      }),
    ]);
  const mutation = useMutation({
    mutationFn: () => {
      const request = {
        runtimeAgentId: baseline.runtimeAgentId,
        revision: baseline.revision,
        labels: [...labels].sort(),
      };
      return replaceRuntimeAgentPrincipalLabels(
        api,
        request.runtimeAgentId,
        request.labels,
        request.revision,
        keyring.keyFor(request),
      );
    },
    onSuccess: async () => {
      await refresh();
      onClose();
    },
    onError: (error) => {
      if (error instanceof PublicAPIError && error.status === 412)
        setConflict(true);
    },
  });
  const reload = useMutation({
    mutationFn: () => getRuntimeAgentPrincipal(api, principal.runtimeAgentId),
    onSuccess: async (current) => {
      await refresh();
      setBaseline(current);
      setLabels([...current.labels]);
      setConflict(false);
      mutation.reset();
    },
  });
  const stale = conflict || principal.revision !== baseline.revision;
  const pending = mutation.isPending || reload.isPending;
  const unchanged =
    JSON.stringify([...labels].sort()) ===
    JSON.stringify([...baseline.labels].sort());
  const selectable = bindings
    .filter((binding) => binding.label !== "default")
    .sort((a, b) => a.label.localeCompare(b.label));
  const unbound = labels.filter(
    (label) => !selectable.some((binding) => binding.label === label),
  );
  return (
    <Dialog
      className="project-dialog panel runtime-agent-label-dialog"
      labelledBy={heading}
      describedBy={description}
      initialFocusRef={firstInput}
      onRequestClose={() => {
        if (!pending) onClose();
      }}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!pending && !stale && !unchanged && unbound.length === 0)
            mutation.mutate();
        }}
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">
              Agent · {principal.runtimeAgentId.slice(0, 6)}…
              {principal.runtimeAgentId.slice(-4)}
            </p>
            <h2 id={heading}>Edit Agent labels</h2>
          </div>
          <button
            className="project-dialog-close"
            type="button"
            aria-label="Close Agent labels"
            disabled={pending}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p className="muted-copy" id={description}>
          Changes apply to future allocations. Work already in progress keeps
          its current settings.
        </p>
        <fieldset
          className="runtime-principal-labels"
          disabled={pending || stale}
        >
          <legend>Agent labels</legend>
          {selectable.length === 0 ? (
            <p className="compact-empty">No named Runtime labels are bound.</p>
          ) : (
            selectable.map((binding, index) => (
              <label className="checkbox-label" key={binding.label}>
                <input
                  ref={index === 0 ? firstInput : undefined}
                  type="checkbox"
                  checked={labels.includes(binding.label)}
                  onChange={(event) => {
                    setLabels((current) =>
                      (event.target.checked
                        ? [...current, binding.label]
                        : current.filter((label) => label !== binding.label)
                      ).sort(),
                    );
                    mutation.reset();
                  }}
                />
                <span>
                  <strong>{binding.label}</strong>
                  <small>
                    {binding.config.name}@{binding.config.version}
                  </small>
                </span>
              </label>
            ))
          )}
        </fieldset>
        {unbound.length === 0 ? null : (
          <p className="form-error" role="alert">
            Labels are no longer bound: {unbound.join(", ")}. Close and reopen
            after refreshing configuration.
          </p>
        )}
        {stale ? (
          <div className="notice notice-warning" role="alert">
            <strong>Agent labels changed in another view.</strong>
            <p>
              Reload the saved labels and review them before applying changes.
            </p>
            <button
              className="secondary-button"
              type="button"
              disabled={pending}
              onClick={() => reload.mutate()}
            >
              {reload.isPending ? "Reloading…" : "Reload labels"}
            </button>
          </div>
        ) : null}
        {mutation.error === null || stale ? null : (
          <ErrorNotice error={mutation.error} />
        )}
        {reload.error === null ? null : <ErrorNotice error={reload.error} />}
        <div className="runtime-agent-dialog-actions">
          <button
            className="secondary-button"
            type="button"
            disabled={pending}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={pending || stale || unchanged || unbound.length > 0}
          >
            {mutation.isPending ? "Saving labels…" : "Save labels"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
