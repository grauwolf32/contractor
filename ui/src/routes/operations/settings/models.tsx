import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  compatibleModelSelection,
  getModelSelection,
  MODEL_ROLES,
  replaceModelSelection,
  sameModelSelection,
  type ModelSelectionValue,
} from "../../../api/model-selection";
import { queryKeys } from "../../../api/query-keys";
import { writeConflict } from "../../../app/conflict";
import { ErrorNotice } from "../../../app/error-notice";
import { formatNumber } from "../../../app/format";
import { QueryView } from "../../../app/query-view";
import { Button, Facts, Field, TechnicalDetails } from "../../../ui";
import { SettingSection } from "../../settings/section";

const labels = {
  planner: "Planner",
  worker: "Worker",
  summarizer: "Summarizer",
  audit_completion_worker: "Audit completion",
};

export function ModelSettings() {
  const api = usePublicAPI();
  const client = useQueryClient();
  const query = useQuery({
    queryKey: queryKeys.operations.modelSelection,
    queryFn: () => getModelSelection(api),
  });
  const [edit, setEdit] = useState<{
    value: ModelSelectionValue;
    etag: string;
  }>();
  const [conflict, setConflict] = useState(false);
  const [savedETag, setSavedETag] = useState<string>();
  const mutation = useMutation({
    mutationFn: ({
      value,
      etag,
    }: {
      value: ModelSelectionValue;
      etag: string;
    }) => replaceModelSelection(api, value, etag),
    onSuccess: (saved) => {
      client.setQueryData(queryKeys.operations.modelSelection, saved);
      setEdit(undefined);
      setConflict(false);
      setSavedETag(saved.etag);
    },
    onError: (error) => {
      setSavedETag(undefined);
      if (writeConflict(error) === "stale") {
        setConflict(true);
        void query.refetch();
      }
    },
  });

  return (
    <SettingSection
      id="model-selection"
      eyebrow="Model selection"
      title="Models"
      scope="Server-wide"
      about={
        <>
          <p>
            Choose one model for new runs and checks. Customize individual roles
            only when they need a different model.
          </p>
          <p>
            Existing runs, check drafts and prepared evaluations keep their
            models. Explicit Run and Eval choices take precedence.
          </p>
        </>
      }
    >
      <QueryView
        query={query}
        loading={<p role="status">Loading saved model choices…</p>}
      >
        {(snapshot) => {
          const selection = snapshot.resource.selection;
          const etag = snapshot.etag;
          if (
            !snapshot.resource.configured ||
            selection === undefined ||
            etag === undefined
          ) {
            return <p>No model catalog is configured for this server.</p>;
          }
          const value = edit?.value ?? selection;
          const stale = conflict || (edit !== undefined && edit.etag !== etag);
          const dirty =
            edit !== undefined && !sameModelSelection(value, selection);
          const compatible = compatibleModelSelection(
            value,
            snapshot.resource.models,
          );
          const savedModel = snapshot.resource.models.find(
            (model) => model.id === selection.defaultModel,
          );
          const previewModel = snapshot.resource.models.find(
            (model) => model.id === value.defaultModel,
          );
          const overrides = MODEL_ROLES.filter(
            (role) => value.roles[role] !== undefined,
          );
          const changes = [
            ...(value.defaultModel === selection.defaultModel
              ? []
              : [
                  {
                    term: "Default",
                    value: `${selection.defaultModel} → ${value.defaultModel}`,
                  },
                ]),
            ...MODEL_ROLES.filter(
              (role) => value.roles[role] !== selection.roles[role],
            ).map((role) => ({
              term: labels[role],
              value: `${selection.roles[role] ?? "Use default"} → ${value.roles[role] ?? "Use default"}`,
            })),
          ];
          const change = (next: ModelSelectionValue) => {
            setEdit({ value: next, etag: edit?.etag ?? etag });
            setSavedETag(undefined);
            mutation.reset();
          };
          const submit = (event: FormEvent<HTMLFormElement>) => {
            event.preventDefault();
            if (edit === undefined || stale || !dirty || !compatible) return;
            mutation.mutate(edit);
          };
          const reset = () => {
            setEdit(undefined);
            setConflict(false);
            setSavedETag(undefined);
            mutation.reset();
          };
          return (
            <form className="ops-setting-form" onSubmit={submit}>
              <Facts
                className="ops-glance"
                aria-label="Saved model selection"
                items={[
                  {
                    term: "Saved default",
                    value: savedModel?.id ?? selection.defaultModel,
                  },
                  {
                    term: "Context window",
                    value:
                      savedModel === undefined
                        ? "Unavailable"
                        : `${formatNumber(savedModel.contextWindowTokens)} tokens`,
                  },
                ]}
              />
              <div className="ops-model-choice">
                <Field label="Default model">
                  {(control) => (
                    <select
                      {...control}
                      value={value.defaultModel}
                      disabled={mutation.isPending}
                      onChange={(event) =>
                        change({
                          ...value,
                          roles: { ...value.roles },
                          defaultModel: event.target.value,
                        })
                      }
                    >
                      {snapshot.resource.models.some(
                        (model) => model.id === value.defaultModel,
                      ) ? null : (
                        <option value={value.defaultModel} disabled>
                          {value.defaultModel} (unavailable)
                        </option>
                      )}
                      {snapshot.resource.models.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.id}
                        </option>
                      ))}
                    </select>
                  )}
                </Field>
                {previewModel === undefined ? null : (
                  <p className="ui-caption">
                    {formatNumber(previewModel.contextWindowTokens)} context
                    tokens · up to {formatNumber(previewModel.maxOutputTokens)}{" "}
                    output tokens supported
                  </p>
                )}
              </div>
              <TechnicalDetails
                summary={`Models by role · ${overrides.length === 0 ? "all use default" : `${overrides.length} custom`}`}
                defaultOpen={Object.keys(selection.roles).length > 0}
                className="ops-model-roles"
                description="Leave a role on Use default to follow future changes to the shared model."
              >
                <div className="ops-model-role-grid">
                  {MODEL_ROLES.map((role) => {
                    const effective = snapshot.resource.models.find(
                      (model) =>
                        model.id === (value.roles[role] ?? value.defaultModel),
                    );
                    return (
                      <div key={role} className="ops-model-choice">
                        <Field label={`${labels[role]} model`}>
                          {(control) => (
                            <select
                              {...control}
                              value={value.roles[role] ?? ""}
                              disabled={mutation.isPending}
                              onChange={(event) => {
                                const roles = { ...value.roles };
                                if (event.target.value === "")
                                  delete roles[role];
                                else roles[role] = event.target.value;
                                change({ ...value, roles });
                              }}
                            >
                              <option value="">
                                Use default ({value.defaultModel})
                              </option>
                              {value.roles[role] === undefined ||
                              snapshot.resource.models.some(
                                (model) =>
                                  model.id === value.roles[role] &&
                                  model.policies[role] !== undefined,
                              ) ? null : (
                                <option value={value.roles[role]} disabled>
                                  {value.roles[role]} (unavailable)
                                </option>
                              )}
                              {snapshot.resource.models
                                .filter(
                                  (model) => model.policies[role] !== undefined,
                                )
                                .map((model) => (
                                  <option key={model.id} value={model.id}>
                                    {model.id}
                                  </option>
                                ))}
                            </select>
                          )}
                        </Field>
                        {effective === undefined ? null : (
                          <p className="ui-caption">
                            {effective.id} ·{" "}
                            {formatNumber(effective.contextWindowTokens)}{" "}
                            context tokens
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
                <Button
                  variant="ghost"
                  disabled={mutation.isPending || overrides.length === 0}
                  onClick={() => change({ ...value, roles: {} })}
                >
                  Use default for all roles
                </Button>
              </TechnicalDetails>
              {dirty && !stale ? (
                <div className="ops-model-preview" role="status">
                  <strong>Unsaved changes</strong>
                  <p>These choices apply to new executions after you save.</p>
                  <Facts aria-label="Model changes to save" items={changes} />
                </div>
              ) : null}
              {compatible ? null : (
                <p role="alert" className="notice notice-warning">
                  The selected model cannot support every role's output budget.
                  Choose a compatible override for those roles.
                </p>
              )}
              {stale ? (
                <p role="alert" className="notice notice-warning">
                  Model settings changed on the server. Your edits are
                  preserved. Load the saved selection before saving again.
                </p>
              ) : null}
              {mutation.error === null ||
              writeConflict(mutation.error) === "stale" ? null : (
                <ErrorNotice error={mutation.error} reconcileWrite />
              )}
              {savedETag === etag ? (
                <p role="status" className="notice notice-success">
                  Model defaults saved for new executions.
                </p>
              ) : null}
              <TechnicalDetails summary="Context and budgets">
                <p className="ui-caption">
                  Context capacity comes from each model profile. Spending and
                  output budgets belong to the role; the AgentTemplate controls
                  when summarization runs. Changing models does not raise these
                  budgets.
                </p>
              </TechnicalDetails>
              <div className="ops-form-actions">
                <Button
                  type="submit"
                  variant="primary"
                  disabled={!dirty || stale || !compatible}
                  pending={mutation.isPending}
                  pendingLabel="Saving…"
                >
                  Save model defaults
                </Button>
                <Button
                  onClick={reset}
                  disabled={
                    mutation.isPending || (edit === undefined && !conflict)
                  }
                >
                  {stale ? "Use saved selection" : "Discard changes"}
                </Button>
                {mutation.error !== null ? (
                  <Button
                    variant="ghost"
                    disabled={mutation.isPending}
                    pending={query.isFetching}
                    pendingLabel="Reloading…"
                    onClick={() => void query.refetch()}
                  >
                    Refresh from server
                  </Button>
                ) : null}
              </div>
            </form>
          );
        }}
      </QueryView>
    </SettingSection>
  );
}
