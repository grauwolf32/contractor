// Runtime configuration under Operations: RuntimeConfig versions with their
// label bindings (bindings.tsx), the publish form (publish-form.tsx, draft.ts)
// and Runtime credentials (credentials.tsx).
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  listAllRuntimeLabels,
  listRuntimeConfigs,
  listRuntimeCredentials,
  type RuntimeConfigResource,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import { runtimeConfigVersionPath } from "../../../app/navigation";
import { useCursorStack } from "../../../app/pagination";
import { QueryView } from "../../../app/query-view";
import { RecordedTime } from "../../../app/recorded-time";
import { Button, StatusChip } from "../../../ui";
import { EntryHeading, OpsSection, RefLink, ScopeChip } from "../common";
import { BindingCell, RuntimeBindingsDialog } from "./bindings";
import {
  RuntimeCredentialCreateForm,
  RuntimeCredentialList,
} from "./credentials";
import { exactKey } from "./identity";
import { RuntimeConfigPublishForm } from "./publish-form";

export function RuntimeConfigurationRoute() {
  const api = usePublicAPI();
  const [createDialog, setCreateDialog] = useState<"config" | "credential">();
  const [bindingTarget, setBindingTarget] = useState<RuntimeConfigResource>();
  const configPages = useCursorStack();
  const credentialPages = useCursorStack();
  const configCursor = configPages.cursor;
  const credentialCursor = credentialPages.cursor;
  const configs = useQuery({
    queryKey: queryKeys.operations.runtimeConfigs.list(configCursor),
    queryFn: () =>
      listRuntimeConfigs(
        api,
        configCursor === undefined ? {} : { cursor: configCursor },
      ),
  });
  const labels = useQuery({
    queryKey: queryKeys.operations.runtimeLabels.list(),
    queryFn: () => listAllRuntimeLabels(api),
  });
  const credentials = useQuery({
    queryKey: queryKeys.operations.runtimeCredentials.list(credentialCursor),
    queryFn: () =>
      listRuntimeCredentials(
        api,
        credentialCursor === undefined ? {} : { cursor: credentialCursor },
      ),
  });
  const loadedConfigs = configs.data?.items ?? [];
  return (
    <>
      {labels.error === null ? null : <ErrorNotice error={labels.error} />}
      <OpsSection
        id="runtime-config-versions-heading"
        title="Runtime configuration"
        eyebrow="Execution defaults"
        description="Published versions, label bindings and Runtime service credentials apply server-wide."
        aside={<ScopeChip>Server-wide</ScopeChip>}
        actions={
          <Button
            variant="primary"
            size="sm"
            icon={<Icon name="add-config" />}
            aria-haspopup="dialog"
            onClick={() => setCreateDialog("config")}
          >
            Publish RuntimeConfig
          </Button>
        }
      >
        <p className="ops-inventory-note">
          Publishing does not activate a version. A default or named label must
          point to its exact digest.
        </p>
        <QueryView
          query={configs}
          loading={
            <p className="ops-loading" role="status">
              Loading RuntimeConfigs…
            </p>
          }
          errorContext="Could not load RuntimeConfig versions"
          isEmpty={(page) => page.items.length === 0}
          empty={
            <p className="ops-empty">
              {configPages.cursor === undefined
                ? "No RuntimeConfig versions are visible."
                : "This page lists no RuntimeConfig versions. Earlier pages may list more."}
            </p>
          }
        >
          {(page) => (
            <div className="ops-resource-list">
              {page.items.map((resource) => {
                const selected = labels.data?.items.filter(
                  (binding) =>
                    exactKey(binding.config) === exactKey(resource.ref),
                );
                return (
                  <details
                    className="ops-resource-row ops-resource-disclosure"
                    key={exactKey(resource.ref)}
                  >
                    <summary className="ops-resource-heading">
                      <EntryHeading
                        icon="binding"
                        title={
                          <strong>
                            {resource.ref.name}@{resource.ref.version}
                          </strong>
                        }
                        subtitle={
                          resource.builtIn
                            ? "Built-in version"
                            : resource.createdBy
                        }
                      >
                        <div className="ops-chips">
                          {selected === undefined ? (
                            <span className="ops-muted">
                              Bindings unavailable
                            </span>
                          ) : selected.length === 0 ? (
                            <StatusChip tone="neutral" size="sm">
                              Not bound
                            </StatusChip>
                          ) : (
                            selected.map((binding) => (
                              <StatusChip
                                key={binding.label}
                                tone="neutral"
                                size="sm"
                              >
                                {binding.label}
                              </StatusChip>
                            ))
                          )}
                        </div>
                      </EntryHeading>
                    </summary>
                    <div className="ops-resource-body">
                      <RefLink
                        to={runtimeConfigVersionPath(resource.ref)}
                        {...resource.ref}
                      />
                      <div className="ops-resource-meta">
                        <span>
                          Created:{" "}
                          {resource.builtIn ? (
                            "Built-in"
                          ) : (
                            <RecordedTime value={resource.createdAt} />
                          )}
                        </span>
                      </div>
                      <BindingCell
                        resource={resource}
                        labels={labels}
                        onManage={() => setBindingTarget(resource)}
                      />
                    </div>
                  </details>
                );
              })}
            </div>
          )}
        </QueryView>
        <CursorControls
          label="RuntimeConfig pages"
          {...configPages.controls(configs.data?.page)}
        />
      </OpsSection>
      <div className="ops-divided">
        <RuntimeCredentialList
          credentials={credentials}
          controls={credentialPages.controls(credentials.data?.page)}
          onCreate={() => setCreateDialog("credential")}
        />
      </div>
      {createDialog === "credential" ? (
        <RuntimeCredentialCreateForm
          onClose={() => setCreateDialog(undefined)}
        />
      ) : null}
      {createDialog === "config" ? (
        <RuntimeConfigPublishForm
          configs={loadedConfigs}
          onClose={() => setCreateDialog(undefined)}
        />
      ) : null}
      {bindingTarget === undefined ? null : (
        <RuntimeBindingsDialog
          target={bindingTarget}
          configs={loadedConfigs}
          bindings={labels.data?.items ?? []}
          onClose={() => setBindingTarget(undefined)}
        />
      )}
    </>
  );
}
