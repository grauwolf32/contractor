import { useInfiniteQuery } from "@tanstack/react-query";
import { useState } from "react";

import { listArtifacts } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import type { WorkflowResource } from "../../api/workflows";
import { ErrorNotice } from "../../app/error-notice";
import { LoadMoreButton } from "../../app/load-more";
import { artifactOptionKey } from "../../run-drafts/validation";
import { Button, Field, TechnicalDetails } from "../../ui";
import { resolveWorkflowWidget } from "../../workflow-widgets/registry";
import { pickerCursor, type useScanSettings } from "./settings-state";
import { RunInputUploadDialog } from "../workflows/run-input-upload-dialog";

export function ScanSettingsFields({
  settings,
  workflow,
  target = false,
}: {
  settings: ReturnType<typeof useScanSettings>;
  workflow: WorkflowResource;
  target?: boolean;
}) {
  const api = usePublicAPI();
  const [upload, setUpload] = useState(false);
  const mediaTypes = workflow.inputs.rulepack?.mediaTypes ?? [];
  const packs = useInfiniteQuery({
    queryKey: ["web-scans", "rulepacks"],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) =>
      listArtifacts(api, pageParam === null ? {} : { cursor: pageParam }),
    getNextPageParam: pickerCursor,
    getPreviousPageParam: () => undefined,
    maxPages: 20,
    enabled: settings.custom,
  });
  const options = new Map(
    (packs.data?.pages.flatMap((page) => page.items) ?? [])
      .filter((item) => mediaTypes.includes(item.mediaType))
      .map((item) => [artifactOptionKey(item.artifact), item]),
  );
  if (settings.pack !== undefined)
    options.set(artifactOptionKey(settings.pack.artifact), settings.pack);
  const widget = resolveWorkflowWidget(workflow);
  if (widget === undefined)
    return (
      <p role="alert">
        This published Workflow is incompatible with the scan widget. Open its
        generic configuration to review the changed contract.
      </p>
    );
  return (
    <div className="ui-stack">
      <Field label="Rule source">
        {(control) => (
          <select
            {...control}
            value={settings.custom ? "custom" : "installed"}
            onChange={(event) =>
              settings.setCustom(event.target.value === "custom")
            }
          >
            <option value="installed">Installed rules</option>
            <option value="custom" disabled={mediaTypes.length === 0}>
              My rulepack
            </option>
          </select>
        )}
      </Field>
      {settings.custom ? (
        <>
          <Field
            label="Rulepack from my library"
            required
            hint="One exact revision is used for the whole scan. Upload once and reuse it across Projects."
          >
            {(control) => (
              <select
                {...control}
                value={
                  settings.pack === undefined
                    ? ""
                    : artifactOptionKey(settings.pack.artifact)
                }
                onChange={(event) =>
                  settings.setPack(options.get(event.target.value))
                }
              >
                <option value="">Choose a rulepack</option>
                {[...options].map(([key, item]) => (
                  <option value={key} key={key}>
                    {item.artifact.namespace}/{item.artifact.name} ·{" "}
                    {item.artifact.revision}
                  </option>
                ))}
              </select>
            )}
          </Field>
          {packs.isPending ? <p role="status">Loading rulepacks…</p> : null}
          {packs.error === null ? null : (
            <ErrorNotice
              error={packs.error}
              onRetry={() => void packs.refetch()}
            />
          )}
          <div className="ui-toolbar">
            <LoadMoreButton query={packs} label="Load more library files" />
            <Button onClick={() => setUpload(true)}>Upload rulepack</Button>
          </div>
          {upload ? (
            <RunInputUploadDialog
              slotName="rulepack"
              mediaTypes={mediaTypes}
              onClose={() => setUpload(false)}
              onUploaded={(result) =>
                settings.setPack({
                  artifact: result.artifact,
                  mediaType: result.mediaType,
                  size: result.size,
                  current: true,
                  frozen: false,
                  createdAt: new Date().toISOString(),
                })
              }
            />
          ) : null}
        </>
      ) : (
        <p className="ui-caption">
          Uses the installed rule package pinned by the published scan
          configuration.
        </p>
      )}
      <widget.Fields
        parameters={settings.parameters}
        onChange={settings.change}
        includeTarget={target}
      />
      <TechnicalDetails>
        <p>
          Workflow:{" "}
          <code>
            {workflow.ref.name}@{workflow.ref.version}
          </code>
        </p>
        {settings.custom && settings.pack !== undefined ? (
          <p>
            Exact input:{" "}
            <code>{artifactOptionKey(settings.pack.artifact)}</code>
          </p>
        ) : null}
      </TechnicalDetails>
    </div>
  );
}
