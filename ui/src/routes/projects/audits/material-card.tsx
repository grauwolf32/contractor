import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { readAuditArtifact } from "../../../api/audit-artifacts";
import {
  MAXIMUM_PREVIEW_BYTES,
  canPreviewArtifact,
} from "../../../api/artifacts";
import type { Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError, requireData } from "../../../api/error";
import { queryKeys } from "../../../api/query-keys";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { saveBlob } from "../../../app/download";
import { ErrorNotice } from "../../../app/error-notice";
import { formatBytes } from "../../../app/format";
import { Icon } from "../../../app/icon";
import { QueryView } from "../../../app/query-view";
import { createMutationIdempotencyKey } from "../../../mutations/idempotency";
import {
  Button,
  ButtonLink,
  Field,
  IconButton,
  TechnicalDetails,
} from "../../../ui";
import { LoadedArtifactPreview } from "../../artifacts/loaded-preview";
import { artifactDetailPath } from "../../artifacts/paths";
import type { ResearchMaterial } from "./research-data";

function Publication({
  audit,
  entry,
  onClose,
}: {
  audit: Audit;
  entry: ResearchMaterial;
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const cache = useQueryClient();
  const [name, setName] = useState(entry.material.ref.name);
  const [namespace, setNamespace] = useState("materials");
  const [request, setRequest] = useState<{
    key: string;
    namespace: string;
    name: string;
  }>();
  const write = useMutation({
    mutationFn: async () => {
      const pending = request ?? {
        key: createMutationIdempotencyKey("publish-audit-material"),
        namespace,
        name,
      };
      setRequest(pending);
      return requireData(
        await api.request((client) =>
          client.POST("/v1/projects/{projectId}/audit-material-publications", {
            params: {
              path: { projectId: audit.projectId },
              header: { "Idempotency-Key": pending.key, "If-None-Match": "*" },
            },
            body: {
              auditId: audit.auditId,
              source: entry.material.ref,
              target: { namespace: pending.namespace, name: pending.name },
            },
          }),
        ),
      );
    },
    onSuccess: () =>
      cache.invalidateQueries({
        queryKey: queryKeys.projects.artifacts.all(audit.projectId),
      }),
    onError: (error) => {
      if (
        error instanceof PublicAPIError &&
        error.status >= 400 &&
        error.status < 500 &&
        ![408, 429].includes(error.status)
      )
        setRequest(undefined);
    },
  });
  return (
    <Dialog
      className="ui-dialog ui-panel research-dialog"
      onRequestClose={() => {
        if (!write.isPending) onClose();
      }}
      labelledBy="publish-material-heading"
    >
      <DialogHeader
        id="publish-material-heading"
        title="Publish exact material to Project"
        close={{
          label: "Close publication",
          disabled: write.isPending,
          onClose,
        }}
      />
      <p>
        The selected revision remains available in the Project independently of
        this check. Source lineage is retained.
      </p>
      {write.data ? (
        <div className="ui-stack">
          <p role="status">Material published.</p>
          <ButtonLink
            to={artifactDetailPath(
              { kind: "project", id: audit.projectId },
              write.data.artifact,
            )}
          >
            Open Project material
          </ButtonLink>
          <Button onClick={onClose}>Close</Button>
        </div>
      ) : (
        <form
          className="ui-stack"
          onSubmit={(e) => {
            e.preventDefault();
            write.mutate();
          }}
        >
          <Field label="Collection" required>
            {(control) => (
              <input
                {...control}
                value={namespace}
                disabled={request !== undefined}
                onChange={(e) => setNamespace(e.target.value)}
              />
            )}
          </Field>
          <Field label="Material name" required>
            {(control) => (
              <input
                {...control}
                value={name}
                disabled={request !== undefined}
                onChange={(e) => setName(e.target.value)}
              />
            )}
          </Field>
          {write.error ? (
            <ErrorNotice error={write.error} reconcileWrite />
          ) : null}
          {request && write.error ? (
            <p className="ui-caption">
              This exact request is retained for retry. An existing destination
              is never overwritten.
            </p>
          ) : null}
          <div className="ui-dialog-actions">
            <Button disabled={write.isPending} onClick={onClose}>
              Close
            </Button>
            <Button type="submit" variant="primary" pending={write.isPending}>
              {request ? "Retry publication" : "Publish exact revision"}
            </Button>
          </div>
        </form>
      )}
    </Dialog>
  );
}

/** A receipt key and exact revision, never an inferred Run/Project fallback. */
export function ResearchMaterialCard({
  audit,
  entry,
  selected = false,
}: {
  audit: Audit;
  entry: ResearchMaterial;
  selected?: boolean;
}) {
  const api = usePublicAPI();
  const cache = useQueryClient();
  const [preview, setPreview] = useState(false);
  const [details, setDetails] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const m = entry.material;
  const read = useQuery({
    queryKey: [
      ...queryKeys.audits.detail(audit.auditId),
      "material-content",
      entry.key,
      m.ref.revision,
    ],
    enabled: preview && m.size_bytes <= MAXIMUM_PREVIEW_BYTES,
    queryFn: async ({ signal }) => {
      const result = await readAuditArtifact(
        api,
        audit.auditId,
        entry.key,
        m.ref.revision,
        undefined,
        signal,
      );
      return { source: await result.blob.text(), mediaType: result.mediaType };
    },
    staleTime: Infinity,
  });
  const download = useMutation({
    mutationFn: () =>
      readAuditArtifact(
        api,
        audit.auditId,
        entry.key,
        m.ref.revision,
        undefined,
        new AbortController().signal,
      ),
    onSuccess: (data) => saveBlob(data.blob, m.ref.name),
  });
  const previewable = canPreviewArtifact({
    mediaType: m.media_type,
    size: m.size_bytes,
  });
  return (
    <article
      className="research-material"
      data-selected={selected || undefined}
    >
      <header>
        <span className="research-symbol" data-tone="info">
          <Icon name="document-code" />
        </span>
        <div className="research-material-title">
          <h3>{m.ref.name}</h3>
          <p className="ui-caption">
            {m.media_type} · {formatBytes(m.size_bytes)}
          </p>
        </div>
        <div className="research-material-actions">
          <Button
            size="sm"
            icon={<Icon name="open" />}
            onClick={() => {
              if (preview)
                void cache.cancelQueries({
                  queryKey: [
                    ...queryKeys.audits.detail(audit.auditId),
                    "material-content",
                    entry.key,
                    m.ref.revision,
                  ],
                  exact: true,
                });
              setPreview(!preview);
            }}
          >
            {preview ? "Hide preview" : "Preview"}
          </Button>
          <IconButton
            size="sm"
            label={`Download ${m.ref.name}`}
            icon={<Icon name="download" />}
            pending={download.isPending}
            onClick={() => download.mutate()}
          />
        </div>
      </header>
      {preview ? (
        !previewable || m.size_bytes > MAXIMUM_PREVIEW_BYTES ? (
          <p className="ui-caption">
            This format or size is not available for inline preview. Download
            the exact material.
          </p>
        ) : (
          <QueryView query={read}>
            {(data) => (
              <LoadedArtifactPreview
                mediaType={data.mediaType}
                source={data.source}
              />
            )}
          </QueryView>
        )
      ) : null}
      {download.error ? <ErrorNotice error={download.error} /> : null}
      <footer>
        <Button variant="link" size="sm" onClick={() => setDetails(!details)}>
          details →
        </Button>
        <Button
          size="sm"
          icon={<Icon name="upload" />}
          onClick={() => setPublishing(true)}
        >
          Publish to Project
        </Button>
      </footer>
      {details ? (
        <TechnicalDetails
          defaultOpen
          description="Exact accepted material and its receipt."
        >
          <dl>
            <dt>Material</dt>
            <dd>
              <code>
                {m.ref.namespace}/{m.ref.name}@{m.ref.revision}
              </code>
            </dd>
            <dt>Digest</dt>
            <dd>
              <code>{m.digest}</code>
            </dd>
            <dt>Receipt</dt>
            <dd>{entry.key}</dd>
          </dl>
        </TechnicalDetails>
      ) : null}
      {publishing ? (
        <Publication
          audit={audit}
          entry={entry}
          onClose={() => setPublishing(false)}
        />
      ) : null}
    </article>
  );
}
