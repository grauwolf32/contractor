import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import type { ArtifactArchiveScope } from "../../api/artifact-archive";
import type { AuditFinding } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { artifactScopeKeys } from "../../api/query-keys";
import { scopedArtifactAPI } from "../../api/scoped-artifacts";
import { canShowEvidenceSnippet } from "./evidence";
import { saveBlob } from "../../app/download";
import { ErrorNotice } from "../../app/error-notice";
import { formatBytes } from "../../app/format";
import { Icon } from "../../app/icon";
import { Button, IconButton, IdChip } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { ArtifactPreviewPanel } from "../artifacts/preview";
import { HTTP_EXCHANGE_MEDIA_TYPE } from "../../api/artifacts";

type Evidence = AuditFinding["firstProposal"]["evidence"][number];

/** Exact retained scope takes precedence over the possibly deleted source Run. */
export function FindingEvidenceFile({
  finding,
  index,
  snippet = false,
}: {
  finding: AuditFinding;
  index: number;
  snippet?: boolean;
}) {
  const receipt = finding.firstProposal;
  const original = receipt.evidence[index];
  if (original === undefined) return null;
  const hold = receipt.auditHolds.find(
    (item) => item.auditId === finding.auditId,
  );
  const retained = hold?.evidence[index];
  const scope: ArtifactArchiveScope | undefined =
    retained !== undefined && hold !== undefined
      ? { kind: "project", id: hold.projectId }
      : receipt.origin.runDeleted
        ? undefined
        : { kind: "run", id: receipt.origin.runId };
  const evidence = retained ?? original;
  const evidenceId = receipt.document.evidence_ids[index];
  const label = evidenceId ?? `evidence ${index + 1}`;
  const caption =
    evidence.mediaType === HTTP_EXCHANGE_MEDIA_TYPE
      ? "Request and available response"
      : evidenceId !== undefined &&
          evidenceId ===
            receipt.document.http_exchange?.response_body_evidence_id
        ? "Response body"
        : undefined;
  return (
    <EvidenceFile
      key={JSON.stringify([scope, evidence.ref])}
      evidence={evidence}
      scope={scope}
      label={label}
      caption={caption}
      snippet={snippet}
    />
  );
}

function EvidenceFile({
  evidence,
  scope,
  label,
  caption,
  snippet,
}: {
  evidence: Evidence;
  scope: ArtifactArchiveScope | undefined;
  snippet: boolean;
  label: string;
  caption?: string | undefined;
}) {
  return scope === undefined ? (
    <div className="issues-evidence-file">
      <EvidenceName evidence={evidence} label={label} caption={caption} />
      <p className="ui-caption">Source is unavailable.</p>
    </div>
  ) : (
    <AvailableEvidence
      evidence={evidence}
      scope={scope}
      label={label}
      caption={caption}
      snippet={snippet}
    />
  );
}
function EvidenceName({
  evidence,
  label,
  caption,
}: {
  evidence: Evidence;
  label: string;
  caption?: string | undefined;
}) {
  return (
    <div className="issues-evidence-name">
      <span className="issues-file-icon">
        <Icon name="document-code" size={16} />
      </span>
      <div>
        <strong>{label}</strong>
        <p className="ui-caption">
          <code>
            {evidence.ref.namespace}/{evidence.ref.name}@{evidence.ref.revision}
          </code>{" "}
          · {caption === undefined ? "" : `${caption} · `}
          {evidence.mediaType} · {formatBytes(evidence.sizeBytes)}
        </p>
      </div>
    </div>
  );
}
function AvailableEvidence({
  evidence,
  scope,
  label,
  caption,
  snippet,
}: {
  evidence: Evidence;
  scope: ArtifactArchiveScope;
  snippet: boolean;
  label: string;
  caption?: string | undefined;
}) {
  const api = usePublicAPI();
  const client = scopedArtifactAPI(api, scope);
  const cache = useQueryClient();
  const [requested, setRequested] = useState(snippet);
  const [open, setOpen] = useState(snippet);
  const [explicit, setExplicit] = useState(false);
  const key = artifactScopeKeys(scope).metadata(
    evidence.ref.namespace,
    evidence.ref.name,
    evidence.ref.revision,
  );
  const metadata = useQuery({
    queryKey: key,
    queryFn: ({ signal }) => client.metadata({ ...evidence.ref, signal }),
    enabled: requested,
    retry: false,
  });
  const download = useMutation({
    mutationFn: async () => {
      const exact = await cache.fetchQuery({
        queryKey: key,
        queryFn: () => client.metadata(evidence.ref),
      });
      return client.download(exact);
    },
    onSuccess: (downloaded) => saveBlob(downloaded.blob, downloaded.filename),
  });
  const data = metadata.data;
  const inline =
    data !== undefined &&
    canShowEvidenceSnippet({
      ...evidence,
      mediaType: data.mediaType,
      sizeBytes: data.size,
    });
  const identifier = `${evidence.ref.namespace}/${evidence.ref.name}@${evidence.ref.revision}`;
  return (
    <div className="issues-evidence-file" data-snippet={snippet || undefined}>
      <div className="issues-evidence-toolbar">
        <EvidenceName evidence={evidence} label={label} caption={caption} />
        <div className="issues-evidence-actions">
          <Button
            size="xs"
            aria-expanded={open}
            onClick={() => {
              setRequested(true);
              setExplicit(true);
              setOpen((value) => !value);
            }}
          >
            {open ? "Hide preview" : "Preview"}
          </Button>
          <span className="issues-evidence-copy">
            <IdChip value={identifier} label="artifact identifier" />
          </span>
          <IconButton
            size="xs"
            label={`Download ${label}`}
            icon={<Icon name="download" size={14} />}
            disabled={download.isPending}
            onClick={() => download.mutate()}
          />
        </div>
      </div>
      {download.error === null ? null : <ErrorNotice error={download.error} />}
      {!open ? null : (
        <div className="issues-evidence-preview">
          {metadata.error !== null ? (
            <>
              <ErrorNotice error={metadata.error} />
              <Button size="xs" onClick={() => void metadata.refetch()}>
                Retry
              </Button>
            </>
          ) : data === undefined ? (
            <p className="ui-caption" role="status">
              Loading evidence…
            </p>
          ) : (
            <ArtifactPreviewPanel
              compact={snippet && inline}
              archiveScope={scope}
              metadata={data}
              loadPreview={() => client.preview(data)}
              loadOnMountKey={
                snippet && !explicit && !inline
                  ? undefined
                  : [...key, "finding-preview"]
              }
              unavailableCopy="This evidence cannot be previewed here. Download the exact revision to inspect it."
            />
          )}
        </div>
      )}
      <div className="issues-evidence-footer">
        <Link
          to={artifactDetailPath(scope, evidence.ref)}
          aria-label={
            evidence.mediaType === HTTP_EXCHANGE_MEDIA_TYPE
              ? "Open HTTP exchange"
              : `Details for ${identifier}`
          }
        >
          details →
        </Link>
      </div>
    </div>
  );
}
