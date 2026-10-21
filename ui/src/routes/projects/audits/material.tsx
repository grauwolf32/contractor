import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";

import { readAuditArtifact } from "../../../api/audit-artifacts";
import { MAXIMUM_PREVIEW_BYTES } from "../../../api/artifacts";
import { usePublicAPI } from "../../../api/context";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { saveBlob } from "../../../app/download";
import { ErrorNotice } from "../../../app/error-notice";
import { formatBytes } from "../../../app/format";
import { Button, TechnicalDetails } from "../../../ui";
import type { Audit } from "../../../api/audits";

type Exact = Audit["inputs"][string];

function MaterialPreview({
  auditId,
  receipt,
  artifact,
  member,
  label,
  close,
}: {
  auditId: string;
  receipt: string;
  artifact: Exact;
  member: string | undefined;
  label: string;
  close: () => void;
}) {
  const api = usePublicAPI();
  const heading = useId();
  const query = useQuery({
    queryKey: [
      "audit-material",
      auditId,
      receipt,
      artifact.ref.revision,
      member,
    ],
    queryFn: async ({ signal }) => {
      const read = await readAuditArtifact(
        api,
        auditId,
        receipt,
        artifact.ref.revision,
        member,
        signal,
      );
      const text =
        read.blob.size <= MAXIMUM_PREVIEW_BYTES &&
        /^(text\/|application\/(json|yaml|x-yaml)(;|$))/u.test(read.mediaType)
          ? await read.blob.text()
          : undefined;
      return { ...read, text };
    },
    retry: false,
    gcTime: 0,
    refetchOnWindowFocus: false,
  });
  return (
    <Dialog
      className="ui-dialog ui-panel checks-material-dialog"
      labelledBy={heading}
      onRequestClose={close}
    >
      <DialogHeader
        id={heading}
        title={label}
        eyebrow="Saved material"
        close={{ label: "Close material", onClose: close }}
      />
      {query.isPending ? <p role="status">Loading saved material…</p> : null}
      {query.error === null ? null : (
        <ErrorNotice
          error={query.error}
          onRetry={() => void query.refetch()}
          retryPending={query.isFetching}
        />
      )}
      {query.data === undefined ? null : (
        <>
          <p className="ui-caption">
            {query.data.mediaType} · {formatBytes(query.data.blob.size)}
          </p>
          {query.data.text === undefined ? (
            <p>
              Preview unavailable for this format or size. Download the saved
              material to inspect it.
            </p>
          ) : (
            <pre className="checks-material-preview" tabIndex={0}>
              {query.data.text}
            </pre>
          )}
          <Button
            onClick={() =>
              saveBlob(query.data!.blob, member ?? artifact.ref.name)
            }
          >
            Download {label}
          </Button>
        </>
      )}
      <TechnicalDetails>
        <code className="ui-mono">
          {artifact.ref.namespace}/{artifact.ref.name}@{artifact.ref.revision}
        </code>
        <p className="ui-mono">{artifact.digest}</p>
      </TechnicalDetails>
    </Dialog>
  );
}

/** The receipt, rather than a Project-prefix browser, authorizes this read. */
export function AuditMaterial({
  auditId,
  receipt,
  artifact,
  member,
  label,
}: {
  auditId: string;
  receipt: string;
  artifact: Exact;
  member?: string | undefined;
  label: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button variant="link" size="sm" onClick={() => setOpen(true)}>
        {label}
      </Button>
      {open ? (
        <MaterialPreview
          auditId={auditId}
          receipt={receipt}
          artifact={artifact}
          member={member}
          label={label}
          close={() => setOpen(false)}
        />
      ) : null}
    </>
  );
}
