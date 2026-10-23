import { useState } from "react";
import { Link } from "react-router";

import type { ArtifactArchiveScope } from "../../../api/artifact-archive";
import {
  FINDING_PROPOSAL_NAMESPACE,
  type ArtifactMetadata,
} from "../../../api/artifacts";
import {
  messageBytes,
  type HTTPExchangeDocument,
  type HTTPHeader,
  type HTTPMessage,
} from "../../../api/http-exchange";
import { saveBlob } from "../../../app/download";
import { formatBytes } from "../../../app/format";
import { Button, MethodChip, TechnicalDetails } from "../../../ui";
import { artifactDetailPath } from "../paths";
import { PanelTabs } from "../tabs";
import "./http-exchange.css";

const BODY_PREVIEW_CHARACTERS = 8192;
const HEADERS_PER_PAGE = 50;
const OUTCOME_LABELS: Record<string, string> = {
  transport_error: "Transport failed",
  timeout: "Request timed out",
  cancelled: "Request was interrupted",
  response_invalid: "Response could not be decoded",
  response_too_large: "Response exceeded the size limit",
  response_unavailable: "Response was unavailable when captured",
  target_denied: "Destination was denied",
  request_invalid: "A following request could not be sent",
};

export interface HTTPPreviewContext {
  metadata: ArtifactMetadata;
  scope?: ArtifactArchiveScope | undefined;
}

function Headers({ rows }: { rows: HTTPHeader[] }) {
  const [page, setPage] = useState(0);
  const start = page * HEADERS_PER_PAGE;
  return (
    <section aria-label="Headers">
      <h4>Headers</h4>
      {rows.length === 0 ? (
        <p className="ui-caption">No headers.</p>
      ) : (
        <dl className="http-evidence-headers">
          {rows.slice(start, start + HEADERS_PER_PAGE).map((row, index) => (
            <div key={start + index}>
              <dt>{row.name}</dt>
              <dd>{row.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {rows.length <= HEADERS_PER_PAGE ? null : (
        <div className="http-evidence-actions">
          <Button
            size="xs"
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            Previous headers
          </Button>
          <span>
            {start + 1}–{Math.min(start + HEADERS_PER_PAGE, rows.length)} of{" "}
            {rows.length}
          </span>
          <Button
            size="xs"
            disabled={start + HEADERS_PER_PAGE >= rows.length}
            onClick={() => setPage(page + 1)}
          >
            Next headers
          </Button>
        </div>
      )}
    </section>
  );
}

function Body({
  data,
  label = "Body",
}: {
  data: Uint8Array<ArrayBuffer>;
  label?: string;
}) {
  let text: string | undefined;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(data);
    // NUL and other binary control bytes should not appear as empty text.
    if (data.some((byte) => byte < 32 && ![9, 10, 13].includes(byte)))
      text = undefined;
  } catch {
    text = undefined;
  }
  let display = text;
  if (
    text !== undefined &&
    (text.trimStart().startsWith("{") || text.trimStart().startsWith("["))
  ) {
    try {
      display = JSON.stringify(JSON.parse(text), null, 2);
    } catch {
      /* Keep source text. */
    }
  }
  const characters = display === undefined ? [] : Array.from(display);
  const truncated = characters.length > BODY_PREVIEW_CHARACTERS;
  return (
    <section aria-label={label}>
      <div className="http-evidence-actions">
        <h4>{label}</h4>
        {data.length === 0 ? null : (
          <Button
            size="xs"
            onClick={() =>
              saveBlob(
                new Blob([data], { type: "application/octet-stream" }),
                label.toLowerCase().replaceAll(" ", "-") +
                  (text === undefined ? ".bin" : ".txt"),
              )
            }
          >
            Download {label.toLowerCase()}
          </Button>
        )}
      </div>
      {data.length === 0 ? (
        <p className="ui-caption">Empty body (0 bytes).</p>
      ) : display === undefined ? (
        <p className="ui-caption">
          Binary body · {formatBytes(data.length)}. Use Download.
        </p>
      ) : (
        <pre className="artifact-preview" tabIndex={0}>
          {characters.slice(0, BODY_PREVIEW_CHARACTERS).join("")}
        </pre>
      )}
      {!truncated ? null : (
        <p className="ui-caption">
          Showing the first 8192 characters. Download this body or the complete
          exchange for all bytes.
        </p>
      )}
    </section>
  );
}

function Message({
  value,
  direction,
}: {
  value: HTTPMessage;
  direction: "request" | "response";
}) {
  const data = messageBytes(value);
  if (value.kind === "structured")
    return (
      <>
        <Headers rows={value.headers} />
        <Body
          data={data}
          label={direction === "request" ? "Request body" : "Response body"}
        />
      </>
    );
  // Split raw framing only for display. Downloads preserve the original bytes.
  let separator = -1;
  let width = 4;
  for (let i = 0; i < data.length - 1; i++) {
    if (
      data[i] === 13 &&
      data[i + 1] === 10 &&
      data[i + 2] === 13 &&
      data[i + 3] === 10
    ) {
      separator = i;
      break;
    }
    if (data[i] === 10 && data[i + 1] === 10) {
      separator = i;
      width = 2;
      break;
    }
  }
  const headerBytes = data.subarray(0, separator < 0 ? data.length : separator);
  const shown = headerBytes.subarray(0, BODY_PREVIEW_CHARACTERS);
  const headerText = Array.from(shown, (byte) =>
    String.fromCharCode(byte),
  ).join("");
  return (
    <>
      <div className="http-evidence-actions">
        <h4>Raw {direction}</h4>
        <Button
          size="xs"
          onClick={() =>
            saveBlob(
              new Blob([data], { type: "application/octet-stream" }),
              direction + ".http",
            )
          }
        >
          Download raw {direction}
        </Button>
      </div>
      <section aria-label="Headers">
        <h4>Headers</h4>
        <pre className="artifact-preview" tabIndex={0}>
          {headerText}
        </pre>
        {headerBytes.length <= shown.length ? null : (
          <p className="ui-caption">
            Headers preview is shortened. Download the raw message for all
            bytes.
          </p>
        )}
      </section>
      {separator < 0 ? (
        <p className="ui-caption">
          Raw framing has no header/body separator; inspect the downloaded
          message.
        </p>
      ) : (
        <Body
          data={data.slice(separator + width)}
          label={direction === "request" ? "Request body" : "Response body"}
        />
      )}
    </>
  );
}

function Attempt({
  document,
  index,
}: {
  document: HTTPExchangeDocument;
  index: number;
}) {
  const attempt = document.capture.attempts[index]!;
  const response = attempt.response;
  return (
    <PanelTabs
      label="HTTP message"
      tabs={[
        {
          id: "request",
          label: "Request",
          panel: (
            <Message value={attempt.request.message} direction="request" />
          ),
        },
        {
          id: "response",
          label: "Response",
          panel: (
            <>
              {response === undefined ? (
                <p className="http-evidence-outcome">No response captured.</p>
              ) : (
                <>
                  <p className="http-evidence-outcome">
                    HTTP {response.status} ·{" "}
                    {response.complete
                      ? "Complete response"
                      : "Incomplete response"}
                  </p>
                  {response.complete ? null : (
                    <p className="ui-caption">
                      The saved body may be absent or partial. A redirect or
                      retry can retain only status and headers.
                    </p>
                  )}
                  <Message value={response.message} direction="response" />
                </>
              )}
              {attempt.error === undefined ? null : (
                <p className="http-evidence-outcome">
                  {OUTCOME_LABELS[attempt.error]}
                </p>
              )}
            </>
          ),
        },
      ]}
    />
  );
}

export default function HTTPExchangePreview({
  document,
  context,
}: {
  document: HTTPExchangeDocument;
  context?: HTTPPreviewContext | undefined;
}) {
  const [index, setIndex] = useState(0);
  const capture = document.capture;
  const attempt = capture.attempts[index]!;
  const scope = context?.scope;
  const trustedAssociation =
    scope?.kind === "run" &&
    context?.metadata.frozen === true &&
    context.metadata.artifact.namespace === FINDING_PROPOSAL_NAMESPACE &&
    context.metadata.artifact.name ===
      "http.exchange." + document.finding.proposal_id;
  return (
    <div className="http-evidence">
      <div className="http-evidence-heading">
        <MethodChip method={attempt.request.method} />
        <code>{attempt.request.url}</code>
      </div>
      {trustedAssociation && scope?.kind === "run" ? (
        <p className="ui-caption">
          Saved with a possible issue.{" "}
          <Link
            to={artifactDetailPath(scope, {
              namespace: FINDING_PROPOSAL_NAMESPACE,
              name: document.finding.proposal_id,
            })}
          >
            Open possible issue
          </Link>
        </p>
      ) : null}
      {capture.attempts.length === 1 ? null : (
        <label className="http-evidence-attempt">
          Attempt
          <select
            value={index}
            onChange={(event) => setIndex(Number(event.target.value))}
          >
            {capture.attempts.map((_, i) => (
              <option key={i} value={i}>
                {i + 1} of {capture.attempts.length}
              </option>
            ))}
          </select>
        </label>
      )}
      <Attempt key={index} document={document} index={index} />
      <TechnicalDetails>
        <dl className="http-evidence-headers">
          <div>
            <dt>Source</dt>
            <dd>{capture.source}</dd>
          </div>
          <div>
            <dt>Request ID</dt>
            <dd>{capture.request_id}</dd>
          </div>
          {capture.request_tag === undefined ? null : (
            <div>
              <dt>Request tag</dt>
              <dd>{capture.request_tag}</dd>
            </div>
          )}
          <div>
            <dt>Proposal ID</dt>
            <dd>{document.finding.proposal_id}</dd>
          </div>
          <div>
            <dt>Receipt ID</dt>
            <dd>{document.finding.receipt_id}</dd>
          </div>
        </dl>
      </TechnicalDetails>
    </div>
  );
}
