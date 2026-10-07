import { useState } from "react";

import type { components } from "../../../api/generated/public";
import { formatBytes } from "../../../app/format";
import { MethodChip } from "../../../ui";
import {
  attemptOutcome,
  decodeBody,
  isCredentialHeader,
} from "../../issues/evidence";

import "../../issues/issues.css";

type Proposal = components["schemas"]["FindingProposalDocument"];
type FindingLocation = components["schemas"]["FindingLocation"];
type HTTPExchange = components["schemas"]["FindingHTTPExchange"];
type HTTPHeader = components["schemas"]["FindingHTTPHeader"];

/**
 * Authored source and web locations, as the producer wrote them: paths and
 * URLs are claims, so they are shown as text and never turned into links.
 */
export function LocationList({
  locations,
}: {
  locations: readonly FindingLocation[];
}) {
  return (
    <ul className="issues-location-list">
      {locations.map((location, index) => (
        // Locations are positional; the producer may repeat one.
        <li key={index}>
          {"file" in location ? (
            <code>
              {location.file}
              {location.line !== undefined ? `:${location.line}` : ""}
              {location.range
                ? `:${location.range.start_line}–${location.range.end_line}`
                : ""}
            </code>
          ) : (
            <span className="issues-location-web">
              {location.method ? (
                <>
                  <MethodChip method={location.method} />{" "}
                </>
              ) : null}
              <code>{location.url}</code>
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}

/**
 * A captured header value. Credential-looking values (see
 * isCredentialHeader) are not in the page until the user shows them.
 */
function HeaderValue({ name, value }: HTTPHeader) {
  const [shown, setShown] = useState(false);
  if (!isCredentialHeader(name))
    return <span className="issues-header-text">{value}</span>;
  return (
    <span className="issues-header-secret">
      {shown ? (
        <span className="issues-header-text">{value}</span>
      ) : (
        <span className="issues-masked" title="Hidden credential">
          <span aria-hidden="true">••••••••</span>
          <span className="ui-visually-hidden">Hidden</span>
        </span>
      )}
      <button
        type="button"
        className="ui-btn issues-reveal"
        data-size="xs"
        data-variant="ghost"
        aria-label={`${shown ? "Hide" : "Show"} ${name} value`}
        onClick={() => setShown((value) => !value)}
      >
        {shown ? "Hide" : "Show"}
      </button>
    </span>
  );
}

function HeaderList({
  label,
  headers,
}: {
  label: string;
  headers: readonly HTTPHeader[];
}) {
  return (
    <div className="issues-exchange-part">
      <p className="issues-exchange-label">{label}</p>
      {headers.length === 0 ? (
        <p className="issues-quiet">None</p>
      ) : (
        <dl className="issues-headers">
          {headers.map((header, index) => (
            // Headers are positional; a name can repeat.
            <div key={index}>
              <dt>{header.name}</dt>
              <dd>
                <HeaderValue name={header.name} value={header.value} />
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function RequestBody({ base64 }: { base64: string }) {
  const body = decodeBody(base64);
  if (body.kind === "empty")
    return (
      <div className="issues-exchange-part">
        <p className="issues-exchange-label">Request body</p>
        <p className="issues-quiet">None</p>
      </div>
    );
  return (
    <div className="issues-exchange-part">
      <p className="issues-exchange-label">
        Request body{" "}
        <span className="issues-quiet">
          {body.kind === "text"
            ? formatBytes(body.bytes)
            : `binary, ${formatBytes(body.bytes)}, base64`}
        </span>
      </p>
      <pre className="issues-code">
        {body.kind === "text" ? body.text : base64}
      </pre>
    </div>
  );
}

/**
 * The HTTP exchange the Worker captured: per attempt the request line and
 * its outcome, request headers, request body and response headers.
 * Credential-looking header values are masked with a "Show" button each.
 * `collapsible` wraps it in a disclosure named "Captured HTTP evidence".
 */
export function CapturedExchange({
  exchange,
  collapsible = true,
  open = false,
}: {
  exchange: HTTPExchange;
  collapsible?: boolean | undefined;
  open?: boolean | undefined;
}) {
  const count = exchange.attempts.length;
  const attempts = (
    <div className="issues-exchange-body">
      {exchange.attempts.map((attempt, index) => (
        // Attempts are positional: redirects and retries in order.
        <div className="issues-attempt" key={index}>
          <p className="issues-attempt-line">
            <span className="issues-attempt-number">
              {count === 1 ? "Request" : `Attempt ${index + 1} of ${count}`}
            </span>
            <code className="issues-attempt-request">
              {attempt.method} {attempt.url}
            </code>
            <span className="issues-attempt-outcome">
              {attemptOutcome(attempt)}
            </span>
          </p>
          <HeaderList label="Request headers" headers={attempt.headers} />
          <RequestBody base64={attempt.body_base64} />
          {attempt.response_headers === undefined ? null : (
            <HeaderList
              label="Response headers"
              headers={attempt.response_headers}
            />
          )}
        </div>
      ))}
      {exchange.response_body_evidence_id === undefined ? null : (
        <p className="issues-quiet">
          Response body: retained as evidence{" "}
          <code>{exchange.response_body_evidence_id}</code>
        </p>
      )}
    </div>
  );
  if (!collapsible)
    return (
      <div className="issues-exchange">
        <p className="issues-exchange-title">
          Request {exchange.request_id}
          {exchange.request_tag === "" ? null : (
            <span className="issues-quiet"> · {exchange.request_tag}</span>
          )}
        </p>
        {attempts}
      </div>
    );
  return (
    <details className="issues-exchange" open={open || undefined}>
      <summary>{`Captured HTTP evidence · request ${exchange.request_id}`}</summary>
      {attempts}
    </details>
  );
}

/**
 * Where the possible issue was found: authored source and web locations,
 * then the captured HTTP exchange, if any. Renders nothing without either.
 */
export function FindingLocations({
  document,
  exchangeOpen = false,
}: {
  document: Proposal;
  /** Start with the captured HTTP evidence expanded. Default false. */
  exchangeOpen?: boolean | undefined;
}) {
  const locations = document.locations ?? [];
  const exchange = document.http_exchange;
  if (locations.length === 0 && !exchange) return null;

  return (
    <section aria-label="Finding locations" className="issues-locations">
      {locations.length > 0 ? <LocationList locations={locations} /> : null}
      {exchange ? (
        <CapturedExchange exchange={exchange} open={exchangeOpen} />
      ) : null}
    </section>
  );
}
