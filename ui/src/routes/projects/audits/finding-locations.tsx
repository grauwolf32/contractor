import type { components } from "../../../api/generated/public";

type Proposal = components["schemas"]["FindingProposalDocument"];

export function FindingLocations({ document }: { document: Proposal }) {
  const locations = document.locations ?? [];
  const exchange = document.http_exchange;
  if (locations.length === 0 && !exchange) return null;

  return (
    <section aria-label="Finding locations">
      {locations.length > 0 ? (
        <ul>
          {locations.map((location, index) => (
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
                <span>
                  {location.method ? <code>{location.method} </code> : null}
                  <span>{location.url}</span>
                </span>
              )}
            </li>
          ))}
        </ul>
      ) : null}
      {exchange ? (
        <details>
          <summary>
            Captured HTTP evidence · request {exchange.request_id}
          </summary>
          {exchange.attempts.map((attempt, index) => (
            <details key={index}>
              <summary>
                {attempt.method} {attempt.url} ·{" "}
                {attempt.status ?? attempt.error}
              </summary>
              <p>Request headers</p>
              <pre>
                {attempt.headers
                  .map(({ name, value }) => `${name}: ${value}`)
                  .join("\n")}
              </pre>
              {attempt.body_base64 ? (
                <>
                  <p>Request body (base64)</p>
                  <pre>{attempt.body_base64}</pre>
                </>
              ) : null}
              {attempt.response_headers?.length ? (
                <>
                  <p>Response headers</p>
                  <pre>
                    {attempt.response_headers
                      .map(({ name, value }) => `${name}: ${value}`)
                      .join("\n")}
                  </pre>
                </>
              ) : null}
            </details>
          ))}
        </details>
      ) : null}
    </section>
  );
}
