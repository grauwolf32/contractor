import "./common.css";
import type { ReactNode } from "react";
import { Link, type To } from "react-router";

import type { AllocationObservation } from "../../api/operations";
import { ErrorNotice } from "../../app/error-notice";
import { compactDigest, formatNumber, formatTimestamp } from "../../app/format";
import { Icon } from "../../app/icon";
import { Facts, StatusChip } from "../../ui";
import { configurationVersionPath } from "./llm-configurations/kinds";
import type { ExactConfigurationRef } from "./references";
import { operationsTone } from "./state";

/** An exact version: its name@version, linked to `to`, and short digest. */
export function RefLink({
  to,
  name,
  version,
  digest,
}: {
  to: To;
  name: string;
  version: string;
  digest: string;
}) {
  return (
    <span className="ops-ref">
      <Link to={to}>
        {name}@{version}
      </Link>
      <code title={digest}>{compactDigest(digest)}</code>
    </span>
  );
}

export function ConfigurationRefLink({
  value,
}: {
  value: ExactConfigurationRef;
}) {
  return <RefLink to={configurationVersionPath(value)} {...value} />;
}

export function OptionalTimestamp({ value }: { value: string | undefined }) {
  return value === undefined ? (
    <span className="ops-muted">Not observed</span>
  ) : (
    formatTimestamp(value)
  );
}

export function SafeReason({
  reason,
}: {
  reason: { code: string; retryable: boolean } | undefined;
}) {
  if (reason === undefined) {
    return <span className="ops-muted">None</span>;
  }
  return (
    <span className="ops-reason">
      <code>{reason.code}</code>
      <span>{reason.retryable ? "retryable" : "not retryable"}</span>
    </span>
  );
}

/** The Server's state word in a status chip (glyph + word, never colour alone). */
export function OperationsState({
  state,
  prefix,
}: {
  state: string;
  /** Says whose state it is, e.g. "Runtime" for an observed phase. */
  prefix?: string | undefined;
}) {
  return (
    <StatusChip tone={operationsTone(state)} size="sm">
      {prefix === undefined ? null : (
        <>
          <span className="ops-state-prefix">{prefix}</span>{" "}
        </>
      )}
      <span className="ops-state-word">
        {state === "ok" ? "OK" : state.replaceAll("_", " ")}
      </span>
    </StatusChip>
  );
}

/**
 * The chevron of TechnicalDetails (src/ui), for the area's own disclosures:
 * each disclosure turns it with its own open rule.
 */
export function DisclosureChevron({
  size = 13,
}: {
  size?: number | undefined;
}) {
  return (
    <Icon
      name="chevron-right"
      size={size}
      strokeWidth={2}
      className="ui-tech-chevron"
    />
  );
}

/** "Server-wide": the scope of a configuration or setting. */
export function ScopeChip({ children }: { children: ReactNode }) {
  return <span className="ops-scope">{children}</span>;
}

/**
 * A titled block of an Operations page: heading, an optional eyebrow and
 * description, and actions at the right.
 */
export function OpsSection({
  id,
  title,
  titleAs: Heading = "h2",
  eyebrow,
  description,
  aside,
  actions,
  className,
  children,
}: {
  /** Heading id; the section is labelled by it. */
  id: string;
  title: ReactNode;
  titleAs?: "h2" | "h3" | undefined;
  eyebrow?: ReactNode;
  description?: ReactNode;
  /** Quiet text at the right, such as a count. */
  aside?: ReactNode;
  actions?: ReactNode;
  className?: string | undefined;
  children?: ReactNode;
}) {
  return (
    <section
      className={
        className === undefined ? "ops-section" : `ops-section ${className}`
      }
      aria-labelledby={id}
    >
      <header className="ops-section-head">
        <div className="ops-section-heading">
          {eyebrow === undefined ? null : (
            <p className="ops-eyebrow">{eyebrow}</p>
          )}
          <Heading id={id} className="ops-section-title">
            {title}
          </Heading>
          {description === undefined ? null : (
            <div className="ops-section-description">{description}</div>
          )}
        </div>
        {aside === undefined && actions === undefined ? null : (
          <div className="ops-section-actions">
            {aside === undefined ? null : (
              <span className="ops-section-aside">{aside}</span>
            )}
            {actions}
          </div>
        )}
      </header>
      {children}
    </section>
  );
}

/** One term and value of FactRows, as in Facts. */
interface FactRow {
  term: ReactNode;
  /** undefined, null and false leave the row out. */
  value: ReactNode;
  key?: string;
}

/**
 * Label / value rows with a hairline between them; the labels share one
 * column (.ops-facts). Like Facts, a row whose value is undefined, null or
 * false is left out, and no row left renders nothing.
 * V385: needs shared Facts layout "rows" (Facts has grid, row and list).
 */
export function FactRows({
  items,
  className,
}: {
  items: readonly FactRow[];
  className?: string | undefined;
}) {
  const shown = items.filter(
    (row) =>
      row.value !== undefined && row.value !== null && row.value !== false,
  );
  if (shown.length === 0) return null;
  return (
    <dl
      className={
        className === undefined ? "ops-facts" : `ops-facts ${className}`
      }
    >
      {shown.map((row, index) => (
        <div key={row.key ?? (typeof row.term === "string" ? row.term : index)}>
          <dt>{row.term}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function MetricsSummary({
  metrics,
}: {
  metrics: AllocationObservation["metrics"];
}) {
  const values = [
    ["Model calls", metrics.modelCalls],
    ["Tool calls", metrics.toolCalls],
    ["Tool failures", metrics.toolFailures],
    ["Input tokens", metrics.inputTokens],
    ["Output tokens", metrics.outputTokens],
    ["Total tokens", metrics.totalTokens],
    ["Errors", metrics.errorCount],
  ] as const;
  return (
    <Facts
      className="ops-glance"
      items={[
        ...values.map(([term, value]) => ({
          term,
          value: formatNumber(value),
        })),
        {
          term: "Reports",
          value: metrics.reportsComplete ? "complete" : "incomplete",
        },
        { term: "Truncated", value: metrics.truncated ? "yes" : "no" },
      ]}
    />
  );
}

/** Draft validation errors followed by the failed publish request, if any. */
export function PublicationFeedback({
  title,
  errors,
  mutationError,
}: {
  title: string;
  errors: readonly string[];
  mutationError: unknown;
}) {
  return (
    <>
      {errors.length === 0 ? null : (
        <div className="notice notice-error" role="alert">
          <strong>{title}</strong>
          <ul>
            {errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      {mutationError === null ? null : <ErrorNotice error={mutationError} />}
    </>
  );
}
