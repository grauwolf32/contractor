import "./common.css";
import type { ReactNode } from "react";
import { Link } from "react-router";

import type { AllocationObservation } from "../../api/operations";
import { ErrorNotice } from "../../app/error-notice";
import { compactDigest, formatTimestamp } from "../../app/format";
import { StatusChip } from "../../ui";
import type { ExactConfigurationRef } from "./references";
import { operationsTone } from "./state";

export function ConfigurationRefLink({
  value,
}: {
  value: ExactConfigurationRef;
}) {
  return (
    <span className="ops-ref">
      <Link
        to={`/operations/configurations/${value.kind}/${encodeURIComponent(value.name)}/${encodeURIComponent(value.version)}`}
      >
        {value.name}@{value.version}
      </Link>
      <code title={value.digest}>{compactDigest(value.digest)}</code>
    </span>
  );
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
 * it turns with `.ui-tech[open]` and with each disclosure's own open rule.
 */
export function DisclosureChevron({
  size = 13,
}: {
  size?: number | undefined;
}) {
  return (
    <svg
      className="ui-tech-chevron"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M9.5 6l6 6-6 6" />
    </svg>
  );
}

/**
 * TechnicalDetails (src/ui) with a class name on the <details> (browser specs
 * find the diagnostics by it) and a toggle callback, which the primitive does
 * not take yet. Replace it with the primitive once it does.
 */
export function TechnicalDisclosure({
  className,
  summary,
  description,
  onToggle,
  children,
}: {
  className: string;
  summary: string;
  description?: ReactNode;
  onToggle?: ((open: boolean) => void) | undefined;
  children: ReactNode;
}) {
  return (
    <details
      className={`ui-tech ${className}`}
      onToggle={
        onToggle === undefined
          ? undefined
          : (event) => onToggle(event.currentTarget.open)
      }
    >
      <summary>
        <span className="ui-tech-label">
          <DisclosureChevron />
          {summary}
        </span>
        {description === undefined ? null : (
          <span className="ui-tech-description">{description}</span>
        )}
      </summary>
      <div className="ui-tech-body">{children}</div>
    </details>
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

/** Label and value pairs in a hairline grid ("at a glance"). */
export function Glance({
  items,
  label,
  className,
}: {
  items: readonly (readonly [ReactNode, ReactNode] | null | false)[];
  label?: string | undefined;
  className?: string | undefined;
}) {
  return (
    <dl
      className={
        className === undefined ? "ops-glance" : `ops-glance ${className}`
      }
      aria-label={label}
    >
      {items
        .filter((item): item is readonly [ReactNode, ReactNode] => !!item)
        .map(([term, value], index) => (
          <div key={index}>
            <dt>{term}</dt>
            <dd>{value}</dd>
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
    <Glance
      className="ops-metrics"
      items={[
        ...values.map(
          ([label, value]) => [label, value.toLocaleString()] as const,
        ),
        ["Reports", metrics.reportsComplete ? "complete" : "incomplete"],
        ["Truncated", metrics.truncated ? "yes" : "no"],
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
