import type { ReactNode, SyntheticEvent } from "react";
import { Link } from "react-router";

import type { components } from "../../api/generated/public";
import type { WorkflowRunState } from "../../api/runs";
import { compactDigest } from "../../app/format";
import { Icon } from "../../app/icon";
import { runStateLabel } from "../../app/vocabulary";
import { StatusChip } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";

type ArtifactRef = components["schemas"]["ExactArtifactRef"];

/** Controlled `<details>` state shared by the heavy Run detail sections. */
export type RunDisclosureProps = {
  open: boolean;
  onToggle: (event: SyntheticEvent<HTMLDetailsElement>) => void;
};

/** The rotating chevron of the Runs disclosures. */
export function DisclosureChevron() {
  return (
    <Icon
      name="chevron-right"
      size={13}
      strokeWidth={2}
      className="runs-section-chevron"
    />
  );
}

/**
 * One collapsible section of the Run page's Technical details: a native
 * `<details>` whose summary row carries the title (an h3, which the summary
 * content model allows), a quiet description and a count at the right.
 * `open` is controlled by the page so choices survive live remounts.
 */
export function RunSection({
  id,
  className,
  title,
  description,
  aside,
  children,
  ...disclosure
}: {
  id?: string | undefined;
  className?: string | undefined;
  title: string;
  description?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
} & RunDisclosureProps) {
  return (
    <details
      id={id}
      className={["runs-section", className].filter(Boolean).join(" ")}
      {...disclosure}
    >
      <summary className="runs-section-summary">
        <DisclosureChevron />
        <h3 className="runs-section-title">{title}</h3>
        {description === undefined ? null : (
          <span className="runs-section-description">{description}</span>
        )}
        {aside === undefined ? null : (
          <span className="runs-section-aside">{aside}</span>
        )}
      </summary>
      <div className="runs-section-body">{children}</div>
    </details>
  );
}

/** WorkflowRun state as a V3B status chip: glyph and word. */
export function RunStateChip({
  state,
  size = "sm",
}: {
  state: WorkflowRunState;
  size?: "sm" | "md";
}) {
  const { label, tone } = runStateLabel(state);
  return (
    <StatusChip tone={tone} size={size}>
      {label}
    </StatusChip>
  );
}

export function RunMetadataLabelChips({
  labels,
  empty = "none",
}: {
  labels: Readonly<Record<string, string>>;
  empty?: string;
}) {
  const entries = Object.entries(labels).sort(([left], [right]) =>
    left === right ? 0 : left < right ? -1 : 1,
  );
  if (entries.length === 0) {
    return <span className="muted-copy">{empty}</span>;
  }
  return (
    <span className="run-metadata-label-chips">
      {entries.map(([key, value]) => (
        <code className="run-metadata-label-chip" key={key} title={value}>
          <strong>{key}</strong>
          {value === "" ? null : (
            <>
              :<span>{value}</span>
            </>
          )}
        </code>
      ))}
    </span>
  );
}

export function RunArtifactRef({
  runId,
  slot,
  artifact,
}: {
  runId: string;
  slot?: string;
  artifact: ArtifactRef;
}) {
  return (
    <Link
      className="artifact-ref-link"
      to={artifactDetailPath({ kind: "run", id: runId }, artifact)}
    >
      {slot === undefined ? null : (
        <>
          <strong>{slot}</strong>{" "}
        </>
      )}
      <code>
        {artifact.namespace}/{artifact.name}@{artifact.revision}
      </code>
    </Link>
  );
}

/** A digest in its compact form; the full value is the hover title. */
export function Digest({ value }: { value: string }) {
  return (
    <code className="runs-digest" title={value}>
      {compactDigest(value)}
    </code>
  );
}

export function DefinitionList({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="runs-binding">
      <p className="runs-label">{title}</p>
      {children}
    </div>
  );
}
