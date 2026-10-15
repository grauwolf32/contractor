import type { ReactNode } from "react";

import { TechnicalDetails } from "./content";

interface Fact {
  term: ReactNode;
  /** null, undefined and false leave the fact out. */
  value: ReactNode;
  /** React key; defaults to the term when it is a string. */
  key?: string;
}

function present(fact: Fact): boolean {
  return (
    fact.value !== undefined && fact.value !== null && fact.value !== false
  );
}

function FactList({
  facts,
  layout,
  size,
  label,
  className,
}: {
  facts: readonly Fact[];
  layout: "grid" | "row" | "list";
  size: "sm" | "md";
  label: string | undefined;
  className: string | undefined;
}) {
  const shown = facts.filter(present);
  if (shown.length === 0) return null;
  return (
    <dl
      className={className === undefined ? "ui-facts" : `ui-facts ${className}`}
      data-layout={layout}
      data-size={size === "md" ? undefined : size}
      aria-label={label}
    >
      {shown.map((fact, index) => (
        <div
          key={fact.key ?? (typeof fact.term === "string" ? fact.term : index)}
        >
          <dt>{fact.term}</dt>
          <dd>{fact.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Term and value pairs: "Project crapi", "Started Oct 4, 23:51". Each pair
 * is a <div> of <dt> and <dd> inside one <dl>. Facts whose value is null,
 * undefined or false are left out, so a page lists what it knows without
 * conditionals. `technical` facts (IDs, revisions, digests) sit behind a
 * Technical details disclosure under the list.
 *
 * Layouts: `grid` (default) fills columns of at least 10rem, `row` wraps
 * pairs on one line, `list` stacks them.
 */
export function Facts({
  items,
  layout = "grid",
  size = "md",
  technical,
  technicalSummary,
  className,
  "aria-label": label,
}: {
  items: readonly Fact[];
  layout?: "grid" | "row" | "list" | undefined;
  /** `sm` for dense panes: smaller, lighter values. */
  size?: "sm" | "md" | undefined;
  technical?: readonly Fact[] | undefined;
  technicalSummary?: string | undefined;
  className?: string | undefined;
  "aria-label"?: string | undefined;
}) {
  const list = (
    <FactList
      facts={items}
      layout={layout}
      size={size}
      label={label}
      className={className}
    />
  );
  if (technical === undefined || !technical.some(present)) return list;
  return (
    <>
      {list}
      <TechnicalDetails
        {...(technicalSummary === undefined
          ? {}
          : { summary: technicalSummary })}
      >
        <FactList
          facts={technical}
          layout={layout}
          size="sm"
          label={undefined}
          className={undefined}
        />
      </TechnicalDetails>
    </>
  );
}
