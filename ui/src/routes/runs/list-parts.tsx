import { ContextLink } from "../../app/context-navigation";
import { compactId } from "../../app/format";
import { FilterChips, type FilterChipOption } from "../../ui";

/**
 * A labelled filter of the Runs lists: a short visible word and the chips.
 * The chip group carries the same name for assistive technology.
 */
export function RunsFilter<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly FilterChipOption<T>[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="runs-filter">
      <span className="runs-filter-label" aria-hidden="true">
        {label}
      </span>
      <FilterChips
        label={label}
        options={options}
        value={value}
        onChange={onChange}
      />
    </div>
  );
}

/**
 * The Run ID link of a list row: the compact identifier, named and titled by
 * the full Run ID. The Run page's back link returns to this list view.
 */
export function RunIdLink({
  runId,
  returnLabel,
}: {
  runId: string;
  returnLabel: string;
}) {
  return (
    <ContextLink
      returnLabel={returnLabel}
      className="runs-id-link"
      to={`/runs/${encodeURIComponent(runId)}`}
      aria-label={runId}
      title={runId}
    >
      {compactId(runId)}
    </ContextLink>
  );
}
