import type { ArtifactMetadata } from "../../api/artifacts";
import { Icon } from "../../app/icon";
import { Button, EmptyState } from "../../ui";
import {
  materialCategory,
  type MaterialCategory,
} from "./material-library-filter";
import { MATERIAL_KIND_LABELS, materialKindOf } from "./kinds";

/** Counts and search explicitly describe the loaded page, never unseen material. */
export function MaterialLibraryFilters({
  items,
  category,
  search,
  partial,
  onCategory,
  onSearch,
}: {
  items: readonly ArtifactMetadata[];
  category: MaterialCategory;
  search: string;
  partial: boolean;
  onCategory: (value: MaterialCategory) => void;
  onSearch: (value: string) => void;
}) {
  const primary = ["all", "source", "api", "engagement", "docs"] as const;
  const more = ["architecture", "diffs", "other"] as const;
  const count = (kind: MaterialCategory) =>
    kind === "all"
      ? items.length
      : items.filter((item) => materialKindOf(item) === kind).length;
  return (
    <div className="materials-library-filters">
      <label className="materials-search">
        <Icon name="search" size={17} />
        <span className="ui-visually-hidden">
          {partial ? "Search this page" : "Search materials"}
        </span>
        <input
          type="search"
          placeholder={partial ? "Search this page" : "Search materials"}
          value={search}
          onChange={(event) => onSearch(event.target.value)}
        />
      </label>
      <div
        className="materials-category-chips"
        role="group"
        aria-label="Material category"
      >
        {primary.map((kind) => (
          <Button
            key={kind}
            variant="ghost"
            size="xs"
            aria-pressed={category === kind}
            onClick={() => onCategory(kind)}
          >
            {kind === "all"
              ? "All"
              : kind === "api"
                ? "API specs"
                : MATERIAL_KIND_LABELS[kind]}
            <span className="materials-category-count">{count(kind)}</span>
          </Button>
        ))}
        <select
          aria-label="More material categories"
          className="ui-control"
          value={more.some((kind) => kind === category) ? category : ""}
          onChange={(event) => onCategory(materialCategory(event.target.value))}
        >
          <option value="">More</option>
          {more.map((kind) => (
            <option key={kind} value={kind}>
              {MATERIAL_KIND_LABELS[kind]} · {count(kind)}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

export function NoMatchingMaterials({ onClear }: { onClear: () => void }) {
  return (
    <EmptyState title="No matching materials">
      <p>Try another name or category.</p>
      <Button size="sm" onClick={onClear}>
        Clear filters
      </Button>
    </EmptyState>
  );
}
