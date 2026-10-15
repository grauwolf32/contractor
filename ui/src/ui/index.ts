// Shared V3B building blocks. Styles: ./ui.css (imported from src/main.tsx).
// API and examples: ./README.md.
export type { StatusTone } from "../app/status-tone";

export {
  PaneLayout,
  ListPane,
  DetailPane,
  DetailHeader,
  type BreadcrumbItem,
} from "./panes";
export {
  ListSection,
  ListRow,
  FilterChips,
  type FilterChipOption,
} from "./list";
export {
  StatusGlyph,
  StatusChip,
  ProgressSegments,
  type ProgressSegment,
} from "./status";
export { MethodChip, Kbd, IdChip } from "./chips";
export {
  TechnicalDetails,
  ActivityLog,
  EmptyState,
  type ActivityEntry,
} from "./content";
export {
  DecisionBar,
  type DecisionOption,
  type DecisionSeverity,
} from "./decision-bar";
export {
  useShortcuts,
  useListNavigation,
  isApplePlatform,
  modKeyLabel,
  type ListNavigationContainerProps,
} from "./shortcuts";
export { shortenId } from "./format";
