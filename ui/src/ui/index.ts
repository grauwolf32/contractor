// Shared V3B building blocks. Styles: ./ui.css (imported from src/main.tsx).
// API and examples: ./README.md.
export type { StatusTone } from "../app/status-tone";

export {
  PaneLayout,
  ListPane,
  DetailPane,
  DetailHeader,
  type PaneLayoutProps,
  type ListPaneProps,
  type DetailPaneProps,
  type DetailHeaderProps,
  type BreadcrumbItem,
} from "./panes";
export {
  ListSection,
  ListRow,
  FilterChips,
  type ListSectionProps,
  type ListRowProps,
  type FilterChipsProps,
  type FilterChipOption,
} from "./list";
export {
  StatusGlyph,
  StatusChip,
  ProgressSegments,
  type StatusGlyphProps,
  type StatusChipProps,
  type ProgressSegmentsProps,
  type ProgressSegment,
} from "./status";
export { MethodChip, Kbd, IdChip, type IdChipProps } from "./chips";
export {
  TechnicalDetails,
  ActivityLog,
  EmptyState,
  type TechnicalDetailsProps,
  type ActivityLogProps,
  type ActivityEntry,
  type EmptyStateProps,
} from "./content";
export {
  DecisionBar,
  type DecisionBarProps,
  type DecisionOption,
  type DecisionSeverity,
  type DecisionSeverityOption,
  type DecisionRationale,
  type DecisionNext,
} from "./decision-bar";
export {
  useShortcuts,
  useListNavigation,
  isTextEntryTarget,
  isApplePlatform,
  modKeyLabel,
  type ShortcutHandler,
  type ShortcutBindings,
  type ShortcutOptions,
  type ListNavigation,
  type ListNavigationOptions,
  type ListNavigationContainerProps,
} from "./shortcuts";
export { shortenId, clockTime, type ClockTime } from "./format";
