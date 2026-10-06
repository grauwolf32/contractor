import type { MaterialKind } from "./material-kinds";

const ICON_PROPS = {
  viewBox: "0 0 24 24",
  width: 16,
  height: 16,
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
  focusable: false,
};

/** Decorative icon of a material kind, or of the project's live target. */
export function MaterialKindIcon({ kind }: { kind: MaterialKind | "target" }) {
  switch (kind) {
    case "sources":
      return (
        <svg {...ICON_PROPS}>
          <path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" />
        </svg>
      );
    case "openapi":
      return (
        <svg {...ICON_PROPS}>
          <circle cx="12" cy="12" r="2.5" />
          <path d="M12 3v6.5M12 14.5V21M3 12h6.5M14.5 12H21M5.6 5.6l4.6 4.6M13.8 13.8l4.6 4.6M18.4 5.6l-4.6 4.6M10.2 13.8l-4.6 4.6" />
        </svg>
      );
    case "likec4":
      return (
        <svg {...ICON_PROPS}>
          <rect x="3" y="4" width="7" height="5" rx="1" />
          <rect x="14" y="15" width="7" height="5" rx="1" />
          <path d="M10 6.5h5a2 2 0 0 1 2 2V15M7 9v5a3 3 0 0 0 3 3h4" />
        </svg>
      );
    case "docs":
      return (
        <svg {...ICON_PROPS}>
          <path d="M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h6" />
        </svg>
      );
    case "diffs":
      return (
        <svg {...ICON_PROPS}>
          <path d="M4 7h8M8 3v8M4 17h8M16 5h4M18 3v4M16 17h4" />
        </svg>
      );
    case "results":
      return (
        <svg {...ICON_PROPS}>
          <path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12zM14 3.5V8h4.5M9.2 14l2 2 3.8-3.8" />
        </svg>
      );
    case "target":
      return (
        <svg {...ICON_PROPS}>
          <circle cx="12" cy="12" r="8.5" />
          <path d="M3.5 12h17M12 3.5c2.4 2.4 3.5 5.2 3.5 8.5s-1.1 6.1-3.5 8.5c-2.4-2.4-3.5-5.2-3.5-8.5s1.1-6.1 3.5-8.5z" />
        </svg>
      );
    case "other":
      return (
        <svg {...ICON_PROPS}>
          <path d="M4 7.5h6l2 2h8v10H4zM4 7.5v-3h6l2 3" />
        </svg>
      );
  }
}
