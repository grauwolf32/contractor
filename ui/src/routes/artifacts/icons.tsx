import type { ReactNode } from "react";

import type { MaterialKind } from "./kinds";

// 24×24 stroke shapes in the V3B mockup style (docs/design/ui/mockups/v3b).
const SHAPES = {
  source: <path d="M8.5 8l-4 4 4 4M15.5 8l4 4-4 4" />,
  api: (
    <path d="M8.5 4.5H7A1.5 1.5 0 0 0 5.5 6v4L4 12l1.5 2v4A1.5 1.5 0 0 0 7 19.5h1.5M15.5 4.5H17A1.5 1.5 0 0 1 18.5 6v4l1.5 2-1.5 2v4a1.5 1.5 0 0 1-1.5 1.5h-1.5" />
  ),
  architecture: (
    <>
      <rect x="3.5" y="4" width="7" height="5" rx="1" />
      <rect x="13.5" y="15" width="7" height="5" rx="1" />
      <path d="M10.5 6.5h4.5a2 2 0 0 1 2 2V15M7 9v5a3 3 0 0 0 3 3h3.5" />
    </>
  ),
  docs: (
    <>
      <path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12z" />
      <path d="M14 3.5V8h4.5M9.5 12.5h6M9.5 16h6" />
    </>
  ),
  diffs: (
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="2.5" />
      <path d="M8.5 6.5v5M6 9h5M13 15.5h5" />
    </>
  ),
  other: (
    <>
      <path d="M4 7.5h6l2 2h8v10H4z" />
      <path d="M4 7.5v-3h6l2 3" />
    </>
  ),
  git: (
    <>
      <circle cx="6" cy="5" r="2" />
      <circle cx="6" cy="19" r="2" />
      <circle cx="18" cy="6" r="2" />
      <path d="M6 7v10M18 8v2a5 5 0 0 1-5 5H6" />
    </>
  ),
  plus: <path d="M12 5.5v13M5.5 12h13" />,
  upload: (
    <>
      <path d="M12 15V4.5M7.5 9l4.5-4.5L16.5 9" />
      <path d="M4.5 15.5v3a1.5 1.5 0 0 0 1.5 1.5h12a1.5 1.5 0 0 0 1.5-1.5v-3" />
    </>
  ),
  download: (
    <>
      <path d="M12 4.5V15M7.5 10.5 12 15l4.5-4.5" />
      <path d="M4.5 19.5h15" />
    </>
  ),
  history: (
    <>
      <path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3" />
      <path d="M4.5 4.5v3h3M12 8v4l2.8 1.8" />
    </>
  ),
  lock: (
    <>
      <rect x="5" y="10.5" width="14" height="9.5" rx="2" />
      <path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5" />
    </>
  ),
  close: <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />,
} satisfies Record<string, ReactNode>;

export type MaterialIconName = keyof typeof SHAPES;

/** Decorative stroke icon; the text next to it carries the meaning. */
export function MaterialIcon({
  name,
  size = 16,
}: {
  name: MaterialIconName;
  size?: number;
}) {
  return (
    <svg
      className="materials-icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {SHAPES[name]}
    </svg>
  );
}

/** The icon of a material kind, as on the project's material chips. */
export function MaterialKindIcon({
  kind,
  size,
}: {
  kind: MaterialKind;
  size?: number;
}) {
  return <MaterialIcon name={kind} size={size ?? 16} />;
}
