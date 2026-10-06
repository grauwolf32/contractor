// Small stroke icons of the Start page (docs/design/ui/mockups/v3b/start.html).
// Decorative: every one is aria-hidden next to its words.
import type { ReactNode } from "react";

function Svg({
  size,
  strokeWidth,
  children,
}: {
  size: number;
  strokeWidth: number;
  children: ReactNode;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export function CheckIcon({ size = 14 }: { size?: number }) {
  return (
    <Svg size={size} strokeWidth={2.2}>
      <path d="M5 12.5l4.2 4.2L19 7" />
    </Svg>
  );
}

export function BulbIcon({ size = 14 }: { size?: number }) {
  return (
    <Svg size={size} strokeWidth={2}>
      <path d="M9 17.5h6M10 20.5h4" />
      <path d="M12 3.5a5.8 5.8 0 0 0-3.4 10.5c.6.5.9 1.1.9 1.8v.2h5v-.2c0-.7.3-1.3.9-1.8A5.8 5.8 0 0 0 12 3.5z" />
    </Svg>
  );
}

export function PlayIcon({ size = 16 }: { size?: number }) {
  return (
    <Svg size={size} strokeWidth={1.8}>
      <path d="M8 5.5v13l10.5-6.5z" />
    </Svg>
  );
}

export function ChevronIcon({ size = 13 }: { size?: number }) {
  return (
    <Svg size={size} strokeWidth={2}>
      <path d="M9.5 6l6 6-6 6" />
    </Svg>
  );
}

export function LockIcon({ size = 13 }: { size?: number }) {
  return (
    <Svg size={size} strokeWidth={2}>
      <rect x="5" y="10.5" width="14" height="9.5" rx="2" />
      <path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5" />
    </Svg>
  );
}
