import type { ReactNode } from "react";

const shapes = {
  play: <path d="m7 4 14 8-14 8V4Z" />,
  pause: <path d="M8 4v16M16 4v16" />,
  stop: <rect x="5" y="5" width="14" height="14" rx="2" />,
  home: <path d="m3 10 9-7 9 7M5 9v12h5v-6h4v6h5V9" />,
  projects: (
    <path d="M3 7V5a2 2 0 0 1 2-2h4l3 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />
  ),
  runs: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m10 8 6 4-6 4V8Z" />
    </>
  ),
  catalog: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  artifacts: (
    <path d="m3 7.5 9-4.5 9 4.5v9L12 21l-9-4.5v-9Zm0 0 9 4.5 9-4.5M12 12v9M7.5 5.25l9 4.5" />
  ),
  evals: (
    <path d="M9 3h6M10 3v6l-5.6 9A2 2 0 0 0 6.1 21h11.8a2 2 0 0 0 1.7-3L14 9V3M7.5 14h9" />
  ),
  operations: (
    <>
      <rect x="3" y="3" width="18" height="7" rx="2" />
      <rect x="3" y="14" width="18" height="7" rx="2" />
      <path d="M7 6.5h.01M7 17.5h.01M12 6.5h5M12 17.5h5" />
    </>
  ),
  settings: (
    <>
      <path d="M3 6h4m4 0h10M3 12h10m4 0h4M3 18h4m4 0h10" />
      <circle cx="9" cy="6" r="2" />
      <circle cx="15" cy="12" r="2" />
      <circle cx="9" cy="18" r="2" />
    </>
  ),
  logout: (
    <path d="M10 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h5M9 12h12m-4-4 4 4-4 4" />
  ),
  "add-config": (
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Zm0 0v6h6M8 15h8m-4-4v8" />
  ),
  "add-credential": (
    <>
      <circle cx="7" cy="8" r="4" />
      <path d="m10 11 8 8h3v-3l-8-8M18 3v6m-3-3h6" />
    </>
  ),
  binding: (
    <path d="M10 13a5 5 0 0 0 7 .1l3-3a5 5 0 0 0-7.1-7.1l-1.7 1.7M14 11a5 5 0 0 0-7-.1l-3 3a5 5 0 0 0 7.1 7.1l1.7-1.7" />
  ),
} satisfies Record<string, ReactNode>;

export function Icon({ name }: { name: keyof typeof shapes }) {
  return (
    <svg
      className="app-icon"
      viewBox="0 0 24 24"
      width="20"
      height="20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {shapes[name]}
    </svg>
  );
}
