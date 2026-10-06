/** Decorative stroke icons of the V3B decision bar and issue summary. */
export type DecisionIconName = "confirm" | "reject" | "unsure" | "next";

const PATHS: Readonly<Record<DecisionIconName, readonly string[]>> = {
  confirm: ["M5 12.5l4.2 4.2L19 7"],
  reject: ["M9.2 9.2l5.6 5.6M14.8 9.2l-5.6 5.6"],
  unsure: ["M9.7 9.6a2.4 2.4 0 0 1 4.6 1c0 1.7-2.3 2-2.3 3.4M12 16.8v.01"],
  next: ["M12 5v14M6.5 13.5L12 19l5.5-5.5"],
};

const CIRCLED: ReadonlySet<DecisionIconName> = new Set(["reject", "unsure"]);

export function DecisionIcon({
  name,
  size = 16,
}: {
  name: DecisionIconName;
  size?: number;
}) {
  return (
    <svg
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
      {CIRCLED.has(name) ? <circle cx="12" cy="12" r="8.5" /> : null}
      {PATHS[name].map((path) => (
        <path key={path} d={path} />
      ))}
    </svg>
  );
}
