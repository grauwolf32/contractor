import { useEffect, useRef, type ReactNode } from "react";

/** Native disclosure: regular Tab order, with dismissal and focus restoration. */
export function ActionMenu({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  const disclosure = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    function dismiss(event: PointerEvent | KeyboardEvent) {
      const element = disclosure.current;
      if (!element?.open || element.closest("[inert]")) return;
      if (event instanceof KeyboardEvent) {
        if (event.key !== "Escape") return;
        event.preventDefault();
        element.open = false;
        element.querySelector("summary")?.focus();
      } else if (
        event.target instanceof Node &&
        !element.contains(event.target)
      ) {
        element.open = false;
      }
    }
    document.addEventListener("keydown", dismiss);
    document.addEventListener("pointerdown", dismiss);
    return () => {
      document.removeEventListener("keydown", dismiss);
      document.removeEventListener("pointerdown", dismiss);
    };
  }, []);
  return (
    <details
      ref={disclosure}
      className="additional-actions project-actions-menu"
    >
      <summary aria-label={label} title={label}>
        ⋯
      </summary>
      <div className="additional-actions-menu">{children}</div>
    </details>
  );
}
