import { useEffect, useRef, useState, type ReactNode } from "react";

import { shortenId } from "./format";

/** HTTP method in a small monospace chip ("GET", "POST", …). */
export function MethodChip({ method }: { method: string }) {
  return <span className="ui-method-chip">{method.toUpperCase()}</span>;
}

/** A key hint. Hide it from the accessible name when the control already has aria-keyshortcuts. */
export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="ui-kbd">{children}</kbd>;
}

const COPIED = "Copied";
const COPY_BY_HAND = "Press Ctrl+C to copy";

export interface IdChipProps {
  /** The full identifier; copied and shown on hover. */
  value: string;
  /** What the identifier is, for the copy button: "Copy " + label. */
  label: string;
  /** Visible text instead of the shortened value (e.g. "name@version"). */
  display?: string | undefined;
}

/**
 * Monospace identifier with a copy button. Shows `display`, or the value
 * shortened to its first 8 and last 4 characters; the full value is the
 * hover title. Copying uses the Clipboard API. When it is missing or
 * refused, the full value is selected in a hidden element and the chip asks
 * the user to press Ctrl+C. Results are announced politely.
 */
export function IdChip({ value, label, display }: IdChipProps) {
  const [message, setMessage] = useState("");
  const [selectRequest, setSelectRequest] = useState(0);
  const selectable = useRef<HTMLSpanElement>(null);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  useEffect(() => {
    if (selectRequest === 0) return;
    const node = selectable.current;
    const selection = window.getSelection();
    if (node === null || selection === null) return;
    selection.selectAllChildren(node);
  }, [selectRequest]);

  function announce(text: string, duration: number) {
    window.clearTimeout(timer.current);
    setMessage(text);
    timer.current = window.setTimeout(() => setMessage(""), duration);
  }

  function copyByHand() {
    setSelectRequest((request) => request + 1);
    announce(COPY_BY_HAND, 8000);
  }

  function copy() {
    const clipboard: Clipboard | undefined = navigator.clipboard;
    if (clipboard === undefined || typeof clipboard.writeText !== "function") {
      copyByHand();
      return;
    }
    try {
      // Called inside the click handler, so the browser sees a user gesture.
      clipboard.writeText(value).then(() => announce(COPIED, 2000), copyByHand);
    } catch {
      copyByHand();
    }
  }

  const copied = message === COPIED;
  return (
    <span className="ui-id">
      <span className="ui-id-chip">
        <span className="ui-id-chip-value" title={value}>
          {display ?? shortenId(value)}
        </span>
        <button
          type="button"
          className="ui-id-chip-copy"
          aria-label={`Copy ${label}`}
          title={`Copy ${label}`}
          onClick={copy}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            focusable="false"
          >
            {copied ? (
              <path d="M5 12.5l4.2 4.2L19 7" />
            ) : (
              <>
                <rect x="8.5" y="8.5" width="11" height="11" rx="2" />
                <path d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2" />
              </>
            )}
          </svg>
        </button>
      </span>
      <span className="ui-id-status" aria-live="polite">
        {message}
      </span>
      {selectRequest > 0 ? (
        <span ref={selectable} className="ui-id-selectable" aria-hidden="true">
          {value}
        </span>
      ) : null}
    </span>
  );
}
