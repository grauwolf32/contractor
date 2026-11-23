import { useCallback, useEffect, useState } from "react";

/** How long a status message such as "Decision recorded: …" stays. */
const ANNOUNCEMENT_MS = 8_000;

interface Announcement {
  /** The message for the live region, or "". */
  text: string;
  announce: (text: string) => void;
  clear: () => void;
}

/**
 * A short message for a polite live region. It stays until the caller clears
 * it on the user's next action, or for ANNOUNCEMENT_MS, so screen readers
 * read it even when refetched data replaces the view a moment later.
 */
export function useAnnouncement(): Announcement {
  // The id restarts the timer when the same text is announced again.
  const [message, setMessage] = useState<{ text: string; id: number } | null>(
    null,
  );
  useEffect(() => {
    if (message === null) return undefined;
    const timer = window.setTimeout(() => setMessage(null), ANNOUNCEMENT_MS);
    return () => window.clearTimeout(timer);
  }, [message]);
  const announce = useCallback((text: string) => {
    setMessage((previous) => ({ text, id: (previous?.id ?? 0) + 1 }));
  }, []);
  const clear = useCallback(() => setMessage(null), []);
  return { text: message?.text ?? "", announce, clear };
}
