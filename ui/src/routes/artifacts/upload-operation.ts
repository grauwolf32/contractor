import { useCallback, useEffect, useRef } from "react";

/**
 * The running upload or import of a dialog. Starting one aborts the one
 * before; closing the dialog or unmounting it aborts the running one, which
 * the Server may still have stored, so the dialog tells the user to look
 * before retrying. A refused or lost write is explained and never retried.
 */
export function useUploadOperation(onClose: () => void) {
  const operation = useRef<AbortController | null>(null);
  useEffect(() => () => operation.current?.abort(), []);
  const start = useCallback((): AbortSignal => {
    operation.current?.abort();
    const controller = new AbortController();
    operation.current = controller;
    return controller.signal;
  }, []);
  return {
    start,
    /** Whether the user closed the dialog while the operation ran. */
    aborted: () => operation.current?.signal.aborted === true,
    close: () => {
      operation.current?.abort();
      onClose();
    },
  };
}

/** What closing an upload dialog means, for a `noun` in the list behind it. */
export function uploadCancelNote(noun: "file" | "material"): string {
  return `Closing this dialog cancels a running upload. If you cancel it or lose the response, look for the ${noun} in the list before trying again: it may already be stored.`;
}
