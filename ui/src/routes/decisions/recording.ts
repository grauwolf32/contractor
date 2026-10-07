import { useEffect, useLayoutEffect, useRef } from "react";

/**
 * Tells the page when a decision component starts and stops recording, so
 * the page can keep it mounted meanwhile: `true` once a decision is sent,
 * `false` when the recorded decision's refetched subject arrived, after a
 * refused decision's refresh, or when the component unmounts mid-way.
 * Called only on changes.
 */
export function useRecordingCallback(
  recording: boolean,
  onRecording: ((recording: boolean) => void) | undefined,
): void {
  const latest = useRef(onRecording);
  useLayoutEffect(() => {
    latest.current = onRecording;
  });
  const reported = useRef(false);
  useEffect(() => {
    if (reported.current === recording) return;
    reported.current = recording;
    latest.current?.(recording);
  }, [recording]);
  useEffect(
    () => () => {
      if (!reported.current) return;
      reported.current = false;
      latest.current?.(false);
    },
    [],
  );
}
