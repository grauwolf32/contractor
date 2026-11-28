import {
  Fragment,
  useMemo,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { Button } from "../../ui";
import { sourceLines } from "./source-lines";
import "./source-preview.css";

// LIM-UI-18: preserve the full text without creating an element per line in
// newline-heavy one MiB payloads.
export function SourcePreview({
  source,
  startLine = 1,
  selection,
  highlightedLines,
  label = "Source",
}: {
  source: string;
  startLine?: number | undefined;
  selection?: { startLine: number; endLine: number } | undefined;
  highlightedLines?: readonly ReactNode[] | undefined;
  label?: string | undefined;
}) {
  const [wrap, setWrap] = useState(true);
  const lines = useMemo(() => sourceLines(source), [source]);
  return (
    <div className="materials-source-reader">
      <div className="materials-source-tools">
        <span>{label}</span>
        <Button
          size="xs"
          variant="ghost"
          aria-pressed={wrap}
          onClick={() => setWrap(!wrap)}
        >
          Wrap lines
        </Button>
      </div>
      <pre
        className="artifact-preview materials-numbered-source"
        data-wrap={wrap}
        data-numbered={lines !== undefined}
        aria-label={label}
        style={
          {
            "--source-gutter": `${String(startLine + (lines?.length ?? 0)).length + 3}ch`,
          } as CSSProperties
        }
        tabIndex={0}
      >
        {lines === undefined
          ? source
          : lines.map((line, index) => (
              <Fragment key={index}>
                <span
                  className="materials-source-line"
                  data-line={startLine + index}
                  data-selected={
                    selection !== undefined &&
                    startLine + index >= selection.startLine &&
                    startLine + index <= selection.endLine
                      ? true
                      : undefined
                  }
                >
                  {highlightedLines?.[index] ?? line.text}
                </span>
                {line.ending}
              </Fragment>
            ))}
      </pre>
      {lines === undefined ? (
        <p className="ui-caption">
          Line numbers are hidden for this large file. The complete source is
          shown.
        </p>
      ) : null}
    </div>
  );
}
