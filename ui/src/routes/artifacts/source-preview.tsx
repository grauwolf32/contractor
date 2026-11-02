import { Fragment, useMemo, useState } from "react";
import { Button } from "../../ui";
import "./source-preview.css";

// LIM-UI-18: preserve the full text without creating an element per line in
// newline-heavy one MiB payloads.
const MAXIMUM_NUMBERED_LINES = 4000;
function sourceLines(source: string) {
  const lines: { text: string; ending: string }[] = [];
  let start = 0;
  while (start < source.length) {
    if (lines.length === MAXIMUM_NUMBERED_LINES) return undefined;
    const end = source.indexOf("\n", start);
    if (end === -1) {
      lines.push({ text: source.slice(start), ending: "" });
      break;
    }
    const cr = end > start && source[end - 1] === "\r";
    lines.push({
      text: source.slice(start, cr ? end - 1 : end),
      ending: cr ? "\r\n" : "\n",
    });
    start = end + 1;
  }
  return lines;
}
export function SourcePreview({ source }: { source: string }) {
  const [wrap, setWrap] = useState(true);
  const lines = useMemo(() => sourceLines(source), [source]);
  return (
    <div className="materials-source-reader">
      <div className="materials-source-tools">
        <span>Source</span>
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
        tabIndex={0}
      >
        {lines === undefined
          ? source
          : lines.map((line, index) => (
              <Fragment key={index}>
                <span className="materials-source-line" data-line={index + 1}>
                  {line.text}
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
