import { parseUnifiedDiff } from "./diff-parser";
import { plural } from "../../../app/format";
import "./diff.css";

export default function DiffArtifactPreview({ source }: { source: string }) {
  const diff = parseUnifiedDiff(source);

  return (
    <section
      className="diff-artifact-preview"
      aria-label="Unified diff preview"
    >
      <div className="diff-artifact-summary" aria-label="Diff summary">
        <span>{plural(diff.files, "file")} changed</span>
        <span className="diff-addition-count">
          +{plural(diff.additions, "addition")}
        </span>
        <span className="diff-deletion-count">
          −{plural(diff.deletions, "deletion")}
        </span>
      </div>
      {diff.lines.length === 0 ? (
        <p className="diff-artifact-empty">No changes in this diff.</p>
      ) : (
        <div className="diff-lines" tabIndex={0}>
          {diff.lines.map((line, index) => (
            <div
              className={`diff-line diff-line-${line.kind}`}
              key={`${index}-${line.content}`}
            >
              <span className="diff-line-number" aria-hidden="true">
                {line.oldLine}
              </span>
              <span className="diff-line-number" aria-hidden="true">
                {line.newLine}
              </span>
              <code>{line.content === "" ? " " : line.content}</code>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
