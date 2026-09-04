import { parseUnifiedDiff } from "./diff-parser";

function countLabel(count: number, singular: string, plural: string): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

export default function DiffArtifactPreview({ source }: { source: string }) {
  const diff = parseUnifiedDiff(source);

  return (
    <section
      className="diff-artifact-preview"
      aria-label="Unified diff preview"
    >
      <div className="diff-artifact-summary" aria-label="Diff summary">
        <span>{countLabel(diff.files, "file", "files")} changed</span>
        <span className="diff-addition-count">
          +{countLabel(diff.additions, "addition", "additions")}
        </span>
        <span className="diff-deletion-count">
          −{countLabel(diff.deletions, "deletion", "deletions")}
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
