import { useMemo } from "react";
import { TechnicalDetails } from "../../../ui";
import { SourcePreview } from "../source-preview";
import type { CodeEvidenceDocument } from "./code-evidence-document";
import { highlightCode } from "./code-highlight";
import "./code-evidence.css";

function coordinates(location: { startLine: number; endLine: number }) {
  return location.startLine === location.endLine
    ? `Line ${location.startLine}`
    : `Lines ${location.startLine}–${location.endLine}`;
}

export default function CodeEvidencePreview({
  document,
}: {
  document: CodeEvidenceDocument;
}) {
  const { source, workspace } = document;
  const highlighted = useMemo(
    () => highlightCode(workspace.text, workspace.file),
    [workspace.text, workspace.file],
  );
  const artifact = source.artifact;
  return (
    <section className="code-evidence" aria-label="Code evidence">
      <header className="code-evidence-heading">
        <strong>
          <code>{workspace.file}</code>
        </strong>
        <span className="ui-caption">
          {coordinates(workspace)} selected · saved workspace
        </span>
      </header>
      <div className="code-evidence-origin">
        <span className="ui-caption">Original source</span>
        <code>{source.file}</code>
        <span>{coordinates(source)}</span>
      </div>
      <SourcePreview
        source={workspace.text}
        startLine={workspace.snippetStartLine}
        selection={workspace}
        highlightedLines={highlighted}
        label="Saved code"
      />
      <p className="ui-caption">
        The saved workspace can include annotations. Original source coordinates
        are shown separately.
      </p>
      <TechnicalDetails>
        <dl className="code-evidence-details">
          <dt>Original material</dt>
          <dd>
            <code>
              {artifact.namespace}/{artifact.name}@{artifact.revision}
            </code>
          </dd>
          <dt>Original file digest</dt>
          <dd>
            <code>{source.digest}</code>
          </dd>
          <dt>Workspace file digest</dt>
          <dd>
            <code>{workspace.digest}</code>
          </dd>
        </dl>
      </TechnicalDetails>
    </section>
  );
}
